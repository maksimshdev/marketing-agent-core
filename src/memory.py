"""Память и обучение (спека §9) + рендер объяснимости (§10).

In-memory хранилище (решение D3): текущее состояние армов, журнал решений,
поту-тиковые аккаунт-агрегаты (для kill-switch §7.6 и графиков §12). Обучение =
накопление статистики (cum_*) сдвигает апостериорные → bandit/Wilson сходятся.

Без вызова адаптера. Числа из Config.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import Config
from src.types import ArmState, DecisionRecord, EntityRef, InvType, MetricSnapshot


class Memory:
    """Хранилище состояний и истории решений (§9)."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.states: dict[EntityRef, ArmState] = {}
        self.decisions: list[DecisionRecord] = []
        # Ключи РОВНО 'spend'/'conversions' (+'tick') — для kpi.cpa_acct (коммит 4).
        self.account_ticks: list[dict[str, Any]] = []
        # Богатые срезы для дашборда §12 (разбивка поиск/РСЯ и т.п.).
        self.tick_snapshots: list[dict[str, Any]] = []

    # --- инициализация и накопление ---

    def init_states(self, states: dict[EntityRef, ArmState]) -> None:
        """Принять стартовый набор армов (из make_scenario)."""
        self.states = states

    def ingest(self, snapshots: list[MetricSnapshot]) -> None:
        """Накопить сырьё тика в cum_* счётчики состояний (+= за тик, §9)."""
        for snap in snapshots:
            st = self.states[snap.ref]
            st.cum_imp += snap.impressions
            st.cum_clicks += snap.clicks
            st.cum_conv += snap.conversions
            st.cum_spend += snap.spend
            st.cum_rev += snap.revenue

    def update_state(self, ref: EntityRef, **fields: Any) -> None:
        """Обновить видимые поля арма после исполнения (bid/daily_budget/status/cap)."""
        st = self.states[ref]
        for name, value in fields.items():
            setattr(st, name, value)

    # --- журнал и аккаунт-агрегаты ---

    def log(self, record: DecisionRecord) -> None:
        self.decisions.append(record)

    def snapshot_account_tick(self, tick: int, snapshots: list[MetricSnapshot]) -> None:
        """Сложить аккаунт-агрегат за тик. account_ticks — для kpi.cpa_acct (ключи
        'spend'/'conversions'); tick_snapshots — богатый срез для графиков §12."""
        spend = sum(s.spend for s in snapshots)
        conv = sum(s.conversions for s in snapshots)
        self.account_ticks.append({"tick": tick, "spend": spend, "conversions": conv})

        def _sum(field: str, inv: InvType) -> float:
            return sum(getattr(s, field) for s in snapshots if s.ref.inv_type == inv)

        self.tick_snapshots.append({
            "tick": tick,
            "spend": spend,
            "conversions": conv,
            "impressions": sum(s.impressions for s in snapshots),
            "clicks": sum(s.clicks for s in snapshots),
            "revenue": sum(s.revenue for s in snapshots),
            "spend_search": _sum("spend", InvType.SEARCH),
            "spend_network": _sum("spend", InvType.NETWORK),
            "conv_search": _sum("conversions", InvType.SEARCH),
            "conv_network": _sum("conversions", InvType.NETWORK),
        })

    # --- экспорт ---

    def to_frames(self) -> dict[str, pd.DataFrame]:
        """pandas-представление для дашборда (§12)."""
        decisions = pd.DataFrame([
            {
                "tick": r.tick,
                "id": r.ref.id,
                "inv_type": r.ref.inv_type.value,
                "action": r.action.value,
                "reason": r.reason,
            }
            for r in self.decisions
        ])
        return {
            "decisions": decisions,
            "account_ticks": pd.DataFrame(self.account_ticks),
            "tick_snapshots": pd.DataFrame(self.tick_snapshots),
        }


# ---------------------------------------------------------------------------
# Объяснимость (§10)
# ---------------------------------------------------------------------------

def render_decision(record: DecisionRecord) -> str:
    """Человекочитаемая причина решения (§10). Пример:

    [tick 23] search "s3": ставка 14.00→11.80 ₽
      причина: CPA_est=210₽ vs target=200₽; снижаю ставку 14.00→11.80 (factor 0.840)
    [tick 23] network "n1": ПАУЗА
      причина: CPA_lo=340₽ > target*1.5=300₽ при 5200 показах / 90 кликах — уверенно дорог
    [tick 23] search "s7": бюджет 300→480 ₽
      причина: bandit exploit: бюджет 300→480₽ (пул B=10000)
    """
    inv = record.ref.inv_type.value
    short = _action_short(record)
    return f'[tick {record.tick}] {inv} "{record.ref.id}": {short}\n  причина: {record.reason}'


def _action_short(record: DecisionRecord) -> str:
    a, after, before = record.action.value, record.after, record.before
    if record.action.value == "PAUSE":
        return "ПАУЗА"
    if record.action.value == "SET_BID":
        return f"ставка {before.get('bid', '?')}→{after.get('bid', '?')} ₽"
    if record.action.value == "SET_BUDGET":
        return f"бюджет {before.get('daily_budget', '?')}→{after.get('daily_budget', '?')} ₽"
    if record.action.value == "SCALE":
        return f"cap {before.get('arm_budget_cap', '?')}→{after.get('arm_budget_cap', '?')} ₽"
    return a
