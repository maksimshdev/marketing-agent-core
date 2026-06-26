"""Мок-симулятор Яндекс Директа — источник «правды» (спека §5).

Две модели инвентаря: поиск (CPC) и РСЯ (CPM). Детерминирован: RNG тика =
np.random.default_rng(cfg.SEED + tick) — чистая функция от (SEED, tick), повторный
вызов того же тика даёт тот же результат. Недельная сезонность, мультипликативный
лог-нормальный шум. Опциональные хуки DRIFT (адаптивность) и FREQ_SAT (насыщение
частоты в РСЯ) — по умолчанию off.

Скрытые параметры (HiddenArmParams) агент НЕ видит; ядро к ним не обращается. Доступ
к истине — только get_true_params() (для oracle.py §12 и тестов).

Формы аукциона (impression_share, ratio, pos_uplift) вынесены модульными функциями,
чтобы oracle.py (C1) считал истинную CPA по тем же формулам.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Optional

import numpy as np

from src.adapter import InvalidValueError, PlatformAdapter, UnknownRefError
from src.config import Config
from src.types import (
    ArmState,
    BidUnit,
    EntityRef,
    InvType,
    Level,
    MetricSnapshot,
    Status,
    bid_unit_for,
)

# --- Параметры хуков (off по умолчанию; документированные формы) ---
# FREQ_SAT: насыщение частоты в РСЯ. freq_sat = 1/(1 + cum_imp/HALF) — при накоплении
# HALF показов охват падает вдвое. Активен только при cfg.FREQ_SAT == True.
_FREQ_SAT_HALF_IMP = 500_000.0
# DRIFT: с середины прогона (tick >= M_TICKS/2) истинная конверсия линейно падает,
# к концу прогона — до (1 - _DRIFT_MAX_REL) от исходной. Активен только при cfg.DRIFT.
_DRIFT_MAX_REL = 0.5


# ---------------------------------------------------------------------------
# Формы аукциона (переиспользуемы: мок + oracle §12/C1)
# ---------------------------------------------------------------------------

def impression_share(bid: float, ref: float, k: float) -> float:
    """IS(bid) = bid^k / (bid^k + ref^k) — доля показов, убывающая отдача (§5)."""
    bk = bid ** k
    return bk / (bk + ref ** k)


def ratio(is_share: float, cfg: Config) -> float:
    """cpc_ratio/cpm_ratio = RATIO_BASE + RATIO_SLOPE*IS (≈ вторая цена), ∈[0.6, 0.95]."""
    return cfg.RATIO_BASE + cfg.RATIO_SLOPE * is_share


def pos_uplift(is_share: float, cfg: Config) -> float:
    """Позиционный аплифт CTR для поиска: clamp(1 + UPLIFT_MAX*(IS-0.5), LO, HI)."""
    val = 1.0 + cfg.UPLIFT_MAX * (is_share - 0.5)
    return min(cfg.POS_UPLIFT_HI, max(cfg.POS_UPLIFT_LO, val))


def _freq_sat(cum_imp: int, cfg: Config) -> float:
    """Насыщение частоты (РСЯ). off → 1.0; on → 1/(1 + cum_imp/HALF)."""
    if not cfg.FREQ_SAT:
        return 1.0
    return 1.0 / (1.0 + cum_imp / _FREQ_SAT_HALF_IMP)


def _drift_factor(tick: int, cfg: Config) -> float:
    """Множитель дрейфа истинной конверсии (§5, хук). off → 1.0.
    on → 1.0 до середины прогона, далее линейно к (1 - _DRIFT_MAX_REL) к концу."""
    if not cfg.DRIFT:
        return 1.0
    half = cfg.M_TICKS / 2.0
    if tick < half:
        return 1.0
    progress = min(1.0, (tick - half) / half)  # 0..1 во второй половине
    return 1.0 - _DRIFT_MAX_REL * progress


# ---------------------------------------------------------------------------
# Скрытые параметры арма (агент НЕ видит)
# ---------------------------------------------------------------------------

@dataclass
class HiddenArmParams:
    """Истинные параметры арма (§5). Заполняются по типу инвентаря; поля другого
    типа остаются None. Доступ к ним — только oracle/тесты, ядро не использует."""

    inv_type: InvType
    value_per_conv: float
    k: float
    # search
    base_demand: Optional[float] = None
    true_ctr: Optional[float] = None
    true_cr: Optional[float] = None
    bid_ref: Optional[float] = None
    # network
    base_demand_imp: Optional[float] = None
    true_ctr_net: Optional[float] = None
    true_cr_net: Optional[float] = None
    cpm_ref: Optional[float] = None


# ---------------------------------------------------------------------------
# Генерация стартового сценария
# ---------------------------------------------------------------------------

def _sample(rng: np.random.Generator, rng_range: tuple[float, float]) -> float:
    lo, hi = rng_range
    return float(rng.uniform(lo, hi))


def make_scenario(
    cfg: Config, rng: np.random.Generator
) -> tuple[dict[EntityRef, ArmState], dict[EntityRef, HiddenArmParams]]:
    """Стартовый набор: N_SEARCH поисковых + N_NETWORK сетевых армов под одним
    аккаунтом. Скрытые параметры ~ U(min,max) из cfg.*_PARAM_RANGES (фикс. SEED через
    переданный rng) → набор намеренно неоднородный (часть армов убыточна: высокий
    ref / низкий cr).

    Стартовый daily_budget = B / (N_SEARCH + N_NETWORK) — равная нарезка пула (бандит
    переразложит её на каждом тике, §6.3). arm_budget_cap = ARM_BUDGET_CAP_FRAC * B (C3).
    """
    states: dict[EntityRef, ArmState] = {}
    hidden: dict[EntityRef, HiddenArmParams] = {}

    n_total = cfg.N_SEARCH + cfg.N_NETWORK
    start_budget = cfg.B / n_total
    cap = cfg.ARM_BUDGET_CAP_FRAC * cfg.B

    def add_arm(inv: InvType, idx: int) -> None:
        if inv == InvType.SEARCH:
            level, prefix, parent = Level.KEYWORD, "s", "ag_search"
            r = cfg.SEARCH_PARAM_RANGES
            hp = HiddenArmParams(
                inv_type=inv,
                value_per_conv=_sample(rng, r["value_per_conv"]),
                k=_sample(rng, r["k"]),
                base_demand=_sample(rng, r["base_demand"]),
                true_ctr=_sample(rng, r["true_ctr"]),
                true_cr=_sample(rng, r["true_cr"]),
                bid_ref=_sample(rng, r["bid_ref"]),
            )
        else:
            level, prefix, parent = Level.PLACEMENT, "n", "ag_net"
            r = cfg.NETWORK_PARAM_RANGES
            hp = HiddenArmParams(
                inv_type=inv,
                value_per_conv=_sample(rng, r["value_per_conv"]),
                k=_sample(rng, r["k"]),
                base_demand_imp=_sample(rng, r["base_demand_imp"]),
                true_ctr_net=_sample(rng, r["true_ctr_net"]),
                true_cr_net=_sample(rng, r["true_cr_net"]),
                cpm_ref=_sample(rng, r["cpm_ref"]),
            )
        ref = EntityRef(level=level, id=f"{prefix}{idx}", parent_id=parent, inv_type=inv)
        state = ArmState(
            ref=ref,
            status=Status.EXPLORING,
            bid=cfg.BID_DEFAULT[inv],
            bid_unit=bid_unit_for(inv),
            daily_budget=start_budget,
            arm_budget_cap=cap,
        )
        states[ref] = state
        hidden[ref] = hp

    for i in range(cfg.N_SEARCH):
        add_arm(InvType.SEARCH, i)
    for i in range(cfg.N_NETWORK):
        add_arm(InvType.NETWORK, i)

    return states, hidden


# ---------------------------------------------------------------------------
# Симуляция одного арма (модульные функции — переиспользуемы в тестах)
# ---------------------------------------------------------------------------

def simulate_search(
    state: ArmState, hp: HiddenArmParams, dow: float, noise: float,
    tick: int, cfg: Config, rng: np.random.Generator,
) -> MetricSnapshot:
    """Аукцион поиска (CPC), дословно §5. Шум/dow передаются снаружи; стохастика
    кликов/конверсий — через переданный rng."""
    bid = state.bid
    is_share = impression_share(bid, hp.bid_ref, hp.k)
    impressions = int(round(hp.base_demand * is_share * dow * noise))
    impressions = max(0, impressions)
    cpc = bid * ratio(is_share, cfg)

    ctr = min(1.0, hp.true_ctr * pos_uplift(is_share, cfg))
    clicks = int(rng.binomial(impressions, ctr)) if impressions > 0 else 0

    # budget-cap: при исчерпании показы стоп → пересчёт ОПЛАЧЕННЫХ кликов
    max_paid_clicks = floor(state.daily_budget / cpc) if cpc > 0 else clicks
    clicks_paid = min(clicks, max_paid_clicks)
    spend = clicks_paid * cpc

    cr = min(1.0, hp.true_cr * _drift_factor(tick, cfg))
    conversions = int(rng.binomial(clicks_paid, cr)) if clicks_paid > 0 else 0
    revenue = conversions * hp.value_per_conv

    return MetricSnapshot(
        ref=state.ref, tick=tick, window="tick",
        impressions=impressions, clicks=clicks_paid, spend=spend,
        conversions=conversions, revenue=revenue,
    )


def simulate_network(
    state: ArmState, hp: HiddenArmParams, dow: float, noise: float,
    tick: int, cfg: Config, rng: np.random.Generator,
) -> MetricSnapshot:
    """Аукцион РСЯ (CPM), дословно §5. cpm = текущая ставка арма (bid_unit=CPM)."""
    cpm = state.bid
    is_share = impression_share(cpm, hp.cpm_ref, hp.k)
    freq = _freq_sat(state.cum_imp, cfg)
    impressions_raw = int(round(hp.base_demand_imp * is_share * dow * noise * freq))
    impressions_raw = max(0, impressions_raw)
    cpm_real = cpm * ratio(is_share, cfg)

    # budget-cap по показам
    max_paid_imp = floor(state.daily_budget * 1000.0 / cpm_real) if cpm_real > 0 else impressions_raw
    impressions = min(impressions_raw, max_paid_imp)
    spend = impressions * cpm_real / 1000.0

    clicks = int(rng.binomial(impressions, hp.true_ctr_net)) if impressions > 0 else 0
    cr = min(1.0, hp.true_cr_net * _drift_factor(tick, cfg))
    conversions = int(rng.binomial(clicks, cr)) if clicks > 0 else 0
    revenue = conversions * hp.value_per_conv

    return MetricSnapshot(
        ref=state.ref, tick=tick, window="tick",
        impressions=impressions, clicks=clicks, spend=spend,
        conversions=conversions, revenue=revenue,
    )


# ---------------------------------------------------------------------------
# Адаптер мока
# ---------------------------------------------------------------------------

class MockYandexDirect(PlatformAdapter):
    """Реализация контракта §4 поверх мок-аукционов §5.

    Хранит раздельно видимые ArmState (рычаги bid/daily_budget/status — меняются set_*)
    и скрытые HiddenArmParams (истина, недоступна ядру).
    """

    def __init__(self, cfg: Config, states=None, hidden=None) -> None:
        self.cfg = cfg
        if states is None or hidden is None:
            rng = np.random.default_rng(cfg.SEED)
            states, hidden = make_scenario(cfg, rng)
        self._states: dict[EntityRef, ArmState] = states
        self._hidden: dict[EntityRef, HiddenArmParams] = hidden

    # --- контракт §4 ---

    def get_metrics(self, refs: list[EntityRef], tick: int) -> list[MetricSnapshot]:
        """Сырьё за тик. RNG = default_rng(SEED + tick): детерминизм по (SEED, tick).
        Шум/dow вычисляются на том же rng в порядке refs."""
        rng = np.random.default_rng(self.cfg.SEED + tick)
        dow = self.cfg.seasonality[tick % 7]
        out: list[MetricSnapshot] = []
        for ref in refs:
            state = self._require(ref)
            if state.status == Status.PAUSED:  # paused-арм не показывается/не тратит
                out.append(MetricSnapshot(ref=ref, tick=tick, window="tick"))
                continue
            hp = self._hidden[ref]
            noise = float(rng.lognormal(mean=0.0, sigma=self.cfg.NOISE_SIGMA))
            if ref.inv_type == InvType.SEARCH:
                snap = simulate_search(state, hp, dow, noise, tick, self.cfg, rng)
            else:
                snap = simulate_network(state, hp, dow, noise, tick, self.cfg, rng)
            out.append(snap)
        return out

    def get_state(self, refs: list[EntityRef]) -> list[ArmState]:
        return [self._require(ref) for ref in refs]

    def set_bid(self, ref: EntityRef, bid: float) -> bool:
        state = self._require(ref)
        lo, hi = self.cfg.BID_FLOOR[ref.inv_type], self.cfg.BID_CEIL[ref.inv_type]
        if not (lo <= bid <= hi):
            raise InvalidValueError(
                f"bid {bid} вне [{lo}, {hi}] для {ref.inv_type.value} (ref={ref.id})"
            )
        state.bid = bid  # идемпотентно
        return True

    def set_budget(self, ref: EntityRef, daily: float) -> bool:
        state = self._require(ref)
        if daily < 0:
            raise InvalidValueError(f"daily_budget {daily} < 0 (ref={ref.id})")
        state.daily_budget = daily  # идемпотентно
        return True

    def set_status(self, ref: EntityRef, status: Status) -> bool:
        state = self._require(ref)
        state.status = status  # идемпотентно
        return True

    # --- доступ к истине: ТОЛЬКО oracle/тесты, ядро не использует ---

    def get_true_params(self, ref: EntityRef) -> HiddenArmParams:
        """Истинные скрытые параметры арма. Предназначено для oracle.py (§12/C1) и
        тестов; ядро оптимизации к этому методу обращаться не должно."""
        if ref not in self._hidden:
            raise UnknownRefError(f"неизвестный ref: {ref.id}")
        return self._hidden[ref]

    def refs(self) -> list[EntityRef]:
        """Все зарегистрированные refs (порядок: search, затем network)."""
        return list(self._states.keys())

    def all_states(self) -> dict[EntityRef, ArmState]:
        """Внутренний словарь видимых ArmState (те же объекты, что использует адаптер).
        Memory разделяет эти объекты, чтобы ingest/update_state и set_* были консистентны."""
        return self._states

    # --- внутреннее ---

    def _require(self, ref: EntityRef) -> ArmState:
        if ref not in self._states:
            raise UnknownRefError(f"неизвестный ref: {ref.id}")
        return self._states[ref]
