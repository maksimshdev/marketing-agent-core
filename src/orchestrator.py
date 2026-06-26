"""Цикл-оркестратор (спека §8). Склейка готовых слоёв §6 и инвариантов §7;
новой бизнес-логики нет. Числа из Config.

Один tick строго по порядку §8: ingest → recompute KPI → kill-switch → PRUNE → BID →
BUDGET → SCALE → executor (под guardrails) → лог DecisionRecord. Детерминизм:
bandit-RNG — отдельный поток f(SEED, tick), не коррелирующий с моком; refs/states
итерируются в стабильном порядке adapter.refs().
"""

from __future__ import annotations

import numpy as np

from src import guardrails, kpi, optimizer as opt
from src.adapter import PlatformAdapter
from src.config import Config
from src.memory import Memory
from src.types import (
    Action, DecisionRecord, EntityRef, InvType, Level, Proposal, Status,
)

# Маска отдельного потока RNG бандита (golden ratio constant) — раскоррелирует с моком.
_BANDIT_MASK = 0x9E3779B9

# Синтетический ref аккаунт-уровня для алертов kill-switch (§7.6).
_ACCOUNT_REF = EntityRef(level=Level.CAMPAIGN, id="ACCOUNT", parent_id=None,
                         inv_type=InvType.SEARCH)


class Orchestrator:
    """Оркестратор одного аккаунта поверх адаптера площадки (§8)."""

    def __init__(self, adapter: PlatformAdapter, memory: Memory, cfg: Config) -> None:
        self.adapter = adapter
        self.memory = memory
        self.cfg = cfg
        # стартовый набор армов из адаптера (те же объекты — шеринг с memory)
        memory.init_states(adapter.all_states())
        self._bandit_base = cfg.SEED ^ _BANDIT_MASK

    # ------------------------------------------------------------------
    # Один тик (§8)
    # ------------------------------------------------------------------

    def tick(self, t: int) -> list[DecisionRecord]:
        cfg, mem, adapter = self.cfg, self.memory, self.adapter
        refs = adapter.refs()

        # 1. метрики тика
        snaps = adapter.get_metrics(refs, t)
        # 2. накопить + аккаунт-агрегат
        mem.ingest(snaps)
        mem.snapshot_account_tick(t, snaps)
        last_spend_by_ref = {s.ref: s.spend for s in snaps}

        # 3. пересчёт KPI + переход EXPLORING/ACTIVE по гейту ставки
        for ref in refs:
            state = mem.states[ref]
            kpi.recompute_arm(state, cfg)
            if state.status != Status.PAUSED:
                state.status = (Status.ACTIVE if opt.passes_gate(state, cfg, "bid")
                                else Status.EXPLORING)

        # 4. kill-switch (после warmup, C2) → halt без изменений
        cpa = kpi.cpa_acct(mem.account_ticks, cfg.KILL_WINDOW)
        if guardrails.kill_switch(cpa, t, cfg):
            alert = DecisionRecord(
                tick=t, ref=_ACCOUNT_REF, action=Action.NOOP,
                before={}, after={},
                reason=f"kill-switch: CPA_acct={cpa:.0f}₽ > CATASTROPHE={cfg.CATASTROPHE_CPA:.0f}₽ "
                       f"— заморозка изменений, требуется человек",
                kpis_snapshot={"CPA_acct": cpa},
            )
            mem.log(alert)
            return []

        # 5. PRUNE
        pause_props = [p for p in (opt.prune(mem.states[r], cfg) for r in refs) if p]
        # 6. BID (не-paused)
        bid_props = [p for p in (opt.bid_controller(mem.states[r], cfg) for r in refs) if p]
        # 7. BUDGET (единый пул, отдельный RNG-поток f(SEED,tick))
        rng_tick = np.random.default_rng(self._bandit_base + t)
        budget_props = opt.allocate_budget([mem.states[r] for r in refs], cfg, rng_tick)
        # 8. SCALE
        scale_props = [p for p in
                       (opt.scale(mem.states[r], cfg, last_spend_by_ref.get(r, 0.0)) for r in refs)
                       if p]

        # 9. EXECUTOR под guardrails
        all_props = pause_props + bid_props + budget_props + scale_props
        all_props = guardrails.resolve_conflicts(all_props)
        budget_subset = [p for p in all_props if p.action == Action.SET_BUDGET]
        others = [p for p in all_props if p.action != Action.SET_BUDGET]
        budget_subset = guardrails.enforce_budget_cap(budget_subset, cfg)
        all_props = others + budget_subset
        all_props = guardrails.rate_limit(all_props, cfg.MAX_CHANGES_PER_TICK)

        executed: list[DecisionRecord] = []
        for p in all_props:
            rec = self._execute(p, t)
            if rec is not None:
                mem.log(rec)
                executed.append(rec)
        # 10. вернуть исполненные (tick инкрементит run)
        return executed

    # ------------------------------------------------------------------
    # Исполнение одного proposal под guardrails (§8.8) с записью before/after
    # ------------------------------------------------------------------

    def _execute(self, p: Proposal, t: int) -> DecisionRecord | None:
        cfg, mem, adapter = self.cfg, self.memory, self.adapter
        ref = p.ref
        state = mem.states[ref]
        inv = ref.inv_type
        kpis = {**p.kpis_snapshot, "cum_clicks": state.cum_clicks, "cum_imp": state.cum_imp}

        if p.action == Action.SET_BID:
            old_bid = state.bid
            new = guardrails.clamp_step(old_bid, p.new_value, cfg.MAX_STEP)
            new = guardrails.clamp_bid(new, inv, cfg)
            adapter.set_bid(ref, new)
            mem.update_state(ref, bid=new)
            before, after = {"bid": old_bid}, {"bid": new}

        elif p.action == Action.SET_BUDGET:
            old = state.daily_budget
            adapter.set_budget(ref, p.new_value)
            mem.update_state(ref, daily_budget=p.new_value)
            before, after = {"daily_budget": old}, {"daily_budget": p.new_value}

        elif p.action == Action.PAUSE:
            old_status, old_cap = state.status, state.arm_budget_cap
            adapter.set_status(ref, Status.PAUSED)
            adapter.set_budget(ref, 0.0)  # paused-арм не претендует на бюджет
            mem.update_state(ref, arm_budget_cap=0.0, daily_budget=0.0)
            before = {"status": old_status.value, "arm_budget_cap": old_cap}
            after = {"status": Status.PAUSED.value, "arm_budget_cap": 0.0}

        elif p.action == Action.SCALE:
            old_cap = state.arm_budget_cap
            mem.update_state(ref, arm_budget_cap=p.new_value)
            before, after = {"arm_budget_cap": old_cap}, {"arm_budget_cap": p.new_value}

        else:  # NOOP
            return None

        return DecisionRecord(tick=t, ref=ref, action=p.action, before=before,
                              after=after, reason=p.reason, kpis_snapshot=kpis)

    # ------------------------------------------------------------------
    def run(self, M: int) -> Memory:
        """Прогон M тиков. Возвращает память с полной историей (§9, §12)."""
        for t in range(M):
            self.tick(t)
        return self.memory
