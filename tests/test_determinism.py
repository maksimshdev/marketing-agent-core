"""Тесты оркестратора (§8): детерминизм прогона по SEED и инварианты прогона §7."""

from __future__ import annotations

import dataclasses

from src.config import DEFAULT_CONFIG as CFG
from src.memory import Memory
from src.mock_direct import MockYandexDirect
from src.orchestrator import Orchestrator
from src.types import Action, InvType


def _build(cfg=CFG) -> Orchestrator:
    adapter = MockYandexDirect(cfg)
    return Orchestrator(adapter, Memory(cfg), cfg)


def _signature(mem: Memory):
    # before/after достаточно для сверки исполненных значений
    return [(r.tick, r.ref.id, r.action.value, tuple(sorted(r.after.items())), r.reason)
            for r in mem.decisions]


# ---------------------------------------------------------------------------
# Детерминизм
# ---------------------------------------------------------------------------

def test_run_deterministic_same_seed():
    m1 = _build().run(15)
    m2 = _build().run(15)
    assert _signature(m1) == _signature(m2)
    assert m1.account_ticks == m2.account_ticks


def test_run_differs_with_different_seed():
    m1 = _build(CFG).run(15)
    m2 = _build(dataclasses.replace(CFG, SEED=777)).run(15)
    assert _signature(m1) != _signature(m2) or m1.account_ticks != m2.account_ticks


# ---------------------------------------------------------------------------
# Инварианты прогона §7
# ---------------------------------------------------------------------------

def test_invariant_budget_cap_each_tick():
    orch = _build()
    for t in range(20):
        orch.tick(t)
        total = sum(s.daily_budget for s in orch.memory.states.values())
        assert total <= CFG.B_MAX + 1e-6


def test_invariant_bid_step_within_max():
    mem = _build().run(30)
    for r in mem.decisions:
        if r.action == Action.SET_BID:
            old, new = r.before["bid"], r.after["bid"]
            if old > 0:
                assert abs(new / old - 1.0) <= CFG.MAX_STEP + 1e-9


def test_invariant_no_pause_without_gate():
    mem = _build().run(40)
    pauses = [r for r in mem.decisions if r.action == Action.PAUSE]
    for r in pauses:
        assert r.kpis_snapshot["cum_clicks"] >= CFG.N_MIN_PAUSE
        if r.ref.inv_type == InvType.NETWORK:
            assert r.kpis_snapshot["cum_imp"] >= CFG.N_MIN_IMP


def test_kill_switch_silent_in_warmup():
    # CATASTROPHE=0 → kill-switch сработал бы всегда, но warmup подавляет до WARMUP_TICKS
    cfg = dataclasses.replace(CFG, CATASTROPHE_CPA=0.0)
    orch = _build(cfg)
    for t in range(cfg.WARMUP_TICKS):
        recs = orch.tick(t)  # halt не должен случиться в warmup
    # ни одного kill-switch-алерта за warmup
    alerts = [r for r in orch.memory.decisions
              if r.action == Action.NOOP and "kill-switch" in r.reason]
    assert alerts == []


def test_kill_switch_triggers_after_warmup():
    cfg = dataclasses.replace(CFG, CATASTROPHE_CPA=0.0)
    orch = _build(cfg)
    orch.run(cfg.WARMUP_TICKS + 3)
    alerts = [r for r in orch.memory.decisions
              if r.action == Action.NOOP and "kill-switch" in r.reason]
    assert len(alerts) >= 1


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

def test_smoke_full_run():
    mem = _build().run(60)
    assert len(mem.account_ticks) == 60
    assert len(mem.decisions) > 0
