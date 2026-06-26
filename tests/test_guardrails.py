"""Тесты guardrails (§7) и памяти (§9): инварианты, конфликты, rate-limit,
budget-cap, kill-switch (warmup C2), накопление cum_*, согласованность с kpi.cpa_acct."""

from __future__ import annotations

import pytest

from src.config import DEFAULT_CONFIG as CFG
from src import guardrails as G, kpi
from src.memory import Memory, render_decision
from src.types import (
    Action, ArmState, BidUnit, DecisionRecord, EntityRef, InvType, Level,
    MetricSnapshot, Proposal, Status,
)

_COUNTER = iter(range(1_000_000))


def _ref(inv=InvType.SEARCH) -> EntityRef:
    level = Level.KEYWORD if inv == InvType.SEARCH else Level.PLACEMENT
    return EntityRef(level=level, id=f"x{next(_COUNTER)}", parent_id="ag", inv_type=inv)


def _prop(ref, action, value=0.0) -> Proposal:
    return Proposal(ref=ref, action=action, new_value=value, reason="r")


# ---------------------------------------------------------------------------
# Инвариант 1: границы и шаг ставки
# ---------------------------------------------------------------------------

def test_clamp_bid_both_types():
    assert G.clamp_bid(1.0, InvType.SEARCH, CFG) == CFG.BID_FLOOR[InvType.SEARCH]
    assert G.clamp_bid(999.0, InvType.SEARCH, CFG) == CFG.BID_CEIL[InvType.SEARCH]
    assert G.clamp_bid(12.0, InvType.SEARCH, CFG) == 12.0
    assert G.clamp_bid(5.0, InvType.NETWORK, CFG) == CFG.BID_FLOOR[InvType.NETWORK]
    assert G.clamp_bid(999.0, InvType.NETWORK, CFG) == CFG.BID_CEIL[InvType.NETWORK]
    assert G.clamp_bid(60.0, InvType.NETWORK, CFG) == 60.0


def test_clamp_step():
    # скачок вверх >20% ограничивается до old*1.2
    assert G.clamp_step(10.0, 20.0, CFG.MAX_STEP) == pytest.approx(12.0)
    # скачок вниз >20% ограничивается до old*0.8
    assert G.clamp_step(10.0, 2.0, CFG.MAX_STEP) == pytest.approx(8.0)
    # в пределах шага — без изменений
    assert G.clamp_step(10.0, 11.0, CFG.MAX_STEP) == pytest.approx(11.0)


# ---------------------------------------------------------------------------
# Конфликты и rate-limit
# ---------------------------------------------------------------------------

def test_resolve_conflicts_pause_wins():
    r = _ref()
    props = [_prop(r, Action.SET_BID, 11.0), _prop(r, Action.SET_BUDGET, 500.0),
             _prop(r, Action.PAUSE, None)]
    out = G.resolve_conflicts(props)
    assert len(out) == 1 and out[0].action == Action.PAUSE


def test_resolve_conflicts_keeps_non_conflicting():
    r = _ref()
    props = [_prop(r, Action.SET_BID, 11.0), _prop(r, Action.SET_BUDGET, 500.0)]
    out = G.resolve_conflicts(props)
    actions = {p.action for p in out}
    assert actions == {Action.SET_BID, Action.SET_BUDGET}


def test_resolve_conflicts_independent_refs():
    r1, r2 = _ref(), _ref()
    props = [_prop(r1, Action.PAUSE, None), _prop(r2, Action.SET_BID, 11.0)]
    out = G.resolve_conflicts(props)
    assert len(out) == 2


def test_rate_limit_truncates_by_priority():
    refs = [_ref() for _ in range(4)]
    props = [
        _prop(refs[0], Action.SCALE, 100.0),
        _prop(refs[1], Action.SET_BUDGET, 200.0),
        _prop(refs[2], Action.SET_BID, 11.0),
        _prop(refs[3], Action.PAUSE, None),
    ]
    out = G.rate_limit(props, max_changes=2)
    kept = {p.action for p in out}
    assert kept == {Action.PAUSE, Action.SET_BID}  # высокий приоритет остаётся
    assert len(out) == 2


def test_rate_limit_no_change_under_limit():
    props = [_prop(_ref(), Action.SET_BID, 11.0)]
    assert G.rate_limit(props, max_changes=40) == props


# ---------------------------------------------------------------------------
# Инвариант 2: потолок суммарного бюджета
# ---------------------------------------------------------------------------

def test_enforce_budget_cap_shrinks():
    refs = [_ref(), _ref()]
    props = [_prop(refs[0], Action.SET_BUDGET, 8000.0),
             _prop(refs[1], Action.SET_BUDGET, 8000.0)]  # Σ=16000 > B_MAX=12000
    out = G.enforce_budget_cap(props, CFG)
    total = sum(p.new_value for p in out)
    assert total == pytest.approx(CFG.B_MAX)
    assert out[0].new_value == pytest.approx(6000.0)  # пропорц.


def test_enforce_budget_cap_no_change():
    refs = [_ref(), _ref()]
    props = [_prop(refs[0], Action.SET_BUDGET, 4000.0),
             _prop(refs[1], Action.SET_BUDGET, 5000.0)]  # Σ=9000 ≤ B_MAX
    out = G.enforce_budget_cap(props, CFG)
    assert sum(p.new_value for p in out) == pytest.approx(9000.0)


# ---------------------------------------------------------------------------
# Инвариант 6: kill-switch (+ warmup C2)
# ---------------------------------------------------------------------------

def test_kill_switch_warmup_suppresses():
    # tick < WARMUP_TICKS → False даже при катастрофическом CPA (C2)
    assert G.kill_switch(9999.0, tick=CFG.WARMUP_TICKS - 1, cfg=CFG) is False


def test_kill_switch_triggers_after_warmup():
    assert G.kill_switch(CFG.CATASTROPHE_CPA + 1, tick=CFG.WARMUP_TICKS, cfg=CFG) is True


def test_kill_switch_below_threshold():
    assert G.kill_switch(CFG.CATASTROPHE_CPA - 1, tick=CFG.WARMUP_TICKS + 5, cfg=CFG) is False


# ---------------------------------------------------------------------------
# Память: накопление cum_* и согласованность с kpi.cpa_acct
# ---------------------------------------------------------------------------

def _arm_state(ref) -> ArmState:
    unit = BidUnit.CPC if ref.inv_type == InvType.SEARCH else BidUnit.CPM
    return ArmState(ref=ref, status=Status.EXPLORING, bid=12.0, bid_unit=unit,
                    daily_budget=300.0)


def test_memory_ingest_accumulates():
    r = _ref()
    mem = Memory(CFG)
    mem.init_states({r: _arm_state(r)})
    mem.ingest([MetricSnapshot(ref=r, tick=0, window="tick", impressions=1000,
                               clicks=50, spend=250.0, conversions=5, revenue=2500.0)])
    mem.ingest([MetricSnapshot(ref=r, tick=1, window="tick", impressions=2000,
                               clicks=80, spend=400.0, conversions=8, revenue=4000.0)])
    st = mem.states[r]
    assert (st.cum_imp, st.cum_clicks, st.cum_conv) == (3000, 130, 13)
    assert st.cum_spend == pytest.approx(650.0)
    assert st.cum_rev == pytest.approx(6500.0)


def test_memory_account_ticks_consistent_with_cpa_acct():
    r1, r2 = _ref(InvType.SEARCH), _ref(InvType.NETWORK)
    mem = Memory(CFG)
    mem.init_states({r1: _arm_state(r1), r2: _arm_state(r2)})
    for t, (sp1, cv1, sp2, cv2) in enumerate([(300.0, 2, 200.0, 1), (400.0, 4, 100.0, 0)]):
        snaps = [
            MetricSnapshot(ref=r1, tick=t, window="tick", spend=sp1, conversions=cv1),
            MetricSnapshot(ref=r2, tick=t, window="tick", spend=sp2, conversions=cv2),
        ]
        mem.snapshot_account_tick(t, snaps)
    # ключи 'spend'/'conversions' → kpi.cpa_acct работает напрямую
    assert set(mem.account_ticks[0].keys()) >= {"tick", "spend", "conversions"}
    cpa = kpi.cpa_acct(mem.account_ticks, kill_window=CFG.KILL_WINDOW)
    total_spend = 300.0 + 200.0 + 400.0 + 100.0
    total_conv = 2 + 1 + 4 + 0
    assert cpa == pytest.approx(total_spend / total_conv)


def test_memory_breakdown_by_type():
    r1, r2 = _ref(InvType.SEARCH), _ref(InvType.NETWORK)
    mem = Memory(CFG)
    mem.init_states({r1: _arm_state(r1), r2: _arm_state(r2)})
    snaps = [
        MetricSnapshot(ref=r1, tick=0, window="tick", spend=300.0, conversions=2),
        MetricSnapshot(ref=r2, tick=0, window="tick", spend=200.0, conversions=1),
    ]
    mem.snapshot_account_tick(0, snaps)
    snap = mem.tick_snapshots[0]
    assert snap["spend_search"] == pytest.approx(300.0)
    assert snap["spend_network"] == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# Объяснимость §10
# ---------------------------------------------------------------------------

def test_render_decision_format():
    r = _ref()
    rec = DecisionRecord(tick=23, ref=r, action=Action.SET_BID,
                         before={"bid": 14.0}, after={"bid": 11.8},
                         reason="CPA_est=210₽ vs target=200₽; снижаю ставку")
    text = render_decision(rec)
    assert text.startswith(f'[tick 23] search "{r.id}":')
    assert "причина:" in text
    assert "14.0" in text and "11.8" in text
