"""Тесты ядра оптимизации (§6): гейт, контроллер ставок, bandit-бюджет, prune/scale."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import DEFAULT_CONFIG as CFG
from src import kpi, optimizer as opt
from src.types import Action, ArmState, BidUnit, EntityRef, InvType, Level, Status


_COUNTER = iter(range(1_000_000))


def _arm(inv=InvType.SEARCH, *, bid=12.0, budget=0.0, status=Status.ACTIVE,
         imp=0, clicks=0, conv=0, spend=0.0, cap=None) -> ArmState:
    level = Level.KEYWORD if inv == InvType.SEARCH else Level.PLACEMENT
    unit = BidUnit.CPC if inv == InvType.SEARCH else BidUnit.CPM
    ref = EntityRef(level=level, id=f"x{next(_COUNTER)}", parent_id="ag", inv_type=inv)
    st = ArmState(ref=ref, status=status, bid=bid, bid_unit=unit, daily_budget=budget,
                  cum_imp=imp, cum_clicks=clicks, cum_conv=conv, cum_spend=spend,
                  arm_budget_cap=(CFG.ARM_BUDGET_CAP_FRAC * CFG.B if cap is None else cap))
    return kpi.recompute_arm(st, CFG)


# ---------------------------------------------------------------------------
# §6.1 Гейт
# ---------------------------------------------------------------------------

def test_gate_bid_thresholds_search():
    assert opt.passes_gate(_arm(clicks=29), CFG, "bid") is False
    assert opt.passes_gate(_arm(clicks=30), CFG, "bid") is True


def test_gate_pause_scale_thresholds():
    assert opt.passes_gate(_arm(clicks=99), CFG, "pause") is False
    assert opt.passes_gate(_arm(clicks=100), CFG, "pause") is True
    assert opt.passes_gate(_arm(clicks=99), CFG, "scale") is False
    assert opt.passes_gate(_arm(clicks=100), CFG, "scale") is True


def test_gate_network_requires_imp():
    # кликов достаточно, но показов мало → гейт не пройден
    assert opt.passes_gate(_arm(InvType.NETWORK, clicks=50, imp=2999), CFG, "bid") is False
    assert opt.passes_gate(_arm(InvType.NETWORK, clicks=50, imp=3000), CFG, "bid") is True


# ---------------------------------------------------------------------------
# §6.2 Контроллер ставок
# ---------------------------------------------------------------------------

def test_controller_lowers_bid_when_cpa_high():
    st = _arm(bid=12.0, clicks=50)
    st.CPA_est = 400.0  # > TARGET=200
    p = opt.bid_controller(st, CFG)
    assert p is not None and p.action == Action.SET_BID
    assert p.new_value < 12.0  # factor < 1


def test_controller_raises_bid_when_cpa_low():
    st = _arm(bid=12.0, clicks=50)
    st.CPA_est = 100.0  # < TARGET
    p = opt.bid_controller(st, CFG)
    assert p is not None and p.new_value > 12.0


def test_controller_factor_clamped():
    # CPA_est огромен → ratio=0.5 → factor=0.5^0.5=0.707 → clamp до 1-MAX_STEP=0.8
    st = _arm(bid=12.0, clicks=50)
    st.CPA_est = 1e6
    p = opt.bid_controller(st, CFG)
    assert p.new_value == pytest.approx(12.0 * (1 - CFG.MAX_STEP))


def test_controller_bid_clamped_to_ceil():
    st = _arm(bid=58.0, clicks=50)
    st.CPA_est = 50.0  # сильно ниже target → ставка вверх, но упрётся в CEIL=60
    p = opt.bid_controller(st, CFG)
    assert p.new_value == pytest.approx(CFG.BID_CEIL[InvType.SEARCH])


def test_controller_no_emit_below_eps():
    st = _arm(bid=12.0, clicks=50)
    st.CPA_est = 210.0  # factor≈0.976 → Δ≈2.4% < 3%
    assert opt.bid_controller(st, CFG) is None


def test_controller_gate_not_passed():
    st = _arm(bid=12.0, clicks=10)
    st.CPA_est = 400.0
    assert opt.bid_controller(st, CFG) is None


# ---------------------------------------------------------------------------
# §6.3 Bandit-бюджет
# ---------------------------------------------------------------------------

def _exploit_arm(cr_conv, spend, clicks=200, imp=5000, cap=None):
    return _arm(InvType.SEARCH, status=Status.ACTIVE, clicks=clicks, imp=imp,
                conv=cr_conv, spend=spend, cap=cap)


def test_bandit_sum_equals_B_both_classes():
    # caps широкие → exploit-пул раздаётся полностью, сумма с explore-резервом = B
    rng = np.random.default_rng(42)
    exploit = [_exploit_arm(20, 1000, cap=CFG.B), _exploit_arm(25, 1200, cap=CFG.B)]
    explore = [_arm(status=Status.EXPLORING), _arm(status=Status.EXPLORING)]
    props = opt.allocate_budget(exploit + explore, CFG, rng)
    total = sum(p.new_value for p in props)
    assert total == pytest.approx(CFG.B, rel=1e-9)


def test_bandit_exploring_equal_shares():
    rng = np.random.default_rng(1)
    exploit = [_exploit_arm(20, 1000)]
    explore = [_arm(status=Status.EXPLORING), _arm(status=Status.EXPLORING)]
    props = opt.allocate_budget(exploit + explore, CFG, rng)
    expl = {p.ref.id: p.new_value for p in props
            if p.kpis_snapshot.get("kind") == "explore"}
    vals = list(expl.values())
    assert len(vals) == 2
    assert vals[0] == pytest.approx(vals[1])
    assert sum(vals) == pytest.approx(CFG.B * CFG.EXPLORE_FRAC)


def test_bandit_cap_clamp_and_overflow():
    # Только exploit (C5: весь B=10000 в exploit). A упирается в cap=1000, B забирает остаток.
    rng = np.random.default_rng(3)
    a = _exploit_arm(100, 200, clicks=200, cap=1000.0)   # cr=0.5, cpc=1 → высокий eff
    b = _exploit_arm(10, 2000, clicks=200, cap=CFG.B)    # cr=0.05, cpc=10 → низкий eff
    props = opt.allocate_budget([a, b], CFG, rng)
    by = {p.ref.id: p.new_value for p in props}
    assert by[a.ref.id] == pytest.approx(1000.0)          # capped
    assert by[b.ref.id] == pytest.approx(CFG.B - 1000.0)  # overflow к B


def test_bandit_no_arm_exceeds_cap():
    rng = np.random.default_rng(5)
    arms = [_exploit_arm(20, 1000, cap=2000.0), _exploit_arm(30, 800, cap=2000.0),
            _exploit_arm(15, 1500, cap=2000.0)]
    props = opt.allocate_budget(arms, CFG, rng)
    for p in props:
        assert p.new_value <= 2000.0 + 1e-9


def test_bandit_edge_no_exploit():
    # C5: все в разведке → весь B поровну
    rng = np.random.default_rng(7)
    arms = [_arm(status=Status.EXPLORING) for _ in range(4)]
    props = opt.allocate_budget(arms, CFG, rng)
    vals = [p.new_value for p in props]
    assert sum(vals) == pytest.approx(CFG.B)
    assert all(v == pytest.approx(CFG.B / 4) for v in vals)


def test_bandit_edge_no_exploring():
    # C5: нет EXPLORING → весь B в exploit (caps большие → сумма = B)
    rng = np.random.default_rng(9)
    arms = [_exploit_arm(20, 1000, cap=CFG.B), _exploit_arm(25, 1200, cap=CFG.B)]
    props = opt.allocate_budget(arms, CFG, rng)
    assert sum(p.new_value for p in props) == pytest.approx(CFG.B)


def test_bandit_deterministic():
    exploit = [_exploit_arm(20, 1000), _exploit_arm(25, 1200)]
    explore = [_arm(status=Status.EXPLORING)]
    p1 = opt.allocate_budget(exploit + explore, CFG, np.random.default_rng(11))
    p2 = opt.allocate_budget(exploit + explore, CFG, np.random.default_rng(11))
    assert [p.new_value for p in p1] == [p.new_value for p in p2]


def test_bandit_paused_excluded():
    rng = np.random.default_rng(13)
    arms = [_exploit_arm(20, 1000, cap=CFG.B),
            _arm(status=Status.PAUSED, clicks=200, conv=10, spend=2000)]
    props = opt.allocate_budget(arms, CFG, rng)
    paused_ref = arms[1].ref.id
    assert all(p.ref.id != paused_ref for p in props)
    assert sum(p.new_value for p in props) == pytest.approx(CFG.B)


# ---------------------------------------------------------------------------
# §6.4 Prune / Scale
# ---------------------------------------------------------------------------

def test_prune_when_confidently_expensive():
    st = _arm(clicks=150)
    st.CPA_lo = 400.0  # > 200*1.5 = 300
    p = opt.prune(st, CFG)
    assert p is not None and p.action == Action.PAUSE and p.new_value is None


def test_prune_gate_not_passed():
    st = _arm(clicks=50)  # < N_MIN_PAUSE
    st.CPA_lo = 400.0
    assert opt.prune(st, CFG) is None


def test_prune_not_when_cpa_lo_ok():
    st = _arm(clicks=150)
    st.CPA_lo = 250.0  # < 300
    assert opt.prune(st, CFG) is None


def test_scale_when_winner_at_limit():
    st = _arm(clicks=150, budget=1000.0, cap=3000.0)
    st.CPA_hi = 150.0  # < 200*0.8 = 160
    p = opt.scale(st, CFG, last_spend=950.0)  # ≥ 0.9*1000
    assert p is not None and p.action == Action.SCALE
    assert p.new_value == pytest.approx(min(3000.0 * 1.3, CFG.B))


def test_scale_no_when_not_spending():
    st = _arm(clicks=150, budget=1000.0, cap=3000.0)
    st.CPA_hi = 150.0
    assert opt.scale(st, CFG, last_spend=800.0) is None  # < 900


def test_scale_no_when_cpa_hi_high():
    st = _arm(clicks=150, budget=1000.0, cap=3000.0)
    st.CPA_hi = 180.0  # > 160
    assert opt.scale(st, CFG, last_spend=950.0) is None


def test_scale_no_when_cap_at_B():
    st = _arm(clicks=150, budget=1000.0, cap=CFG.B)
    st.CPA_hi = 150.0
    assert opt.scale(st, CFG, last_spend=950.0) is None  # new_cap не > cap
