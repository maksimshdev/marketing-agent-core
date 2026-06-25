"""Тесты мок-симулятора (§5): монотонности аукциона, budget-cap, детерминизм."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.config import DEFAULT_CONFIG as CFG
from src import mock_direct as M
from src.mock_direct import HiddenArmParams, MockYandexDirect, make_scenario
from src.types import ArmState, BidUnit, EntityRef, InvType, Level, Status

# Конфиг без шума — для стабильных проверок монотонностей (noise = exp(0) = 1.0)
CFG0 = dataclasses.replace(CFG, NOISE_SIGMA=0.0)


def _search_arm(bid: float, budget: float = 1e9) -> tuple[ArmState, HiddenArmParams]:
    ref = EntityRef(level=Level.KEYWORD, id="s", parent_id="ag", inv_type=InvType.SEARCH)
    st = ArmState(ref=ref, status=Status.EXPLORING, bid=bid, bid_unit=BidUnit.CPC,
                  daily_budget=budget)
    hp = HiddenArmParams(inv_type=InvType.SEARCH, value_per_conv=500.0, k=1.5,
                         base_demand=4000.0, true_ctr=0.08, true_cr=0.05, bid_ref=15.0)
    return st, hp


def _network_arm(cpm: float, budget: float = 1e9) -> tuple[ArmState, HiddenArmParams]:
    ref = EntityRef(level=Level.PLACEMENT, id="n", parent_id="ag", inv_type=InvType.NETWORK)
    st = ArmState(ref=ref, status=Status.EXPLORING, bid=cpm, bid_unit=BidUnit.CPM,
                  daily_budget=budget)
    hp = HiddenArmParams(inv_type=InvType.NETWORK, value_per_conv=500.0, k=1.5,
                         base_demand_imp=100000.0, true_ctr_net=0.006, true_cr_net=0.02,
                         cpm_ref=60.0)
    return st, hp


# ---------------------------------------------------------------------------
# Формы аукциона
# ---------------------------------------------------------------------------

def test_impression_share_grows_with_bid():
    prev = None
    for bid in (3, 8, 15, 30, 60):
        is_share = M.impression_share(bid, ref=15.0, k=1.5)
        assert 0.0 < is_share < 1.0
        if prev is not None:
            assert is_share > prev
        prev = is_share


def test_ratio_in_band_and_grows():
    # ratio = RATIO_BASE + RATIO_SLOPE*IS ∈ [0.6, 0.95]
    assert M.ratio(0.0, CFG) == pytest.approx(0.6)
    assert M.ratio(1.0, CFG) == pytest.approx(0.95)
    assert M.ratio(0.3, CFG) < M.ratio(0.7, CFG)


def test_cpc_grows_with_bid():
    prev = None
    for bid in (3.0, 8.0, 15.0, 30.0, 60.0):
        is_share = M.impression_share(bid, 15.0, 1.5)
        cpc = bid * M.ratio(is_share, CFG)
        if prev is not None:
            assert cpc > prev
        prev = cpc


def test_cpm_real_grows_with_bid():
    prev = None
    for cpm in (20.0, 40.0, 60.0, 120.0, 200.0):
        is_share = M.impression_share(cpm, 60.0, 1.5)
        cpm_real = cpm * M.ratio(is_share, CFG)
        if prev is not None:
            assert cpm_real > prev
        prev = cpm_real


# ---------------------------------------------------------------------------
# Монотонность объёма (показы не убывают с ростом ставки) — без шума
# ---------------------------------------------------------------------------

def test_search_impressions_monotonic_in_bid():
    rng = np.random.default_rng(0)
    prev = None
    for bid in (3.0, 8.0, 15.0, 30.0, 60.0):
        st, hp = _search_arm(bid)
        snap = M.simulate_search(st, hp, dow=1.0, noise=1.0, tick=0, cfg=CFG0, rng=rng)
        if prev is not None:
            assert snap.impressions >= prev
        prev = snap.impressions


def test_network_impressions_monotonic_in_bid():
    rng = np.random.default_rng(0)
    prev = None
    for cpm in (20.0, 40.0, 60.0, 120.0, 200.0):
        st, hp = _network_arm(cpm)
        snap = M.simulate_network(st, hp, dow=1.0, noise=1.0, tick=0, cfg=CFG0, rng=rng)
        if prev is not None:
            assert snap.impressions >= prev
        prev = snap.impressions


# ---------------------------------------------------------------------------
# Budget-cap: spend не превышает дневной лимит
# ---------------------------------------------------------------------------

def test_budget_cap_search():
    rng = np.random.default_rng(1)
    st, hp = _search_arm(bid=20.0, budget=5.0)  # очень маленький лимит
    snap = M.simulate_search(st, hp, dow=1.0, noise=1.0, tick=0, cfg=CFG, rng=rng)
    assert snap.spend <= 5.0 + 1e-9


def test_budget_cap_network():
    rng = np.random.default_rng(1)
    st, hp = _network_arm(cpm=100.0, budget=5.0)
    snap = M.simulate_network(st, hp, dow=1.0, noise=1.0, tick=0, cfg=CFG, rng=rng)
    assert snap.spend <= 5.0 + 1e-9


# ---------------------------------------------------------------------------
# Детерминизм по (SEED, tick)
# ---------------------------------------------------------------------------

def test_determinism_same_seed():
    m1 = MockYandexDirect(CFG)
    m2 = MockYandexDirect(CFG)
    refs1, refs2 = m1.refs(), m2.refs()
    s1 = m1.get_metrics(refs1, tick=7)
    s2 = m2.get_metrics(refs2, tick=7)
    assert s1 == s2


def test_determinism_repeat_same_tick():
    m = MockYandexDirect(CFG)
    refs = m.refs()
    assert m.get_metrics(refs, tick=3) == m.get_metrics(refs, tick=3)


def test_different_seed_differs():
    m1 = MockYandexDirect(CFG)
    m2 = MockYandexDirect(dataclasses.replace(CFG, SEED=999))
    s1 = m1.get_metrics(m1.refs(), tick=7)
    s2 = m2.get_metrics(m2.refs(), tick=7)
    assert s1 != s2


# ---------------------------------------------------------------------------
# Сценарий
# ---------------------------------------------------------------------------

def test_make_scenario_counts_and_defaults():
    states, hidden = make_scenario(CFG, np.random.default_rng(CFG.SEED))
    assert len(states) == CFG.N_SEARCH + CFG.N_NETWORK
    assert len(hidden) == len(states)
    n_search = sum(1 for r in states if r.inv_type == InvType.SEARCH)
    n_net = sum(1 for r in states if r.inv_type == InvType.NETWORK)
    assert (n_search, n_net) == (CFG.N_SEARCH, CFG.N_NETWORK)
    cap = CFG.ARM_BUDGET_CAP_FRAC * CFG.B
    for ref, st in states.items():
        assert st.status == Status.EXPLORING
        assert st.bid == CFG.BID_DEFAULT[ref.inv_type]
        assert st.arm_budget_cap == pytest.approx(cap)
        assert st.daily_budget == pytest.approx(CFG.B / (CFG.N_SEARCH + CFG.N_NETWORK))


def test_set_bid_validation_and_idempotency():
    m = MockYandexDirect(CFG)
    ref = next(r for r in m.refs() if r.inv_type == InvType.SEARCH)
    assert m.set_bid(ref, 25.0) is True
    assert m.get_state([ref])[0].bid == 25.0
    assert m.set_bid(ref, 25.0) is True  # идемпотентно
    with pytest.raises(Exception):
        m.set_bid(ref, 9999.0)  # вне [FLOOR, CEIL]


def test_get_true_params_access():
    m = MockYandexDirect(CFG)
    ref = m.refs()[0]
    hp = m.get_true_params(ref)
    assert hp.inv_type == ref.inv_type
    assert hp.value_per_conv > 0
