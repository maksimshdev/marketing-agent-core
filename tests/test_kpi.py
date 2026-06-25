"""Тесты KPI-движка (§3): сырые/сглаженные формулы, Wilson, CPA/eff/границы,
защита от деления на ноль, согласованность recompute_arm."""

from __future__ import annotations

import math

import pytest

from src.config import DEFAULT_CONFIG as CFG
from src import kpi
from src.types import ArmState, BidUnit, EntityRef, InvType, Level, Status

INF = math.inf


def _arm(inv: InvType, *, imp=0, clicks=0, conv=0, spend=0.0) -> ArmState:
    level = Level.KEYWORD if inv == InvType.SEARCH else Level.PLACEMENT
    unit = BidUnit.CPC if inv == InvType.SEARCH else BidUnit.CPM
    ref = EntityRef(level=level, id="a", parent_id="g", inv_type=inv)
    return ArmState(
        ref=ref, status=Status.EXPLORING, bid=10.0, bid_unit=unit, daily_budget=300.0,
        cum_imp=imp, cum_clicks=clicks, cum_conv=conv, cum_spend=spend,
    )


# ---------------------------------------------------------------------------
# Сырые KPI
# ---------------------------------------------------------------------------

def test_raw_formulas():
    assert kpi.raw_ctr(50, 1000) == pytest.approx(0.05)
    assert kpi.raw_cpc(200.0, 40) == pytest.approx(5.0)
    assert kpi.raw_cpm(120.0, 10000) == pytest.approx(12.0)       # 120/10000*1000
    assert kpi.raw_cr(8, 40) == pytest.approx(0.2)
    assert kpi.raw_cpa(600.0, 4) == pytest.approx(150.0)
    assert kpi.roas(900.0, 300.0) == pytest.approx(3.0)
    assert kpi.romi(900.0, 300.0) == pytest.approx(2.0)           # (900-300)/300


def test_raw_zero_guards():
    assert kpi.raw_ctr(0, 0) == 0.0
    assert kpi.raw_cr(0, 0) == 0.0
    assert kpi.raw_cpc(10.0, 0) == INF
    assert kpi.raw_cpm(10.0, 0) == INF
    assert kpi.raw_cpa(10.0, 0) == INF
    assert kpi.roas(10.0, 0.0) == INF
    assert kpi.romi(10.0, 0.0) == INF


# ---------------------------------------------------------------------------
# Сглаженные оценки
# ---------------------------------------------------------------------------

def test_smoothed_estimates():
    # CR_hat = (cum_conv + a0)/(cum_clicks + a0 + b0); priors (1,1)
    assert kpi.smoothed_cr(8, 40, 1, 1) == pytest.approx(9 / 42)
    # CTR_hat = (cum_clicks + ac)/(cum_imp + ac + bc)
    assert kpi.smoothed_ctr(50, 1000, 1, 1) == pytest.approx(51 / 1002)
    # cold start → 0.5 при priors (1,1)
    assert kpi.smoothed_cr(0, 0, 1, 1) == pytest.approx(0.5)
    assert kpi.smoothed_ctr(0, 0, 1, 1) == pytest.approx(0.5)
    assert kpi.cpc_hat(200.0, 40) == pytest.approx(5.0)
    assert kpi.cpm_hat(120.0, 10000) == pytest.approx(12.0)
    assert kpi.cpc_hat(10.0, 0) == INF
    assert kpi.cpm_hat(10.0, 0) == INF


# ---------------------------------------------------------------------------
# Wilson-интервал
# ---------------------------------------------------------------------------

def test_wilson_reference_values():
    # Эталон Wilson 95% (z=1.96): значения сверены с R Hmisc::binconf / statsmodels.
    lo, hi = kpi.wilson_interval(1, 10, 1.96)
    assert lo == pytest.approx(0.01787575, abs=1e-6)
    assert hi == pytest.approx(0.40415639, abs=1e-6)
    lo2, hi2 = kpi.wilson_interval(50, 100, 1.96)
    assert lo2 == pytest.approx(0.40382983, abs=1e-6)
    assert hi2 == pytest.approx(0.59617017, abs=1e-6)


def test_wilson_edge_n_zero():
    assert kpi.wilson_interval(0, 0, 1.96) == (0.0, 1.0)


def test_wilson_narrows_with_n():
    # При фиксированной доле интервал сужается с ростом n
    prev = None
    for n in (10, 100, 1000, 10000):
        lo, hi = kpi.wilson_interval(n // 10, n, 1.96)  # p = 0.1
        width = hi - lo
        if prev is not None:
            assert width < prev
        prev = width


def test_wilson_bounds_in_unit_interval():
    for s, n in [(0, 5), (5, 5), (3, 7), (1, 1000)]:
        lo, hi = kpi.wilson_interval(s, n, 1.96)
        assert 0.0 <= lo <= hi <= 1.0


# ---------------------------------------------------------------------------
# CPA / eff / границы — search и network
# ---------------------------------------------------------------------------

def test_cpa_eff_bounds_search():
    cpa = kpi.cpa_est(InvType.SEARCH, cpc_hat=10.0, cr_hat=0.05)
    assert cpa == pytest.approx(200.0)                            # 10/0.05
    e = kpi.eff(InvType.SEARCH, cpc_hat=10.0, cr_hat=0.05)
    assert e == pytest.approx(0.005)                             # 0.05/10
    lo, hi = kpi.cpa_bounds(InvType.SEARCH, cpc_hat=10.0, cr_lo=0.02, cr_hi=0.08)
    assert lo == pytest.approx(125.0)                            # 10/0.08
    assert hi == pytest.approx(500.0)                            # 10/0.02
    assert lo <= cpa <= hi


def test_cpa_eff_bounds_network():
    cpa = kpi.cpa_est(InvType.NETWORK, cpm_hat=60.0, ctr_hat=0.01, cr_hat=0.02)
    assert cpa == pytest.approx(300.0)                           # 60/(1000*0.01*0.02)
    e = kpi.eff(InvType.NETWORK, cpm_hat=60.0, ctr_hat=0.01, cr_hat=0.02)
    assert e == pytest.approx(0.2 / 60.0)                        # 1000*0.01*0.02/60
    lo, hi = kpi.cpa_bounds(
        InvType.NETWORK, cpm_hat=60.0, ctr_lo=0.008, ctr_hi=0.012, cr_lo=0.015, cr_hi=0.025
    )
    assert lo == pytest.approx(200.0)                            # 60/(1000*0.012*0.025)
    assert hi == pytest.approx(500.0)                            # 60/(1000*0.008*0.015)
    assert lo <= cpa <= hi


# ---------------------------------------------------------------------------
# Защита от деления на ноль (холодный старт) — оба типа
# ---------------------------------------------------------------------------

def test_zero_protection_cold_arm_search():
    cpa = kpi.cpa_est(InvType.SEARCH, cpc_hat=INF, cr_hat=0.5)
    assert cpa == INF
    assert kpi.eff(InvType.SEARCH, cpc_hat=INF, cr_hat=0.5) == 0.0


def test_zero_protection_cold_arm_network():
    cpa = kpi.cpa_est(InvType.NETWORK, cpm_hat=INF, ctr_hat=0.5, cr_hat=0.5)
    assert cpa == INF
    assert kpi.eff(InvType.NETWORK, cpm_hat=INF, ctr_hat=0.5, cr_hat=0.5) == 0.0


def test_recompute_cold_arm_both_types():
    for inv in (InvType.SEARCH, InvType.NETWORK):
        st = kpi.recompute_arm(_arm(inv), CFG)  # все счётчики = 0
        assert st.CPA_est == INF
        # eff пересчитывается отдельно в §6.3; здесь проверяем согласованность хатов
        e = kpi.eff(inv, cpc_hat=st.CPC_hat, cpm_hat=st.CPM_hat,
                    ctr_hat=st.CTR_hat, cr_hat=st.CR_hat)
        assert e == 0.0


# ---------------------------------------------------------------------------
# recompute_arm: заполняет поля и согласован с поточечными функциями
# ---------------------------------------------------------------------------

def test_recompute_arm_search_consistency():
    st = _arm(InvType.SEARCH, imp=5000, clicks=200, conv=10, spend=2000.0)
    out = kpi.recompute_arm(st, CFG)
    assert out is st  # контракт: мутирует и возвращает тот же объект

    assert st.CR_hat == pytest.approx(kpi.smoothed_cr(10, 200, CFG.alpha0, CFG.beta0))
    assert st.CTR_hat == pytest.approx(kpi.smoothed_ctr(200, 5000, CFG.alpha_c, CFG.beta_c))
    assert st.CPC_hat == pytest.approx(kpi.cpc_hat(2000.0, 200))
    assert st.CPM_hat == pytest.approx(kpi.cpm_hat(2000.0, 5000))

    cr_lo, cr_hi = kpi.wilson_interval(10, 200, CFG.Z)
    assert (st.CR_lo, st.CR_hi) == pytest.approx((cr_lo, cr_hi))

    assert st.CPA_est == pytest.approx(
        kpi.cpa_est(InvType.SEARCH, cpc_hat=st.CPC_hat, cr_hat=st.CR_hat)
    )
    assert st.CPA_lo <= st.CPA_est <= st.CPA_hi


def test_recompute_arm_network_consistency():
    st = _arm(InvType.NETWORK, imp=100000, clicks=600, conv=12, spend=3000.0)
    out = kpi.recompute_arm(st, CFG)
    assert out is st

    assert st.CPA_est == pytest.approx(
        kpi.cpa_est(InvType.NETWORK, cpm_hat=st.CPM_hat, ctr_hat=st.CTR_hat, cr_hat=st.CR_hat)
    )
    assert st.CPA_lo <= st.CPA_est <= st.CPA_hi
    # все производные заполнены
    for fld in ("CTR_hat", "CR_hat", "CPC_hat", "CPM_hat", "CPA_est",
                "CTR_lo", "CTR_hi", "CR_lo", "CR_hi", "CPA_lo", "CPA_hi"):
        assert getattr(st, fld) is not None


# ---------------------------------------------------------------------------
# CPA аккаунта (скользящее окно)
# ---------------------------------------------------------------------------

def test_cpa_acct_window():
    history = [
        {"spend": 1000.0, "conversions": 2},
        {"spend": 1200.0, "conversions": 4},
        {"spend": 800.0, "conversions": 6},
    ]
    # окно 2 последних: (1200+800)/(4+6) = 200
    assert kpi.cpa_acct(history, kill_window=2) == pytest.approx(200.0)
    # всё окно: (1000+1200+800)/(2+4+6) = 250
    assert kpi.cpa_acct(history, kill_window=7) == pytest.approx(3000.0 / 12.0)


def test_cpa_acct_no_conversions():
    history = [{"spend": 500.0, "conversions": 0}]
    assert kpi.cpa_acct(history, kill_window=7) == INF
