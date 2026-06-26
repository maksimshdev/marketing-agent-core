"""Оракул-оптимум на истинных параметрах мока (приёмка §12, правка C1).

Доступ к истине — ТОЛЬКО через adapter.get_true_params(ref) (ядро так не делает).
Истинная CPA(ставка) считается через ТЕ ЖЕ формы аукциона, что и в моке
(mock_direct.impression_share / ratio / pos_uplift) — формулы не дублируются:

  search:  cpc = b * ratio(IS(b, bid_ref, k));  CPA(b) = cpc / true_cr
  network: cpm_real = c * ratio(IS(c, cpm_ref, k));            # C1: с множителем cpm_ratio
           CPA(c) = cpm_real / (1000 * true_ctr_net * true_cr_net)

Алгоритм (статический оптимум — верхняя планка конверсий):
  1. Для каждого арма — грид по [BID_FLOOR, BID_CEIL] (GRID точек). В каждой точке —
     ожидаемые (conv/день, spend/день, rev/день) при истинных параметрах, БЕЗ шума,
     dow = среднее seasonality (репрезентативный день).
  2. Каждый арм раскладывается на «слои» по лестнице ставок: базовый слой (floor-ставка)
     и инкременты вверх; маржинальная CPA слоя = Δspend/Δconv.
  3. Жадно льём дневной B в слои с наименьшей маржинальной CPA, пропуская слои с
     marginal_cpa > TARGET_CPA (так блендед CPA_acct ≤ TARGET гарантированно), пока
     Σspend ≤ B. Последний слой — частично (линейно).

При DRIFT=on истина смещается (см. mock); для честного эталона можно пересчитать
оракул в нужном тике — compute_oracle принимает tick (по умолчанию 0).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from src.config import Config
from src.mock_direct import (
    HiddenArmParams, MockYandexDirect, impression_share, pos_uplift, ratio,
)
from src.types import InvType

GRID = 50


@dataclass
class OracleResult:
    """Статический дневной оптимум аккаунта (верхняя планка)."""
    conv_per_day: float
    spend_per_day: float
    revenue_per_day: float
    cpa: float
    n_arms_used: int


def _operating_point(hp: HiddenArmParams, bid: float, cfg: Config, dow: float):
    """Ожидаемые (spend, conv, revenue) арма за день при ставке bid (без шума)."""
    if hp.inv_type == InvType.SEARCH:
        is_share = impression_share(bid, hp.bid_ref, hp.k)
        imp = hp.base_demand * is_share * dow
        cpc = bid * ratio(is_share, cfg)
        ctr = min(1.0, hp.true_ctr * pos_uplift(is_share, cfg))
        clicks = imp * ctr
        conv = clicks * hp.true_cr
        spend = clicks * cpc
    else:
        is_share = impression_share(bid, hp.cpm_ref, hp.k)
        imp = hp.base_demand_imp * is_share * dow
        cpm_real = bid * ratio(is_share, cfg)
        spend = imp * cpm_real / 1000.0
        clicks = imp * hp.true_ctr_net
        conv = clicks * hp.true_cr_net
    revenue = conv * hp.value_per_conv
    return spend, conv, revenue


def _arm_slices(hp: HiddenArmParams, cfg: Config, dow: float):
    """Лестница слоёв арма: список (marginal_cpa, dspend, dconv, drevenue) по росту ставки."""
    floor, ceil = cfg.BID_FLOOR[hp.inv_type], cfg.BID_CEIL[hp.inv_type]
    grid = np.linspace(floor, ceil, GRID)
    points = [_operating_point(hp, float(b), cfg, dow) for b in grid]

    slices = []
    prev_spend = prev_conv = prev_rev = 0.0
    for spend, conv, rev in points:
        dspend, dconv, drev = spend - prev_spend, conv - prev_conv, rev - prev_rev
        if dconv > 0 and dspend > 0:
            slices.append((dspend / dconv, dspend, dconv, drev))
            prev_spend, prev_conv, prev_rev = spend, conv, rev
    return slices


def compute_oracle(adapter: MockYandexDirect, cfg: Config, tick: int = 0) -> OracleResult:
    """Статический дневной оптимум при истинных параметрах (приёмка §12)."""
    dow = float(np.mean(cfg.seasonality))  # репрезентативный день
    refs = adapter.refs()
    arm_slices = {r: _arm_slices(adapter.get_true_params(r), cfg, dow) for r in refs}

    # куча: (marginal_cpa, ref_id, slice_idx); продвигаем слой только после взятия (лестница)
    heap = []
    for r in refs:
        if arm_slices[r]:
            heapq.heappush(heap, (arm_slices[r][0][0], r.id, r, 0))

    remaining = cfg.B
    conv = spend = revenue = 0.0
    used: set = set()
    while heap and remaining > 1e-9:
        mcpa, _id, ref, idx = heapq.heappop(heap)
        if mcpa > cfg.TARGET_CPA:  # слой неэффективен → арм здесь останавливается
            continue
        _, dspend, dconv, drev = arm_slices[ref][idx]
        if dspend <= remaining:
            spend += dspend
            conv += dconv
            revenue += drev
            remaining -= dspend
            used.add(ref)
            if idx + 1 < len(arm_slices[ref]):
                heapq.heappush(heap, (arm_slices[ref][idx + 1][0], ref.id, ref, idx + 1))
        else:  # частичный последний слой (линейно)
            frac = remaining / dspend
            spend += remaining
            conv += dconv * frac
            revenue += drev * frac
            used.add(ref)
            remaining = 0.0

    cpa = spend / conv if conv > 0 else float("inf")
    return OracleResult(conv_per_day=conv, spend_per_day=spend, revenue_per_day=revenue,
                        cpa=cpa, n_arms_used=len(used))
