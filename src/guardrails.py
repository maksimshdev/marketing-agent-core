"""Guardrails — жёсткие инварианты (спека §7). Чистые функции, без вызова адаптера.

Порядок применения здесь НЕ собирается — оркестратор (коммит 8) вызовет эти функции
в нужной последовательности на executor-шаге §8.8. Числа берутся из Config.

Инварианты:
  1. Шаг ставки ≤ MAX_STEP; bid ∈ [BID_FLOOR, BID_CEIL]      → clamp_bid, clamp_step
  2. Суммарный дневной расход ≤ B_MAX                          → enforce_budget_cap
  3. Нет паузы без гейта (§6.1)                                → проверяется в optimizer.prune
  4. ≤ MAX_CHANGES_PER_TICK изменений за тик                   → rate_limit
  5. Каждое действие логируется/обратимо                       → memory (§9)
  6. Kill-switch при CPA_acct > CATASTROPHE (после warmup, C2) → kill_switch
Конфликты (пауза>ставка>бюджет>масштаб)                        → resolve_conflicts
"""

from __future__ import annotations

import dataclasses

from src.config import Config
from src.types import Action, InvType, Proposal


# ---------------------------------------------------------------------------
# Инвариант 1: границы и шаг ставки
# ---------------------------------------------------------------------------

def clamp_bid(bid: float, inv_type: InvType, cfg: Config) -> float:
    """Ограничить ставку коридором [BID_FLOOR, BID_CEIL] по типу инвентаря (инв.1)."""
    return max(cfg.BID_FLOOR[inv_type], min(cfg.BID_CEIL[inv_type], bid))


def clamp_step(old_bid: float, new_bid: float, max_step: float) -> float:
    """Ограничить относительный шаг ставки за тик: |new/old − 1| ≤ max_step (инв.1).
    Возвращает new_bid, прижатый к допустимому коридору вокруг old_bid."""
    if old_bid <= 0:
        return new_bid
    lo = old_bid * (1.0 - max_step)
    hi = old_bid * (1.0 + max_step)
    return max(lo, min(hi, new_bid))


# ---------------------------------------------------------------------------
# Конфликты (пауза > ставка > бюджет > масштаб)
# ---------------------------------------------------------------------------

def resolve_conflicts(proposals: list[Proposal]) -> list[Proposal]:
    """Разрешить конфликты по каждому ref. Если для ref есть PAUSE — оставить ТОЛЬКО
    её (бюджет/ставка/масштаб для paused-арма бессмысленны). Иначе сохранить все
    предложения ref (bid/budget/scale — разные рычаги, не конфликтуют).
    Порядок: сначала по первому появлению ref, внутри ref — по приоритету (desc)."""
    order: list = []
    groups: dict = {}
    for p in proposals:
        if p.ref not in groups:
            groups[p.ref] = []
            order.append(p.ref)
        groups[p.ref].append(p)

    out: list[Proposal] = []
    for ref in order:
        group = groups[ref]
        pauses = [p for p in group if p.action == Action.PAUSE]
        kept = [pauses[0]] if pauses else group
        kept = sorted(kept, key=lambda p: p.priority, reverse=True)
        out.extend(kept)
    return out


# ---------------------------------------------------------------------------
# Инвариант 4: rate-limit
# ---------------------------------------------------------------------------

def rate_limit(proposals: list[Proposal], max_changes: int) -> list[Proposal]:
    """Не более max_changes изменений за тик (инв.4). При превышении — оставить
    высокоприоритетные (PAUSE/BID), пожертвовав масштабом, затем бюджетом."""
    if len(proposals) <= max_changes:
        return list(proposals)
    ordered = sorted(proposals, key=lambda p: p.priority, reverse=True)
    return ordered[:max_changes]


# ---------------------------------------------------------------------------
# Инвариант 2: потолок суммарного бюджета
# ---------------------------------------------------------------------------

def enforce_budget_cap(budget_proposals: list[Proposal], cfg: Config) -> list[Proposal]:
    """Σ дневных бюджетов ≤ B_MAX (инв.2). При превышении — пропорционально ужать все
    SET_BUDGET до суммы B_MAX. Не мутирует вход (возвращает новые Proposal)."""
    total = sum(p.new_value for p in budget_proposals if p.new_value is not None)
    if total <= cfg.B_MAX or total <= 0:
        return list(budget_proposals)
    factor = cfg.B_MAX / total
    out: list[Proposal] = []
    for p in budget_proposals:
        if p.new_value is None:
            out.append(p)
            continue
        scaled = p.new_value * factor
        out.append(dataclasses.replace(
            p, new_value=scaled,
            reason=p.reason + f" [ужато ×{factor:.3f} под B_MAX={cfg.B_MAX:.0f}]",
        ))
    return out


# ---------------------------------------------------------------------------
# Инвариант 6: kill-switch (+ warmup, C2)
# ---------------------------------------------------------------------------

def kill_switch(cpa_acct: float, tick: int, cfg: Config) -> bool:
    """True → заморозить изменения и алертить человека (§7.6). Неактивен первые
    WARMUP_TICKS тиков (холодный старт, правка C2)."""
    if tick < cfg.WARMUP_TICKS:
        return False
    return cpa_acct > cfg.CATASTROPHE_CPA
