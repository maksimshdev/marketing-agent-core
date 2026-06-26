"""Ядро оптимизации (спека §6). Четыре слоя — каждый ЧИСТАЯ функция:
вход (состояния + cfg [+ rng]) → выход Proposal/список Proposal. Без побочных
эффектов и без вызова адаптера (исполнение под guardrails — коммит 8). Числа из Config.

Слои:
  §6.1 passes_gate            — статистический гейт значимости
  §6.2 bid_controller         — демпфированный контроллер ставок → целевой CPA
  §6.3 allocate_budget        — Thompson-bandit, единый пул, per-arm cap (C3), края (C5)
  §6.4 prune / scale          — пауза лузерам / рост cap победителям

Разрешение конфликтов между слоями (пауза>ставка>бюджет>масштаб) здесь НЕ делается —
это работа executor'а (коммит 8); Proposal лишь несёт priority из ACTION_PRIORITY.
"""

from __future__ import annotations

import numpy as np

from src import kpi
from src.config import Config
from src.types import Action, ArmState, InvType, Proposal, Status


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _eff_point(state: ArmState, cfg: Config) -> float:
    """Точечная эффективность из сглаженных хатов (для отбора exploit-кандидатов)."""
    return kpi.eff(
        state.ref.inv_type,
        cpc_hat=state.CPC_hat,
        cpm_hat=state.CPM_hat,
        ctr_hat=state.CTR_hat,
        cr_hat=state.CR_hat,
    )


def _cap_of(state: ArmState, cfg: Config) -> float:
    """Текущий per-arm потолок (C3); None → стартовый ARM_BUDGET_CAP_FRAC*B."""
    if state.arm_budget_cap is None:
        return cfg.ARM_BUDGET_CAP_FRAC * cfg.B
    return state.arm_budget_cap


# ---------------------------------------------------------------------------
# §6.1 Статистический гейт
# ---------------------------------------------------------------------------

def passes_gate(state: ArmState, cfg: Config, purpose: str) -> bool:
    """Гейт значимости (§6.1). purpose ∈ {"bid","pause","scale"}.
    Для network дополнительно требуется cum_imp >= N_MIN_IMP (CTR достоверен)."""
    thresholds = {
        "bid": cfg.N_MIN_BID,
        "pause": cfg.N_MIN_PAUSE,
        "scale": cfg.N_MIN_SCALE,
    }
    if purpose not in thresholds:
        raise ValueError(f"неизвестный purpose: {purpose}")
    if state.cum_clicks < thresholds[purpose]:
        return False
    if state.ref.inv_type == InvType.NETWORK and state.cum_imp < cfg.N_MIN_IMP:
        return False
    return True


# ---------------------------------------------------------------------------
# §6.2 Контроллер ставок → целевой CPA
# ---------------------------------------------------------------------------

def bid_controller(state: ArmState, cfg: Config) -> Proposal | None:
    """Демпфированная пропорциональная подстройка ставки к TARGET_CPA (§6.2).
    Применяется только если пройден гейт ставки и арм не на паузе. Эмитит Proposal
    лишь при материальном изменении ставки (C4: |Δ|/old > CHANGE_EPS_BID)."""
    if state.status == Status.PAUSED or not passes_gate(state, cfg, "bid"):
        return None

    inv = state.ref.inv_type
    old_bid = state.bid
    ratio = _clamp(cfg.TARGET_CPA / state.CPA_est, 0.5, 2.0)
    factor = _clamp(ratio ** cfg.gamma, 1 - cfg.MAX_STEP, 1 + cfg.MAX_STEP)
    new_bid = _clamp(old_bid * factor, cfg.BID_FLOOR[inv], cfg.BID_CEIL[inv])

    if old_bid <= 0 or abs(new_bid - old_bid) / old_bid <= cfg.CHANGE_EPS_BID:
        return None

    direction = "снижаю" if new_bid < old_bid else "повышаю"
    reason = (
        f"CPA_est={state.CPA_est:.0f}₽ vs target={cfg.TARGET_CPA:.0f}₽; "
        f"{direction} ставку {old_bid:.2f}→{new_bid:.2f} (factor {factor:.3f})"
    )
    return Proposal(
        ref=state.ref, action=Action.SET_BID, new_value=new_bid, reason=reason,
        kpis_snapshot={"CPA_est": state.CPA_est, "factor": factor, "old_bid": old_bid},
    )


# ---------------------------------------------------------------------------
# §6.3 Bandit-аллокация дневного бюджета (Thompson, единый пул)
# ---------------------------------------------------------------------------

def _sample_eff(state: ArmState, cfg: Config, rng: np.random.Generator) -> float:
    """Сэмпл-эффективность (Thompson) для exploit-распределения (§6.3)."""
    cr_s = rng.beta(state.cum_conv + cfg.alpha0,
                    state.cum_clicks - state.cum_conv + cfg.beta0)
    if state.ref.inv_type == InvType.SEARCH:
        return cr_s / state.CPC_hat if state.CPC_hat > 0 else 0.0
    ctr_s = rng.beta(state.cum_clicks + cfg.alpha_c,
                     state.cum_imp - state.cum_clicks + cfg.beta_c)
    return 1000.0 * ctr_s * cr_s / state.CPM_hat if state.CPM_hat > 0 else 0.0


def _waterfill(
    arms: list[ArmState], weights: dict, pool: float, cfg: Config
) -> dict:
    """Распределить pool пропорц. weights с клампом по arm_budget_cap (C3).
    Излишек упёршихся переливается среди неупёршихся (итеративно). Если все упёрлись —
    остаток не форсим (недорасход допустим).

    Обход — в каноническом порядке по ref.id (не по set): суммирование float идёт в
    фиксированном порядке независимо от порядка входа и от per-process hash-рандомизации,
    что даёт побитовый детерминизм между процессами. Логика аллокации при этом неизменна.
    """
    budgets: dict = {}
    cap = {a.ref: _cap_of(a, cfg) for a in arms}
    uncapped = sorted((a.ref for a in arms), key=lambda r: r.id)
    remaining = pool
    for _ in range(5):
        if not uncapped:
            break
        total_w = sum(weights[r] for r in uncapped)
        if total_w <= 0:
            break
        newly = [r for r in uncapped if remaining * weights[r] / total_w > cap[r]]
        if not newly:
            for r in uncapped:
                budgets[r] = remaining * weights[r] / total_w
            uncapped = []
            break
        for r in newly:
            budgets[r] = cap[r]
            remaining -= cap[r]
        newly_set = set(newly)
        uncapped = [r for r in uncapped if r not in newly_set]
    # не сошлось за 5 итераций → раздать остаток по весам среди неупёршихся
    if uncapped:
        total_w = sum(weights[r] for r in uncapped)
        for r in uncapped:
            budgets[r] = remaining * weights[r] / total_w if total_w > 0 else remaining / len(uncapped)
    return budgets


def allocate_budget(
    states: list[ArmState], cfg: Config, rng: np.random.Generator
) -> list[Proposal]:
    """Единый бюджетный пул B для всех НЕ-paused армов (§6.3 + C3 cap + C5 края).

    Explore-резерв (B*EXPLORE_FRAC) поровну между EXPLORING; exploit-пул
    (B*(1-EXPLORE_FRAC)) — пропорц. сэмпл-эффективности с water-filling по cap.
    C5: нет EXPLORING → весь B в exploit; нет exploit → весь B поровну в explore.
    C4: Proposal(SET_BUDGET) эмитится лишь при материальном изменении бюджета.
    """
    arms = [s for s in states if s.status != Status.PAUSED]
    if not arms:
        return []

    exploit = [s for s in arms
               if s.status == Status.ACTIVE and passes_gate(s, cfg, "bid")
               and _eff_point(s, cfg) > 0]
    exploit_refs = {s.ref for s in exploit}
    explore = [s for s in arms if s.ref not in exploit_refs]

    budgets: dict = {}

    if not exploit:  # C5: все в разведке → весь B поровну между EXPLORING
        share = cfg.B / len(explore)
        budgets = {s.ref: share for s in explore}
    else:
        if not explore:  # C5: нет EXPLORING → весь B в exploit
            exploit_pool = cfg.B
        else:
            explore_reserve = cfg.B * cfg.EXPLORE_FRAC
            exploit_pool = cfg.B * (1 - cfg.EXPLORE_FRAC)
            share = explore_reserve / len(explore)
            for s in explore:
                budgets[s.ref] = share
        # exploit через Thompson + water-filling
        weights = {s.ref: _sample_eff(s, cfg, rng) for s in exploit}
        if sum(weights.values()) <= 0:  # вырожденный сэмпл → равный пул
            eq = exploit_pool / len(exploit)
            for s in exploit:
                budgets[s.ref] = eq
        else:
            budgets.update(_waterfill(exploit, weights, exploit_pool, cfg))

    # C4: материальный порог изменения бюджета
    proposals: list[Proposal] = []
    by_ref = {s.ref: s for s in arms}
    for ref, new_budget in budgets.items():
        old = by_ref[ref].daily_budget
        material = old == 0 or abs(new_budget - old) / old > cfg.CHANGE_EPS_BUDGET
        if not material:
            continue
        kind = "exploit" if ref in exploit_refs else "explore"
        reason = f"bandit {kind}: бюджет {old:.0f}→{new_budget:.0f}₽ (пул B={cfg.B:.0f})"
        proposals.append(Proposal(
            ref=ref, action=Action.SET_BUDGET, new_value=new_budget, reason=reason,
            kpis_snapshot={"old_budget": old, "kind": kind},
        ))
    return proposals


# ---------------------------------------------------------------------------
# §6.4 Отсечение (пауза) и масштабирование
# ---------------------------------------------------------------------------

def prune(state: ArmState, cfg: Config) -> Proposal | None:
    """Пауза уверенно дорогому арму (§6.4): гейт паузы пройден И CPA_lo (лучший случай)
    хуже TARGET_CPA*PAUSE_FACTOR. Исполнитель: status=PAUSED, arm_budget_cap→0 (C3)."""
    if not passes_gate(state, cfg, "pause"):
        return None
    threshold = cfg.TARGET_CPA * cfg.PAUSE_FACTOR
    if state.CPA_lo <= threshold:
        return None
    reason = (
        f"CPA_lo={state.CPA_lo:.0f}₽ > target*{cfg.PAUSE_FACTOR}={threshold:.0f}₽ "
        f"при {state.cum_imp} показах / {state.cum_clicks} кликах — уверенно дорог"
    )
    return Proposal(
        ref=state.ref, action=Action.PAUSE, new_value=None, reason=reason,
        kpis_snapshot={"CPA_lo": state.CPA_lo, "threshold": threshold},
    )


def scale(state: ArmState, cfg: Config, last_spend: float) -> Proposal | None:
    """Поднять per-arm cap уверенно прибыльному арму, упёршемуся в лимит (§6.4 + C3):
    гейт scale пройден И CPA_hi (худший случай) < TARGET_CPA*SCALE_FACTOR И последний
    дневной spend ≥ SCALE_SPEND_FRAC*daily_budget. new_cap = min(cap*(1+SCALE_UP), B)."""
    if state.status == Status.PAUSED or not passes_gate(state, cfg, "scale"):
        return None
    if state.CPA_hi >= cfg.TARGET_CPA * cfg.SCALE_FACTOR:
        return None
    if state.daily_budget <= 0 or last_spend < cfg.SCALE_SPEND_FRAC * state.daily_budget:
        return None
    cap = _cap_of(state, cfg)
    new_cap = min(cap * (1 + cfg.SCALE_UP), cfg.B)
    if new_cap <= cap:
        return None
    reason = (
        f"CPA_hi={state.CPA_hi:.0f}₽ < target*{cfg.SCALE_FACTOR}="
        f"{cfg.TARGET_CPA * cfg.SCALE_FACTOR:.0f}₽; упёрся в лимит "
        f"(spend {last_spend:.0f}≥{cfg.SCALE_SPEND_FRAC * state.daily_budget:.0f}); "
        f"cap {cap:.0f}→{new_cap:.0f}₽"
    )
    return Proposal(
        ref=state.ref, action=Action.SCALE, new_value=new_cap, reason=reason,
        kpis_snapshot={"CPA_hi": state.CPA_hi, "old_cap": cap, "last_spend": last_spend},
    )
