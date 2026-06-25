"""KPI-движок (спека §3): сырые/сглаженные метрики, Wilson-интервалы, CPA/eff/границы.

Чистые функции без состояния и побочных эффектов (кроме recompute_arm, который
по контракту мутирует переданный ArmState — см. его docstring). Числовые параметры
приходят через Config; в формулах ничего не хардкодим.

Защита от деления на ноль (для гейта/бандита §6, ответ C5):
  - CPA_est / CPA_lo / CPA_hi при нулевом знаменателе → +inf (арм «бесконечно дорог»);
  - eff при нулевом знаменателе → 0.0 (арм не идёт в exploit, остаётся в explore);
  - исключения не бросаем — оркестратор не должен падать на холодном старте.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from src.config import Config
from src.types import ArmState, InvType

INF = math.inf


# ---------------------------------------------------------------------------
# Вспомогательные безопасные деления
# ---------------------------------------------------------------------------

def _div(num: float, den: float, on_zero: float) -> float:
    """num/den с защитой: при den == 0 возвращает on_zero (не бросает)."""
    if den == 0:
        return on_zero
    return num / den


# ---------------------------------------------------------------------------
# Сырые KPI (§3) — на счётчиках за окно
# ---------------------------------------------------------------------------

def raw_ctr(clicks: int, impressions: int) -> float:
    """CTR = clicks / impressions."""
    return _div(clicks, impressions, on_zero=0.0)


def raw_cpc(spend: float, clicks: int) -> float:
    """CPC = spend / clicks. Нет кликов → +inf (цена клика «бесконечна»)."""
    return _div(spend, clicks, on_zero=INF)


def raw_cpm(spend: float, impressions: int) -> float:
    """CPM = spend / impressions * 1000. Нет показов → +inf."""
    return _div(spend * 1000.0, impressions, on_zero=INF)


def raw_cr(conversions: int, clicks: int) -> float:
    """CR = conversions / clicks."""
    return _div(conversions, clicks, on_zero=0.0)


def raw_cpa(spend: float, conversions: int) -> float:
    """CPA = spend / conversions. Нет конверсий → +inf."""
    return _div(spend, conversions, on_zero=INF)


def roas(revenue: float, spend: float) -> float:
    """ROAS = revenue / spend. Нет расхода → +inf."""
    return _div(revenue, spend, on_zero=INF)


def romi(revenue: float, spend: float) -> float:
    """ROMI = (revenue - spend) / spend. Нет расхода → +inf."""
    return _div(revenue - spend, spend, on_zero=INF)


# ---------------------------------------------------------------------------
# Сглаженные оценки (Beta-prior, §3)
# ---------------------------------------------------------------------------

def smoothed_cr(cum_conv: int, cum_clicks: int, alpha0: float, beta0: float) -> float:
    """CR_hat = (cum_conv + alpha0) / (cum_clicks + alpha0 + beta0)."""
    return (cum_conv + alpha0) / (cum_clicks + alpha0 + beta0)


def smoothed_ctr(cum_clicks: int, cum_imp: int, alpha_c: float, beta_c: float) -> float:
    """CTR_hat = (cum_clicks + alpha_c) / (cum_imp + alpha_c + beta_c)."""
    return (cum_clicks + alpha_c) / (cum_imp + alpha_c + beta_c)


def cpc_hat(cum_spend: float, cum_clicks: int) -> float:
    """CPC_hat = cum_spend / cum_clicks. Нет кликов → +inf."""
    return _div(cum_spend, cum_clicks, on_zero=INF)


def cpm_hat(cum_spend: float, cum_imp: int) -> float:
    """CPM_hat = cum_spend / cum_imp * 1000. Нет показов → +inf."""
    return _div(cum_spend * 1000.0, cum_imp, on_zero=INF)


# ---------------------------------------------------------------------------
# Wilson-интервал (§3)
# ---------------------------------------------------------------------------

def wilson_interval(successes: int, n: int, z: float) -> tuple[float, float]:
    """Доверительный интервал Уилсона для доли. n == 0 → (0.0, 1.0) (нет данных).

    center = (p + z²/2n) / (1 + z²/n)
    margin = z·sqrt(p(1-p)/n + z²/4n²) / (1 + z²/n)
    Результат клампится в [0, 1].
    """
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return (lo, hi)


# ---------------------------------------------------------------------------
# CPA / eff / границы (§3, §6.3) — формула зависит от типа инвентаря
# ---------------------------------------------------------------------------

def cpa_est(
    inv_type: InvType,
    *,
    cpc_hat: float | None = None,
    cpm_hat: float | None = None,
    ctr_hat: float | None = None,
    cr_hat: float | None = None,
) -> float:
    """CPA_est (§3). search: CPC_hat/CR_hat; network: CPM_hat/(1000·CTR_hat·CR_hat).
    Нулевой знаменатель → +inf."""
    if inv_type == InvType.SEARCH:
        return _div(cpc_hat, cr_hat, on_zero=INF)
    return _div(cpm_hat, 1000.0 * ctr_hat * cr_hat, on_zero=INF)


def eff(
    inv_type: InvType,
    *,
    cpc_hat: float | None = None,
    cpm_hat: float | None = None,
    ctr_hat: float | None = None,
    cr_hat: float | None = None,
) -> float:
    """Эффективность (конверсий на ₽, §6.3). search: CR_hat/CPC_hat;
    network: 1000·CTR_hat·CR_hat/CPM_hat. Нулевой/бесконечный знаменатель → 0.0."""
    if inv_type == InvType.SEARCH:
        if cpc_hat == 0 or cpc_hat == INF:
            return 0.0
        return _div(cr_hat, cpc_hat, on_zero=0.0)
    if cpm_hat == 0 or cpm_hat == INF:
        return 0.0
    return _div(1000.0 * ctr_hat * cr_hat, cpm_hat, on_zero=0.0)


def cpa_bounds(
    inv_type: InvType,
    *,
    cpc_hat: float | None = None,
    cpm_hat: float | None = None,
    ctr_lo: float | None = None,
    ctr_hi: float | None = None,
    cr_lo: float | None = None,
    cr_hi: float | None = None,
) -> tuple[float, float]:
    """Границы CPA (§3): lo = лучший случай (верхние CR/CTR), hi = худший (нижние).
    search:  CPA_lo = CPC_hat/CR_hi ;            CPA_hi = CPC_hat/CR_lo
    network: CPA_lo = CPM_hat/(1000·CTR_hi·CR_hi); CPA_hi = CPM_hat/(1000·CTR_lo·CR_lo)
    Нулевой знаменатель → +inf."""
    if inv_type == InvType.SEARCH:
        lo = _div(cpc_hat, cr_hi, on_zero=INF)
        hi = _div(cpc_hat, cr_lo, on_zero=INF)
        return (lo, hi)
    lo = _div(cpm_hat, 1000.0 * ctr_hi * cr_hi, on_zero=INF)
    hi = _div(cpm_hat, 1000.0 * ctr_lo * cr_lo, on_zero=INF)
    return (lo, hi)


# ---------------------------------------------------------------------------
# Пересчёт состояния арма и аккаунт-уровень
# ---------------------------------------------------------------------------

def recompute_arm(state: ArmState, cfg: Config) -> ArmState:
    """Пересчитывает все производные KPI арма из накопленных счётчиков (§3).

    КОНТРАКТ: мутирует переданный `state` на месте и возвращает его же
    (тот же объект) — удобно для in-memory стора (§9). Поля CTR_hat/CR_hat/
    CPC_hat/CPM_hat/CPA_est и границы CTR_lo/hi, CR_lo/hi, CPA_lo/hi.
    """
    inv = state.ref.inv_type

    state.CR_hat = smoothed_cr(state.cum_conv, state.cum_clicks, cfg.alpha0, cfg.beta0)
    state.CTR_hat = smoothed_ctr(state.cum_clicks, state.cum_imp, cfg.alpha_c, cfg.beta_c)
    state.CPC_hat = cpc_hat(state.cum_spend, state.cum_clicks)
    state.CPM_hat = cpm_hat(state.cum_spend, state.cum_imp)

    # Wilson: CR по кликам, CTR по показам
    state.CR_lo, state.CR_hi = wilson_interval(state.cum_conv, state.cum_clicks, cfg.Z)
    state.CTR_lo, state.CTR_hi = wilson_interval(state.cum_clicks, state.cum_imp, cfg.Z)

    state.CPA_est = cpa_est(
        inv,
        cpc_hat=state.CPC_hat,
        cpm_hat=state.CPM_hat,
        ctr_hat=state.CTR_hat,
        cr_hat=state.CR_hat,
    )
    state.CPA_lo, state.CPA_hi = cpa_bounds(
        inv,
        cpc_hat=state.CPC_hat,
        cpm_hat=state.CPM_hat,
        ctr_lo=state.CTR_lo,
        ctr_hi=state.CTR_hi,
        cr_lo=state.CR_lo,
        cr_hi=state.CR_hi,
    )
    return state


def cpa_acct(history: Sequence[Mapping[str, float]], kill_window: int) -> float:
    """Скользящий CPA аккаунта за последние kill_window тиков (§3, kill-switch §7.6).

    `history` — последовательность поту-тиковых агрегатов аккаунта, каждый — Mapping
    с ключами 'spend' и 'conversions'. Берутся последние `kill_window` записей;
    CPA_acct = Σspend / Σconversions. Нет конверсий → +inf.
    """
    window = history[-kill_window:] if kill_window > 0 else list(history)
    total_spend = sum(rec["spend"] for rec in window)
    total_conv = sum(rec["conversions"] for rec in window)
    return _div(total_spend, total_conv, on_zero=INF)
