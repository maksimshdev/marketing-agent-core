"""Конфигурация — единственный источник истины по ЧИСЛОВЫМ значениям (спека §11).

Спека символьная; все числа живут здесь. Один frozen-dataclass `Config` + готовый
экземпляр `DEFAULT_CONFIG`. Логики нет — данные + валидация инвариантов значений
в `__post_init__`. Значения соответствуют зафиксированным дефолтам (LOCKED).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.types import InvType


@dataclass(frozen=True)
class Config:
    """Все параметры ядра и мока. Доступ к ставочным границам — по InvType через
    dict-поля BID_FLOOR/BID_CEIL/BID_DEFAULT (раздельно search/network)."""

    # --- Цель и бюджет (§11, §6) ---
    TARGET_CPA: float = 200.0
    B: float = 10000.0                 # дневной бюджет (пул)
    B_MAX: float = 12000.0             # жёсткий потолок расхода (§7.2), = 1.2*B
    ARM_BUDGET_CAP_FRAC: float = 0.30  # стартовый per-arm cap = доля от B (правка C3)

    # --- Beta-prior'ы и доверие (§3) ---
    alpha0: float = 1.0
    beta0: float = 1.0
    alpha_c: float = 1.0
    beta_c: float = 1.0
    Z: float = 1.96

    # --- Статистический гейт (§6.1) ---
    N_MIN_BID: int = 30
    N_MIN_PAUSE: int = 100
    N_MIN_SCALE: int = 100
    N_MIN_IMP: int = 3000              # только network

    # --- Контроллер ставок (§6.2) ---
    gamma: float = 0.5
    MAX_STEP: float = 0.20

    # --- Границы ставок по типу инвентаря (§6.2, §7.1) ---
    # search: CPC ₽; network: CPM ₽/1000 показов
    BID_FLOOR: dict[InvType, float] = field(
        default_factory=lambda: {InvType.SEARCH: 3.0, InvType.NETWORK: 20.0}
    )
    BID_CEIL: dict[InvType, float] = field(
        default_factory=lambda: {InvType.SEARCH: 60.0, InvType.NETWORK: 200.0}
    )
    BID_DEFAULT: dict[InvType, float] = field(
        default_factory=lambda: {InvType.SEARCH: 12.0, InvType.NETWORK: 60.0}
    )

    # --- Bandit-аллокация бюджета (§6.3) ---
    EXPLORE_FRAC: float = 0.15

    # --- Отсечение и масштабирование (§6.4) ---
    PAUSE_FACTOR: float = 1.5
    SCALE_FACTOR: float = 0.8
    SCALE_SPEND_FRAC: float = 0.9
    SCALE_UP: float = 0.30

    # --- Guardrails (§7) ---
    MAX_CHANGES_PER_TICK: int = 40
    KILL_WINDOW: int = 7
    CATASTROPHE_CPA: float = 800.0
    WARMUP_TICKS: int = 10             # kill-switch неактивен первые N тиков (правка C2)

    # --- Материальные изменения (правка C4) ---
    CHANGE_EPS_BID: float = 0.03
    CHANGE_EPS_BUDGET: float = 0.05

    # --- Детерминизм и шум мока (§5) ---
    SEED: int = 42
    NOISE_SIGMA: float = 0.15
    UPLIFT_MAX: float = 0.5
    # Параметры хуков мока (M2: вынесены из модульных констант mock_direct)
    DRIFT_MAX_REL: float = 0.5          # дрейф true_cr к концу прогона: до (1-этого)
    FREQ_SAT_HALF_IMP: float = 500000.0  # показов до падения охвата РСЯ вдвое
    seasonality: list[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0, 1.05, 1.15, 0.9, 0.8]
    )

    # --- Сценарий и прогон (§12) ---
    N_SEARCH: int = 12
    N_NETWORK: int = 8
    M_TICKS: int = 60
    WINDOW: str = "cumulative"
    DRIFT_WINDOW: int = 14
    OPT_LEVEL: str = "arm"
    DRIFT: bool = False
    FREQ_SAT: bool = False

    # --- Параметры форм мок-аукциона (§5; потребит коммит 5) ---
    RATIO_BASE: float = 0.6           # cpc_ratio/cpm_ratio = RATIO_BASE + RATIO_SLOPE*IS
    RATIO_SLOPE: float = 0.35
    POS_UPLIFT_LO: float = 0.8        # clamp pos_uplift (search)
    POS_UPLIFT_HI: float = 1.25

    # --- Диапазоны скрытых параметров армов (min, max) для make_scenario (коммит 5) ---
    SEARCH_PARAM_RANGES: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "base_demand": (500.0, 5000.0),
            "true_ctr": (0.03, 0.12),
            "true_cr": (0.02, 0.10),
            "value_per_conv": (300.0, 1500.0),
            "bid_ref": (8.0, 30.0),
            "k": (1.0, 2.0),
        }
    )
    NETWORK_PARAM_RANGES: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "base_demand_imp": (20000.0, 200000.0),
            "true_ctr_net": (0.002, 0.012),
            "true_cr_net": (0.005, 0.03),
            "value_per_conv": (300.0, 1500.0),
            "cpm_ref": (30.0, 120.0),
            "k": (1.0, 2.0),
        }
    )

    def __post_init__(self) -> None:
        # Бюджет (§7.2, LOCKED: B_MAX = 1.2*B)
        assert self.B <= self.B_MAX, f"B ({self.B}) должен быть <= B_MAX ({self.B_MAX})"
        assert math.isclose(self.B_MAX, 1.2 * self.B), (
            f"B_MAX ({self.B_MAX}) должен быть 1.2*B ({1.2 * self.B})"
        )
        # Доли (§6.3, C3)
        assert 0.0 < self.EXPLORE_FRAC < 1.0, (
            f"EXPLORE_FRAC ({self.EXPLORE_FRAC}) должен быть в (0, 1)"
        )
        assert 0.0 < self.ARM_BUDGET_CAP_FRAC <= 1.0, (
            f"ARM_BUDGET_CAP_FRAC ({self.ARM_BUDGET_CAP_FRAC}) должен быть в (0, 1]"
        )
        # Границы ставок: FLOOR < DEFAULT < CEIL для обоих типов (§6.2)
        for inv in (InvType.SEARCH, InvType.NETWORK):
            lo, dflt, hi = self.BID_FLOOR[inv], self.BID_DEFAULT[inv], self.BID_CEIL[inv]
            assert lo < dflt < hi, (
                f"{inv.value}: требуется BID_FLOOR < BID_DEFAULT < BID_CEIL, "
                f"получено {lo} < {dflt} < {hi}"
            )
        # Сезонность и шаг ставки (§5, §6.2)
        assert len(self.seasonality) == 7, (
            f"seasonality должен иметь длину 7, получено {len(self.seasonality)}"
        )
        assert 0.0 < self.MAX_STEP < 1.0, (
            f"MAX_STEP ({self.MAX_STEP}) должен быть в (0, 1)"
        )


# Готовый экземпляр со всеми дефолтами (LOCKED).
DEFAULT_CONFIG = Config()
