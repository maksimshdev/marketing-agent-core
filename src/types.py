"""Канонная (платформо-независимая) модель данных — спека §2.

Адаптер площадки (§4) приводит сырьё к этим структурам; всё ядро (§3, §6–§10)
работает только с ними. Единицы: деньги — ₽; CPC/CPA — ₽; CPM — ₽/1000 показов;
tick = один цикл = «день».

Модуль намеренно без внешних зависимостей (только стандартная библиотека):
типы должны импортироваться любым слоём без тяги numpy/pandas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Перечисления
# ---------------------------------------------------------------------------

class Level(str, Enum):
    """Уровень сущности в иерархии Директа (§1): Account → Campaign → AdGroup →
    (Keyword | Placement). Arm по умолчанию — keyword (search) или placement (network)."""

    KEYWORD = "keyword"
    PLACEMENT = "placement"
    ADGROUP = "adgroup"
    CAMPAIGN = "campaign"


class InvType(str, Enum):
    """Тип инвентаря (§1). Определяет рычаг ставки и формулы CPA/eff (§3, §6.2)."""

    SEARCH = "search"    # поиск, ставка = CPC (₽/клик)
    NETWORK = "network"  # РСЯ, ставка = CPM (₽/1000 показов)


class BidUnit(str, Enum):
    """Единица ставки арма. Жёстко связана с InvType (см. bid_unit_for)."""

    CPC = "CPC"
    CPM = "CPM"


class Status(str, Enum):
    """Статус арма (§1). EXPLORING — недоисследован, получает explore-бюджет (§6.3);
    ACTIVE — гейт ставки пройден (§6.1); PAUSED — отсечён (§6.4, per-arm cap = 0)."""

    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    EXPLORING = "EXPLORING"


class Action(str, Enum):
    """Тип действия в DecisionRecord/Proposal (§9, §10).

    Приоритет конфликтов (§6): пауза > ставка > бюджет > масштаб — см. ACTION_PRIORITY.
    """

    PAUSE = "PAUSE"            # §6.4: per-arm cap → 0
    SET_BID = "SET_BID"        # §6.2: контроллер ставок
    SET_BUDGET = "SET_BUDGET"  # §6.3: bandit-аллокация в пределах cap
    SCALE = "SCALE"            # §6.4: поднять per-arm cap победителю
    NOOP = "NOOP"              # действие отброшено/без изменений


# Приоритет разрешения конфликтов (§6): больше число = выше приоритет.
ACTION_PRIORITY: dict[Action, int] = {
    Action.PAUSE: 3,
    Action.SET_BID: 2,
    Action.SET_BUDGET: 1,
    Action.SCALE: 0,
    Action.NOOP: -1,
}


def bid_unit_for(inv_type: InvType) -> BidUnit:
    """Единица ставки, жёстко определяемая типом инвентаря (§1)."""
    return BidUnit.CPC if inv_type == InvType.SEARCH else BidUnit.CPM


# ---------------------------------------------------------------------------
# Канонные структуры (§2)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EntityRef:
    """Ссылка на сущность. frozen → хешируема, годна как ключ стора памяти (§9)."""

    level: Level
    id: str
    parent_id: Optional[str]
    inv_type: InvType


@dataclass
class MetricSnapshot:
    """Сырые счётчики за окно одного тика (§2). Производные KPI здесь не хранятся —
    они считаются в kpi.py (§3) из накопленных счётчиков ArmState."""

    ref: EntityRef
    tick: int
    window: str  # "cumulative" | "N" (метка окна, по которому собран снапшот)
    impressions: int = 0
    clicks: int = 0
    spend: float = 0.0
    conversions: int = 0
    revenue: float = 0.0


@dataclass
class ArmState:
    """Состояние арма: рычаги управления + накопленные счётчики (для апостериорных)
    + производные KPI с границами (§2, §3).

    Производные поля (CTR_hat … CPA_hi) — None до первого пересчёта kpi.recompute_arm.
    arm_budget_cap (правка C3) — потолок дневной доли арма: bandit клампится им (§6.3),
    scale поднимает (§6.4), prune ставит 0 (пауза). None = «ещё не инициализирован»
    (config задаёт стартовое ARM_BUDGET_CAP_FRAC * B при создании сценария).
    """

    ref: EntityRef
    status: Status
    bid: float
    bid_unit: BidUnit
    daily_budget: float

    # Накопленные счётчики (для bandit/Wilson, §6.3/§3)
    cum_imp: int = 0
    cum_clicks: int = 0
    cum_conv: int = 0
    cum_spend: float = 0.0
    cum_rev: float = 0.0

    # Per-arm budget cap (правка C3)
    arm_budget_cap: Optional[float] = None

    # Производные KPI (§3) — заполняются kpi.recompute_arm
    CTR_hat: Optional[float] = None
    CR_hat: Optional[float] = None
    CPC_hat: Optional[float] = None
    CPM_hat: Optional[float] = None
    CPA_est: Optional[float] = None

    # Доверительные границы (Wilson → CPA-границы, §3)
    CTR_lo: Optional[float] = None
    CTR_hi: Optional[float] = None
    CR_lo: Optional[float] = None
    CR_hi: Optional[float] = None
    CPA_lo: Optional[float] = None
    CPA_hi: Optional[float] = None


@dataclass
class DecisionRecord:
    """Запись решения для памяти и объяснимости (§9, §10).

    before/after — словари изменённых полей (обратимость, инвариант §7.5);
    reason — человекочитаемая причина (§10); kpis_snapshot — KPI-срез на момент решения.
    """

    tick: int
    ref: EntityRef
    action: Action
    before: dict[str, Any]
    after: dict[str, Any]
    reason: str
    kpis_snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass
class Proposal:
    """Внутренний носитель предложенного действия между слоями оптимизатора (§6)
    и исполнителем под guardrails (§7/§8.8). Не часть канонной модели §2.

    new_value — целевое значение рычага (ставка/бюджет/cap) в зависимости от action;
    priority — для разрешения конфликтов и rate-limit (по умолчанию из ACTION_PRIORITY).
    """

    ref: EntityRef
    action: Action
    new_value: Optional[float]
    reason: str
    priority: Optional[int] = None
    kpis_snapshot: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.priority is None:
            self.priority = ACTION_PRIORITY[self.action]
