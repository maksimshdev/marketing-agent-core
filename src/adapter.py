"""Контракт адаптера площадки (спека §4).

Ядро работает с любой площадкой только через этот интерфейс. В фазе 1 реализация —
MockYandexDirect (src/mock_direct.py); боевой Директ/иные площадки — фаза 2, тот же
контракт. Здесь только интерфейс и иерархия ошибок, без логики симуляции.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.types import ArmState, EntityRef, MetricSnapshot, Status


# ---------------------------------------------------------------------------
# Типизированные ошибки адаптера
# ---------------------------------------------------------------------------

class AdapterError(Exception):
    """Базовая ошибка адаптера площадки."""


class RateLimitError(AdapterError):
    """Превышен лимит обращений к API (имитация лимитов реального Директа, §7.4)."""


class InvalidValueError(AdapterError):
    """Недопустимое значение (ставка вне [FLOOR, CEIL], отрицательный бюджет и т.п.)."""


class UnknownRefError(AdapterError):
    """Обращение к неизвестной сущности (ref не зарегистрирован в адаптере)."""


# ---------------------------------------------------------------------------
# Контракт
# ---------------------------------------------------------------------------

class PlatformAdapter(ABC):
    """Платформо-независимый контракт (§4).

    Гарантии реализаций: идемпотентность set_* (повторная установка того же значения
    не меняет состояние и возвращает True); типизированные ошибки из иерархии
    AdapterError; rate-limit отражает лимиты реального API (§7).
    """

    @abstractmethod
    def get_metrics(self, refs: list[EntityRef], tick: int) -> list[MetricSnapshot]:
        """Сырьё метрик за тик (не накопленное). Порядок снапшотов соответствует refs."""

    @abstractmethod
    def get_state(self, refs: list[EntityRef]) -> list[ArmState]:
        """Текущее видимое состояние армов (рычаги + накопленные/производные поля)."""

    @abstractmethod
    def set_bid(self, ref: EntityRef, bid: float) -> bool:
        """Установить ставку (в bid_unit арма, ₽). Идемпотентно."""

    @abstractmethod
    def set_budget(self, ref: EntityRef, daily: float) -> bool:
        """Установить дневной бюджетный лимит арма (₽). Идемпотентно."""

    @abstractmethod
    def set_status(self, ref: EntityRef, status: Status) -> bool:
        """Установить статус арма (ACTIVE/PAUSED/...). Идемпотентно."""
