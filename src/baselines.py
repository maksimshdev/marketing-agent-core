"""Бейзлайн приёмки (§12): равный бюджет + фиксированная ставка, без оптимизации.

Прогон M тиков через тот же мок, сбор тех же аккаунт-метрик, что у агента (через
Memory), для честного сравнения CPA_acct / Σconv / Σspend / ROMI.
"""

from __future__ import annotations

from src.config import Config
from src.memory import Memory
from src.mock_direct import MockYandexDirect


def baseline_equal(cfg: Config, adapter: MockYandexDirect | None = None) -> Memory:
    """Равный дневной бюджет B/n и фикс. ставка BID_DEFAULT[inv] на все армы; ставки/
    бюджеты/паузы/масштаб НЕ трогаются. make_scenario уже задаёт bid=BID_DEFAULT и
    daily_budget=B/n, поэтому достаточно прогнать метрики без вмешательства."""
    if adapter is None:
        adapter = MockYandexDirect(cfg)
    mem = Memory(cfg)
    mem.init_states(adapter.all_states())
    refs = adapter.refs()
    for t in range(cfg.M_TICKS):
        snaps = adapter.get_metrics(refs, t)
        mem.ingest(snaps)
        mem.snapshot_account_tick(t, snaps)
    return mem
