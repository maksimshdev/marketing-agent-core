"""Демо-прогон и артефакты приёмки (спека §12).

Сравнивает три конфигурации на ОДНОМ SEED (идентичный скрытый сценарий, но три
независимых инстанса мока — агент мутирует свой через адаптер):
  a. agent    — Orchestrator(...).run(M)
  b. baseline — равный бюджет + фикс. ставка (baselines.baseline_equal)
  c. oracle   — статический оптимум на истинных параметрах (oracle.compute_oracle, C1)

Артефакты в dashboard/sample/: PNG (CPA_acct, конверсии, разбивка бюджета поиск/РСЯ),
CSV «агент vs baseline vs oracle», decisions_sample.txt (объяснимость §10).

Запуск:  python run.py        (нужен matplotlib; headless Agg)
"""

from __future__ import annotations

import dataclasses
import os

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import pandas as pd

from src import kpi, oracle as oracle_mod
from src.baselines import baseline_equal
from src.config import DEFAULT_CONFIG, Config
from src.memory import Memory, render_decision
from src.mock_direct import MockYandexDirect
from src.orchestrator import Orchestrator
from src.types import Action

OUT_DIR = "dashboard/sample"


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------

def _rolling_cpa(account_ticks: list[dict], window: int) -> list[float]:
    return [kpi.cpa_acct(account_ticks[: t + 1], window) for t in range(len(account_ticks))]


def _summary(mem: Memory, cfg: Config) -> dict:
    spend = sum(a["spend"] for a in mem.account_ticks)
    conv = sum(a["conversions"] for a in mem.account_ticks)
    rev = sum(s["revenue"] for s in mem.tick_snapshots)
    cpa_series = _rolling_cpa(mem.account_ticks, cfg.KILL_WINDOW)
    return {
        "final_cpa_acct": cpa_series[-1] if cpa_series else float("inf"),
        "total_conv": conv,
        "total_spend": spend,
        "total_revenue": rev,
        "romi": (rev - spend) / spend if spend > 0 else float("inf"),
        "cpa_series": cpa_series,
    }


# ---------------------------------------------------------------------------
# Прогон
# ---------------------------------------------------------------------------

def run_all(cfg: Config) -> dict:
    M = cfg.M_TICKS
    # a. agent
    agent_mem = Orchestrator(MockYandexDirect(cfg), Memory(cfg), cfg).run(M)
    # b. baseline
    base_mem = baseline_equal(cfg)
    # c. oracle (свежий мок с тем же SEED → тот же скрытый сценарий)
    orc = oracle_mod.compute_oracle(MockYandexDirect(cfg), cfg)
    return {"agent": agent_mem, "baseline": base_mem, "oracle": orc, "M": M, "cfg": cfg}


# ---------------------------------------------------------------------------
# Артефакты
# ---------------------------------------------------------------------------

def _plot_cpa(res: dict, out_dir: str) -> None:
    cfg, M = res["cfg"], res["M"]
    a, b = _summary(res["agent"], cfg), _summary(res["baseline"], cfg)
    ticks = list(range(M))
    plt.figure(figsize=(9, 5))
    plt.plot(ticks, a["cpa_series"], label="agent", color="tab:blue")
    plt.plot(ticks, b["cpa_series"], label="baseline", color="tab:orange")
    plt.axhline(res["oracle"].cpa, ls="--", color="tab:green", label="oracle (опт.)")
    plt.axhline(cfg.TARGET_CPA, ls=":", color="black", label=f"target={cfg.TARGET_CPA:.0f}")
    plt.xlabel("tick (день)")
    plt.ylabel("CPA_acct, ₽ (скользящее окно)")
    plt.title("Сходимость CPA_acct: agent vs baseline vs oracle (§12)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cpa_acct.png"), dpi=110)
    plt.close()


def _plot_conversions(res: dict, out_dir: str) -> None:
    cfg, M = res["cfg"], res["M"]
    agent_ticks = res["agent"].account_ticks
    base_ticks = res["baseline"].account_ticks
    ticks = list(range(M))
    agent_cum, base_cum, s = [], [], 0.0
    for a in agent_ticks:
        s += a["conversions"]
        agent_cum.append(s)
    s = 0.0
    for a in base_ticks:
        s += a["conversions"]
        base_cum.append(s)
    oracle_cum = [res["oracle"].conv_per_day * (t + 1) for t in ticks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(ticks, [a["conversions"] for a in agent_ticks], label="agent", color="tab:blue")
    ax1.plot(ticks, [a["conversions"] for a in base_ticks], label="baseline", color="tab:orange")
    ax1.set_title("Конверсии за тик")
    ax1.set_xlabel("tick"); ax1.set_ylabel("conv/тик"); ax1.legend()
    ax2.plot(ticks, agent_cum, label="agent", color="tab:blue")
    ax2.plot(ticks, base_cum, label="baseline", color="tab:orange")
    ax2.plot(ticks, oracle_cum, ls="--", label="oracle", color="tab:green")
    ax2.set_title("Накопленные конверсии")
    ax2.set_xlabel("tick"); ax2.set_ylabel("Σ conv"); ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "conversions.png"), dpi=110)
    plt.close()


def _plot_budget_split(res: dict, out_dir: str) -> None:
    M = res["M"]
    snaps = res["agent"].tick_snapshots
    ticks = list(range(M))
    sp_search = [s["spend_search"] for s in snaps]
    sp_net = [s["spend_network"] for s in snaps]
    plt.figure(figsize=(9, 5))
    plt.stackplot(ticks, sp_search, sp_net,
                  labels=["поиск (CPC)", "РСЯ (CPM)"], colors=["tab:blue", "tab:purple"])
    plt.xlabel("tick (день)")
    plt.ylabel("расход, ₽")
    plt.title("Разбивка расхода: поиск vs РСЯ (переток бюджета, §12)")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "budget_split.png"), dpi=110)
    plt.close()


def _write_table(res: dict, out_dir: str) -> pd.DataFrame:
    cfg, M = res["cfg"], res["M"]
    a, b = _summary(res["agent"], cfg), _summary(res["baseline"], cfg)
    o = res["oracle"]
    rows = [
        {"config": "agent", "final_cpa_acct": round(a["final_cpa_acct"], 1),
         "total_conv": round(a["total_conv"], 1), "total_spend": round(a["total_spend"], 1),
         "romi": round(a["romi"], 3)},
        {"config": "baseline", "final_cpa_acct": round(b["final_cpa_acct"], 1),
         "total_conv": round(b["total_conv"], 1), "total_spend": round(b["total_spend"], 1),
         "romi": round(b["romi"], 3)},
        {"config": "oracle", "final_cpa_acct": round(o.cpa, 1),
         "total_conv": round(o.conv_per_day * M, 1), "total_spend": round(o.spend_per_day * M, 1),
         "romi": round((o.revenue_per_day - o.spend_per_day) / o.spend_per_day, 3)
         if o.spend_per_day > 0 else float("inf")},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "comparison.csv"), index=False)
    return df


def _write_decisions_sample(res: dict, out_dir: str, limit: int = 40) -> None:
    decisions = [r for r in res["agent"].decisions if r.action != Action.NOOP]
    lines = [render_decision(r) for r in decisions[:limit]]
    with open(os.path.join(out_dir, "decisions_sample.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_artifacts(res: dict, out_dir: str = OUT_DIR) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    _plot_cpa(res, out_dir)
    _plot_conversions(res, out_dir)
    _plot_budget_split(res, out_dir)
    df = _write_table(res, out_dir)
    _write_decisions_sample(res, out_dir)
    return df


def main(cfg: Config = DEFAULT_CONFIG, out_dir: str = OUT_DIR, M: int | None = None) -> dict:
    if M is not None:
        cfg = dataclasses.replace(cfg, M_TICKS=M)
    res = run_all(cfg)
    df = write_artifacts(res, out_dir)
    a = _summary(res["agent"], cfg)
    b = _summary(res["baseline"], cfg)
    print("=== Приёмка §12: agent vs baseline vs oracle ===")
    print(df.to_string(index=False))
    print(f"\nКонверсии: agent/oracle = {a['total_conv'] / max(res['oracle'].conv_per_day * cfg.M_TICKS, 1e-9):.2%}")
    print(f"CPA_acct: agent={a['final_cpa_acct']:.0f}₽ vs baseline={b['final_cpa_acct']:.0f}₽ "
          f"(target={cfg.TARGET_CPA:.0f}₽)")
    print(f"Артефакты: {out_dir}/")
    return res


if __name__ == "__main__":
    main()
