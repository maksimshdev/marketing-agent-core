"""Лёгкий smoke-тест пайплайна run.py (§12): прогон на M=5 создаёт артефакты."""

from __future__ import annotations

import os

import run as run_module


def test_run_pipeline_smoke(tmp_path):
    out = str(tmp_path / "out")
    res = run_module.main(out_dir=out, M=5)
    assert res["M"] == 5
    for fname in ("cpa_acct.png", "conversions.png", "budget_split.png",
                  "comparison.csv", "decisions_sample.txt"):
        assert os.path.exists(os.path.join(out, fname)), fname


def test_oracle_cpa_within_target():
    from src.config import DEFAULT_CONFIG as CFG
    from src.mock_direct import MockYandexDirect
    from src import oracle
    res = oracle.compute_oracle(MockYandexDirect(CFG), CFG)
    # оракул держит блендед CPA ≤ TARGET (берёт только эффективные слои)
    assert res.cpa <= CFG.TARGET_CPA + 1e-6
    assert res.conv_per_day > 0
