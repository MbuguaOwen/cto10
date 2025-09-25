\
import yaml, os
from dataclasses import dataclass
from typing import Any, Dict
from datetime import datetime
import re

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Minimal validation of critical keys + broadened schema checks
    def _require(cfg_node, path, pred, msg):
        node = cfg_node
        for k in path:
            if k not in node:
                raise AssertionError(f"Missing config key: {'.'.join(path)}")
            node = node[k]
        if not pred(node):
            raise AssertionError(msg)

    try:
        labels = cfg.get("labels", {}) if isinstance(cfg, dict) else {}
        _require({"labels": labels}, ["labels", "horizon_hours"], lambda x: (isinstance(x, (int, float)) and x > 0), "labels.horizon_hours>0")
        _require({"labels": labels}, ["labels", "quick_ignition_hours"], lambda x: (isinstance(x, (int, float)) and x >= 0), "labels.quick_ignition_hours>=0")
        _require({"labels": labels}, ["labels", "r_mult"], lambda x: (isinstance(x, (int, float)) and x > 0), "labels.r_mult>0")
        # New label policy fields
        if "non_overlap_policy" in labels:
            _require({"labels": labels}, ["labels", "non_overlap_policy"], lambda x: x in ("quick", "to_exit"), "labels.non_overlap_policy must be quick|to_exit")
        if "intra_bar_epsilon_ms" in labels:
            _require({"labels": labels}, ["labels", "intra_bar_epsilon_ms"], lambda x: isinstance(x, int) and 0 <= x <= 1000, "labels.intra_bar_epsilon_ms in [0,1000]")
        if "min_risk_atr_frac" in labels:
            _require({"labels": labels}, ["labels", "min_risk_atr_frac"], lambda x: (isinstance(x, (int, float)) and 0 <= x <= 0.1), "labels.min_risk_atr_frac in [0,0.1]")

        # execution_sim fees
        fees = (cfg.get("execution_sim", {}) or {}).get("fees", {}) or {}
        if fees:
            for k in ("maker_bps", "taker_bps"):
                if k not in fees:
                    raise AssertionError(f"execution_sim.fees.{k} missing")

        # parity test sanity
        parity = (cfg.get("tests", {}) or {}).get("parity", {}) or {}
        if parity:
            _require({"x": parity.get("r_tol", 0.02)}, ["x"], lambda x: 0 <= x <= 0.1, "tests.parity.r_tol too large")
            _require({"x": parity.get("max_signal_mismatch", 0)}, ["x"], lambda x: isinstance(x, int) and x >= 0, "tests.parity.max_signal_mismatch>=0")

        # walkforward months not in the future (support either flat months or data.months)
        months = None
        if "months" in cfg and isinstance(cfg["months"], (list, tuple)):
            months = list(cfg["months"]) or None
        elif "data" in cfg and isinstance(cfg.get("data", {}), dict):
            dm = cfg["data"].get("months")
            if isinstance(dm, (list, tuple)):
                months = list(dm)
            elif isinstance(dm, dict) and "start" in dm and "end" in dm:
                # expand range to [start..end] yyyy-mm if needed? sanity check only on end
                months = [str(dm.get("end"))]
        today = datetime.utcnow()
        if months:
            try:
                ym_max = max(str(x) for x in months if x)
                y, m = map(int, ym_max.split("-"))
                if (y, m) > (today.year, today.month):
                    raise AssertionError(f"Max month {ym_max} is in the future vs {today:%Y-%m}")
            except Exception:
                # if format unexpected, ignore
                pass
    except Exception:
        # Raise a clear assertion error to surface mis-keys early
        raise
    return cfg
