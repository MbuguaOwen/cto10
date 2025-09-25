from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .ml import (
    build_design_matrix,
    transform_design_matrix,
    train_classifier,
    calibrate_probabilities,
)
from .mining import wilson_lcb

try:
    import joblib
except Exception:  # pragma: no cover - joblib is expected in runtime env
    joblib = None


@dataclass
class GateConfig:
    calibration: str = "isotonic"
    target_ppv_lcb: float = 0.60
    min_coverage: float = 0.02
    crosses_cap: int = 30
    val_months: int = 1


@dataclass
class LoserMaskConfig:
    enabled: bool = True
    min_support: int = 20
    min_months_with_lift: int = 0
    min_ppv_lcb: float = 0.55
    save_artifact: bool = True


@dataclass
class TrainedGate:
    model: Any
    tau: float
    feature_meta: Dict[str, Any]
    loser_literals: List[Dict[str, Any]]

    def score(self, X):  # type: ignore[override]
        proba = self.model.predict_proba(X)
        if isinstance(proba, np.ndarray):
            return proba[:, 1]
        return np.asarray(proba)[:, 1]

    def loser_mask_vec(self, literal_df: pd.DataFrame) -> np.ndarray:
        if not self.loser_literals:
            return np.zeros(len(literal_df), dtype=bool)
        fires = np.zeros(len(literal_df), dtype=bool)
        for rule in self.loser_literals:
            lits = rule.get("lits", []) or []
            mask = np.ones(len(literal_df), dtype=bool)
            for lit in lits:
                col = str(lit.get("col"))
                val = lit.get("val")
                if col not in literal_df.columns:
                    mask &= False
                    break
                series = literal_df[col].astype(str)
                mask &= series == str(val)
            fires |= mask
        return fires


def choose_tau_by_targets(
    probs_val: np.ndarray,
    y_val: np.ndarray,
    target_ppv_lcb: float,
    min_cov: float,
) -> Tuple[float, Dict[str, float]]:
    best = {"tau": 0.5, "ppv": 0.0, "ppv_lcb": 0.0, "cov": 0.0}
    hit = None
    n = len(probs_val)
    if n == 0:
        return best["tau"], best
    finite_probs = probs_val[np.isfinite(probs_val)]
    if finite_probs.size == 0:
        taus = np.array([0.0])
    else:
        qs = np.linspace(1.0, 0.0, 201)
        taus = np.unique(np.quantile(finite_probs, qs))
        taus = taus[::-1]
    for t in taus:
        pred = probs_val >= t
        cov = float(pred.mean())
        if cov <= 0:
            continue
        mask_res = np.isfinite(y_val)
        if mask_res.any():
            pred_eval = pred[mask_res]
            y_eval = y_val[mask_res]
        else:
            pred_eval = pred
            y_eval = y_val
        tp = int(((pred_eval == 1) & (y_eval == 1)).sum())
        pos = int(pred_eval.sum())
        ppv = (tp / pos) if pos > 0 else 0.0
        ppv_lcb = wilson_lcb(tp, pos)
        if ppv_lcb >= target_ppv_lcb:
            hit = {"tau": float(t), "ppv": ppv, "ppv_lcb": ppv_lcb, "cov": cov}
            break
        if cov >= min_cov and ppv_lcb > best["ppv_lcb"]:
            best = {"tau": float(t), "ppv": ppv, "ppv_lcb": ppv_lcb, "cov": cov}
    chosen = hit or best
    if chosen["cov"] < min_cov and hit is None:
        print(
            f"[gate] warning: coverage {chosen['cov']:.3f} below min_coverage {min_cov:.3f}; using best available τ",
            flush=True,
        )
    return chosen["tau"], chosen


def build_loser_literals(mined_df: pd.DataFrame | None, cfg: LoserMaskConfig) -> List[Dict[str, Any]]:
    if mined_df is None or mined_df.empty:
        return []
    required = {"kind", "support", "months_with_lift", "precision", "precision_lcb", "lits"}
    if not required.issubset(mined_df.columns):
        return []
    losers = mined_df[
        (mined_df["kind"].str.lower() == "loser")
        & (mined_df["support"].astype(int) >= int(cfg.min_support))
        & (mined_df["months_with_lift"].astype(int) >= int(cfg.min_months_with_lift))
        & (mined_df["precision_lcb"].astype(float) <= float(cfg.min_ppv_lcb))
    ].copy()
    out: List[Dict[str, Any]] = []
    for _, row in losers.iterrows():
        lits = row.get("lits")
        if not isinstance(lits, (list, tuple)):
            lits = []
        formatted = [
            {"col": str(lit.get("col")), "val": lit.get("val")}
            for lit in lits
            if isinstance(lit, dict)
        ]
        out.append(
            {
                "support": int(row.get("support", 0)),
                "precision": float(row.get("precision", 0.0)),
                "precision_lcb": float(row.get("precision_lcb", 0.0)),
                "months_with_lift": int(row.get("months_with_lift", 0)),
                "lits": formatted,
            }
        )
    return out


def save_loser_literals(path: Path, losers: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"losers": losers, "count": len(losers)}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train_gate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features_cfg: Dict[str, Any],
    gate_cfg: GateConfig,
    loser_cfg: LoserMaskConfig,
    mined_rules_df: pd.DataFrame | None,
    artifacts_dir: Path,
) -> Tuple[TrainedGate, Dict[str, Any]]:
    train_df = train_df.copy()
    val_df = val_df.copy()
    if "y" not in train_df.columns:
        raise ValueError("train_df must include 'y' column for supervision")
    if val_df.empty:
        val_df = train_df.copy()
    elif "y" not in val_df.columns:
        raise ValueError("val_df must include 'y' column for supervision")

    X_tr, y_tr, feature_meta = build_design_matrix(
        train_df,
        features_cfg,
        crosses_cap=int(gate_cfg.crosses_cap),
        val_df=val_df,
    )
    X_val, y_val, literals_val = transform_design_matrix(
        val_df,
        features_cfg,
        feature_meta,
    )

    clf = train_classifier(X_tr, y_tr, features_cfg, X_val=X_val, y_val=y_val)
    calibrated = calibrate_probabilities(clf, X_tr, y_tr, method=gate_cfg.calibration)

    probs_val = calibrated.predict_proba(X_val)[:, 1]
    # Brier score and reliability bins (deciles)
    try:
        from sklearn.metrics import brier_score_loss as _brier_score_loss
        brier = float(_brier_score_loss(y_val, probs_val)) if len(y_val) else float("nan")
    except Exception:
        brier = float("nan")
    import numpy as _np
    bins = _np.linspace(0, 1, 11)
    digitized = _np.digitize(probs_val, bins) - 1
    reliab = []
    for i in range(10):
        sel = (digitized == i)
        if not _np.any(sel):
            reliab.append({"bin": i, "p_mid": float((bins[i] + bins[i + 1]) / 2), "emp": None, "n": 0})
        else:
            emp = float(_np.asarray(y_val)[sel].mean())
            reliab.append({"bin": i, "p_mid": float((bins[i] + bins[i + 1]) / 2), "emp": emp, "n": int(sel.sum())})
    tau, diag = choose_tau_by_targets(
        probs_val,
        y_val,
        float(gate_cfg.target_ppv_lcb),
        float(gate_cfg.min_coverage),
    )

    losers = build_loser_literals(mined_rules_df, loser_cfg) if loser_cfg.enabled else []
    if loser_cfg.enabled and loser_cfg.save_artifact:
        save_loser_literals(artifacts_dir / "loser_mask.json", losers)

    gate = TrainedGate(model=calibrated, tau=float(tau), feature_meta=feature_meta, loser_literals=losers)
    diag_full = {
        "tau": float(tau),
        "ppv": float(diag.get("ppv", 0.0)),
        "ppv_lcb": float(diag.get("ppv_lcb", 0.0)),
        "cov": float(diag.get("cov", 0.0)),
        "loser_rules": int(len(losers)),
        "brier": brier,
        "reliability": reliab,
    }

    summary_path = artifacts_dir / "gate_summary.json"
    summary_path.write_text(json.dumps(diag_full, indent=2), encoding="utf-8")

    if joblib is not None:
        payload = {
            "model": calibrated,
            "tau": float(tau),
            "feature_meta": feature_meta,
            "loser_literals": losers,
        }
        joblib.dump(payload, artifacts_dir / "gate_bundle.pkl")

    return gate, diag_full


def load_trained_gate(path: Path) -> TrainedGate:
    if joblib is None:
        raise RuntimeError("joblib is required to load trained gate")
    payload = joblib.load(path)
    return TrainedGate(
        model=payload["model"],
        tau=float(payload["tau"]),
        feature_meta=payload["feature_meta"],
        loser_literals=payload.get("loser_literals", []),
    )


def literalize_candidates(
    df: pd.DataFrame,
    features_cfg: Dict[str, Any],
    feature_meta: Dict[str, Any],
) -> Tuple[Any, np.ndarray, pd.DataFrame]:
    X, y, literals = transform_design_matrix(df, features_cfg, feature_meta)
    return X, y, literals
