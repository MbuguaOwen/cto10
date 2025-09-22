#!/usr/bin/env python3
"""Utility helpers for leakage checks.

These helpers make it easy to run negative controls such as label permutation,
label time-shifts, and quick BH-FDR summaries on mined rules.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


DEFAULT_EVENTS_CANDIDATES = ("events_train.csv", "labels_train.csv")


@dataclass
class FileSet:
    root: Path
    fold: int

    @property
    def artifacts(self) -> Path:
        base = self.root / f"fold_{self.fold}" / "artifacts"
        if not base.exists():
            raise FileNotFoundError(f"Artifacts directory missing: {base}")
        return base

    def find_first(self, candidates: Sequence[str]) -> Path:
        for name in candidates:
            path = self.artifacts / name
            if path.exists():
                return path
        joined = ", ".join(str(self.artifacts / name) for name in candidates)
        raise FileNotFoundError(f"None of the candidate files exist: {joined}")


# ---------------------------------------------------------------------------
# Label permutation
# ---------------------------------------------------------------------------

def permute_train_labels(args: argparse.Namespace) -> None:
    files = FileSet(Path(args.output_root), args.fold)
    events = files.find_first(args.event_candidates)
    df = pd.read_csv(events)

    label_cols: List[str]
    if args.columns:
        label_cols = list(args.columns)
    elif "outcome" in df.columns:
        label_cols = ["outcome"]
    else:
        label_cols = [c for c in ("is_win", "is_loss", "is_timeout") if c in df.columns]

    if not label_cols:
        raise SystemExit("Could not infer label columns; pass --columns explicitly.")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(df))
    if len(label_cols) == 1:
        col = label_cols[0]
        df[col] = df[col].to_numpy()[perm]
    else:
        df.loc[:, label_cols] = df[label_cols].to_numpy()[perm]

    write_with_backup(events, df, suffix=".perm_bak")
    print(f"Permuted {len(df)} rows in {events}")


# ---------------------------------------------------------------------------
# Label time-shift
# ---------------------------------------------------------------------------

def shift_train_labels(args: argparse.Namespace) -> None:
    files = FileSet(Path(args.output_root), args.fold)
    events = files.find_first(args.event_candidates)
    df = pd.read_csv(events)

    ts_col = args.ts_column
    if ts_col not in df.columns:
        raise SystemExit(f"Timestamp column '{ts_col}' not found in {events}")

    df = df.sort_values(ts_col).reset_index(drop=True)

    label_cols: List[str]
    if args.columns:
        label_cols = list(args.columns)
    elif "outcome" in df.columns:
        label_cols = ["outcome"]
    else:
        label_cols = [c for c in ("is_win", "is_loss", "is_timeout") if c in df.columns]

    if not label_cols:
        raise SystemExit("Could not infer label columns; pass --columns explicitly.")

    original_dtypes = df[label_cols].dtypes
    df[label_cols] = df[label_cols].shift(args.steps)
    before = len(df)
    df = df.dropna(subset=label_cols).reset_index(drop=True)
    dropped = before - len(df)
    for col, dtype in original_dtypes.items():
        try:
            df[col] = df[col].astype(dtype)
        except (TypeError, ValueError):
            # Keep pandas' best-effort dtype if restoration fails
            pass

    write_with_backup(events, df, suffix=".shift_bak")
    print(
        f"Shifted {len(df)} rows (dropped {dropped}) in {events} using {ts_col} order"
    )


# ---------------------------------------------------------------------------
# Simulate twice helper
# ---------------------------------------------------------------------------

def simulate_twice(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "-W",
        "ignore::FutureWarning",
        "-m",
        "cto10r",
        "--config",
        args.config,
        "--mode",
        "walkforward",
        "--only",
        "simulate",
        "--force",
    ]
    if args.extra:
        cmd.extend(args.extra)

    for run in range(1, 3):
        print(f"\n[simulate run {run}] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    print("Finished running simulate twice; inspect outputs for parity.")


# ---------------------------------------------------------------------------
# BH-FDR helper
# ---------------------------------------------------------------------------

def bh_fdr(pvalues: Iterable[float], alpha: float) -> float | None:
    series = pd.Series(pvalues).sort_values()
    m = len(series)
    if m == 0:
        return None
    thresholds = pd.Series(range(1, m + 1), dtype=float) * alpha / m
    mask = series <= thresholds
    if not mask.any():
        return None
    idx = mask[mask].index[-1]
    return float(series.loc[idx])


def summarize_bh(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    for fold in args.folds:
        files = FileSet(root, fold)
        path = files.artifacts / "rules_eval.csv"
        if not path.exists():
            print(f"[fold {fold}] missing: {path}")
            continue
        df = pd.read_csv(path)
        if "precision_lcb" not in df.columns:
            print(f"[fold {fold}] precision_lcb column missing in {path}")
            continue
        faux_p = 1 - df["precision_lcb"].clip(0, 1)
        cut = bh_fdr(faux_p, args.alpha)
        if cut is None:
            print(f"[fold {fold}] no rules pass BH-FDR @ {args.alpha:.0%}")
            continue
        kept = (faux_p <= cut).sum()
        print(
            f"[fold {fold}] keep {kept} / {len(df)} rules @ {args.alpha:.0%} (threshold={cut:.4f})"
        )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_with_backup(path: Path, df: pd.DataFrame, suffix: str) -> None:
    original = Path(path)
    backup = original.with_suffix(suffix + original.suffix)
    if backup.exists():
        backup.unlink()
    original.rename(backup)
    df.to_csv(original, index=False)
    print(f"Backup saved to {backup}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Helpers for leakage checks")
    parser.add_argument(
        "--output-root",
        default="outputs/BTCUSDT",
        help="Root directory that contains fold_{k}/artifacts outputs (default: outputs/BTCUSDT)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    perm = subparsers.add_parser("permute", help="Permute train labels for a fold")
    perm.add_argument("--fold", type=int, required=True, help="Fold index to modify")
    perm.add_argument(
        "--seed", type=int, default=None, help="Optional random seed for reproducibility"
    )
    perm.add_argument(
        "--event-candidates",
        nargs="*",
        default=DEFAULT_EVENTS_CANDIDATES,
        help="Candidate train label filenames to check under the artifacts directory",
    )
    perm.add_argument(
        "--columns",
        nargs="*",
        help="Explicit label columns to permute (defaults to outcome or is_* columns)",
    )
    perm.set_defaults(func=permute_train_labels)

    shift = subparsers.add_parser("shift", help="Shift train labels by a number of rows")
    shift.add_argument("--fold", type=int, required=True, help="Fold index to modify")
    shift.add_argument(
        "--steps", type=int, default=1, help="Number of rows to shift forward (default: 1)"
    )
    shift.add_argument(
        "--ts-column",
        default="ts",
        help="Timestamp column that determines chronological order (default: ts)",
    )
    shift.add_argument(
        "--event-candidates",
        nargs="*",
        default=DEFAULT_EVENTS_CANDIDATES,
        help="Candidate train label filenames to check under the artifacts directory",
    )
    shift.add_argument(
        "--columns",
        nargs="*",
        help="Explicit label columns to shift (defaults to outcome or is_* columns)",
    )
    shift.set_defaults(func=shift_train_labels)

    sim = subparsers.add_parser(
        "simulate-twice", help="Run walkforward simulate twice for parity checks"
    )
    sim.add_argument("--config", required=True, help="Path to the walkforward config YAML")
    sim.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Any extra arguments to pass after the standard simulate command",
    )
    sim.set_defaults(func=simulate_twice)

    bh = subparsers.add_parser("bh", help="Quick BH-FDR summary on rules_eval.csv")
    bh.add_argument(
        "--folds",
        nargs="*",
        type=int,
        default=[0, 1],
        help="Fold indices to inspect (default: 0 1)",
    )
    bh.add_argument(
        "--alpha", type=float, default=0.10, help="Target FDR level (default: 0.10)"
    )
    bh.set_defaults(func=summarize_bh)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
