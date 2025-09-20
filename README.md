cto10r — Clean Take-Off 10R (Bars for features, Ticks for labels)

A deterministic research pipeline to discover and evaluate 10R Clean Take-Off (10R-CTO) fingerprints:

Bars (1-minute by default) → build multi-horizon features and gate candidates.

Ticks → provide first-touch truth for SL/TP labeling (no OHLC ambiguity).

Targets (research): hit +1R inside “quick ignition”, and +10R within horizon.

Entry level: entry-minus-cushion (P = E − η·ATR(t) for longs; mirrored for shorts).

Volatility-aware “R-hero”: SL distance η can adapt to ATR percentile; label time windows can scale with η so +1R/+10R remain feasible in high vol.

CLI
python -m cto10r --config configs/motifs.yaml --mode preflight|walkforward [--force] [--clean]
# Stage controls (added): --only {features|tick_labeling|mining|simulate} | --from <stage> | --until <stage>

Stages (and what they write)

preflight → fail-fast config & data checks

features → loads bars, builds features, fits percentiles on train, writes:

candidates.parquet (now includes per-row eta and risk_dist = eta * ATR)

regimes.csv

tick labeling (ticks = first-touch, +1R quick ignition, +10R horizon, stop-band & dwell rules) → writes:

events.parquet with outcome ∈ {win,loss,timeout} and timestamps

mining (simple interpretable rule miner) → writes:

artifacts/gating.json (+ placeholders for motif banks, if used)

simulate (test-month events only) → writes:

trades.csv, stats.json (wr, counts, expR proxy)

Each stage is resumable: re-runs SKIP (exists) unless --force. On exceptions, a crash_report.json is written with traceback + config snapshot.

What’s new: Volatility-aware R/SL (“R-hero”)

Per-row η from ATR percentile
Configure a piecewise table: labels.eta_by_atr_p: [[lo,hi,eta], ...].
The engine maps atr_p → eta and stores it per candidate (cands.eta).

Optional η caps
labels.eta_cap.{min,max} clamps η to avoid absurd SLs or microscopic sizes downstream.

Time scaling (optional)
labels.time_scale_from_eta scales label windows with η (vs base_eta), with a clamp to avoid exploding/vanishing windows. This reduces artificial timeouts in high vol.

TP stays in R-space
Research uses labels.r_mult (default 10R). Changing η only changes distance, not $ risk if you size positions by R in live.

Typical workflow (step-by-step)

Since you just enabled the R-hero logic, the next command is tick labeling after rebuilding candidates.

# 0) Preflight (sanity)
python -m cto10r --config configs/motifs.yaml --mode preflight

# 1) Rebuild candidates (now include per-row eta)
python -m cto10r --config configs/motifs.yaml --mode walkforward --only features --force

# 2) Label with ticks (uses per-row time windows if enabled)
python -m cto10r --config configs/motifs.yaml --mode walkforward --only tick_labeling --force

# 3) Mine rules (unchanged)
python -m cto10r --config configs/motifs.yaml --mode walkforward --only mining --force

# 4) Simulate (unchanged; TP still = r_mult × SL)
python -m cto10r --config configs/motifs.yaml --mode walkforward --only simulate --force

# Full run from scratch (if needed)
python -m cto10r --config configs/motifs.yaml --mode walkforward --clean --force

Config essentials (excerpt)
labels:
  # Baseline eta (used if no table match) and baseline for time-scaling
  eta_atr: 1.0

  # ATR percentile → eta (volatility-aware SL)
  eta_by_atr_p:
    - [0,   35, 0.50]
    - [35,  65, 1.00]
    - [65,  85, 3.00]
    - [85,  95, 7.50]
    - [95, 101, 12.00]   # cap high-vol

  # Optional hard caps (extra safety)
  eta_cap:
    min: 0.50
    max: 12.00

  # Research targets in R-space
  r_mult: 10.0

  # Optional: scale label windows with eta
  time_scale_from_eta:
    base_eta: 1.0       # compare η against this
    clamp: [0.5, 40.0]  # keep windows reasonable

  # Invariants to respect:
  # risk_floor_rho <= min(eta)  (otherwise candidates get zeroed)


If you want your prior η = 15 world as baseline, set eta_atr: 15.0, raise the top bin/cap (e.g., 20–30), and set time_scale_from_eta.base_eta: 15.0.

Data requirements

Bars: CSV/Parquet with ts (ms), open, high, low, close. Monotonic ts per symbol.

Ticks: CSV/Parquet with ts (ms), price. The engine streams monthly files and filters by candidate windows.

Mixed epoch units (s/ms/µs/ns) are auto-coerced; rows with non-finite ts are skipped.

Quick sanity checks

After features: outputs/<SYM>/fold_i/candidates.parquet has eta, risk_dist, and non-zero rows.

After tick_labeling: console prints events: N (wins=…, losses=…, timeouts=…) per fold and writes events.parquet.

After simulate: stats.json shows non-zero totals; trades.csv exists.

Troubleshooting

Zero candidates: relax gates in this order → close_to_ext_atr_max ↑, t240_abs_min ↓, accel_15_240_min ↓, reduce kappa quorum. Ensure risk_floor_rho ≤ min(eta).

All timeouts in high vol: enable time scaling (time_scale_from_eta) or increase horizon_hours / quick_ignition_hours.

Tick crashes on timestamps: CSV row with bad ts is skipped; Parquet path coerces units. If a month still shows zero ticks, confirm files and month ranges.

Determinism & deps

Seed: seed: 42 (set in YAML).

Python 3.10+, libs: numpy, pandas, pyyaml, tqdm (plus your Parquet engine: pyarrow or fastparquet).

Notes

The built-in miner is intentionally simple (coarse bins, short rules) for transparency. Feel free to swap a more powerful miner later.

Learning excludes TSL by design; use TSL (or BE/partials) only in live after +1R if you want to reduce tail risk time in market.