# Geopolitical Shocks and Defense Equity Valuations: The Boeing–China Event Study (2025–2026)

**Author:** Shivam Parti (Georgetown University)
**Repository:** github.com/partishivam-droid/Boeing_Study

## Overview

This repository contains the full replication package for an empirical event study analyzing Boeing's (NYSE: BA) equity valuation response to major U.S.–China trade tensions, aerospace export controls, and tariff announcements from 2025 through mid-2026.

The analysis uses a two-step quantitative framework:

1. **Data Ingestion & Event Study Windowing (Python):** Pulls daily market data via Yahoo Finance, estimates a static market-model baseline using Calendar Year 2024 data, calculates Cumulative Abnormal Returns (CAR) across six core 2025 event windows plus a 2026 out-of-sample extension, runs robustness and cross-firm comparison checks, and exports the clean returns dataset used by Stata.
2. **Econometric Verification (Stata):** Imports the structured time-series data to run a formal multivariate OLS regression with heteroskedasticity-robust (HC1) standard errors, evaluating the joint statistical significance of the trade shocks.

## File Structure

```
Boeing_Study/
├── boeing_returns_data.csv       # Intermediate daily returns dataset (Jan 2024 – May 2026)
├── boeing_car_script.py          # Primary CAR pipeline (CY2024 baseline, Table 1, Figure 1)
├── robustness_checker_table.py   # 12-specification sensitivity matrix (Table 2)
├── cross_firm_panel.py           # Peer comparison panel: Airbus, GE, Honeywell, RTX (Table 3)
├── volatility_analysis.py        # Pre-/post-event realized volatility ratios (Table 4)
├── out_of_sample_2026.py         # Out-of-sample extension for May 2026 order news (Table 5)
├── Boeing_V2.do                  # Primary Stata do-file: multivariate OLS + HC1 robust SEs
└── boeing_trade_war_study.png    # Output CAR trajectory line plot (Figure 1)
```

## Replication Steps

### Step 1: Python Data Processing & Tables

Requires Python 3.9+:
```
pip install pandas numpy statsmodels scipy yfinance matplotlib
```

Run each script from the repository root:
```
python boeing_car_script.py          # Table 1 CARs + Figure 1
python robustness_checker_table.py   # Table 2 robustness matrix
python cross_firm_panel.py           # Table 3 cross-firm comparison
python volatility_analysis.py        # Table 4 volatility analysis
python out_of_sample_2026.py         # Table 5 out-of-sample extension
```
`boeing_car_script.py` also exports `boeing_returns_data.csv`, which the Stata step below depends on.

### Step 2: Stata Econometric Analysis

Open Stata and run **`Boeing_V2.do`** — the current primary specification. The script will:
- Load `boeing_returns_data.csv` and `tsset` the daily time-series data.
- Generate indicator (dummy) variables for each 7-trading-day event window ([-1,+5]).
- Run the primary multivariate OLS regression with robust (HC1) standard errors.
- Execute the joint F-test of all six event dummies.

## Timeline of Evaluated Events

The study tracks a [-1, +5] trading-day window around each announcement:

- **Apr 2, 2025** — Liberation Day reciprocal tariffs announced
- **Apr 9, 2025** — Tariffs on Chinese goods raised to 145%
- **Apr 15, 2025** — China orders Boeing delivery halt
- **May 12, 2025** — Geneva trade truce announced; Boeing delivery ban lifted
- **Oct 10, 2025** — Trump threatens 100% tariff after China rare earth export expansion
- **Oct 30, 2025** — Busan Summit: US-China tariff and rare earth truce
- **May 14, 2026** *(out-of-sample)* — Trump announces China will order 200 Boeing jets (post Trump-Xi summit)
- **May 20, 2026** *(out-of-sample)* — China Commerce Ministry formally confirms the 200-aircraft order

## Summary of Empirical Results

**Table 1 — Primary Event Study Results ([-1,+5])**

| Date | Event | CAR | t-stat | Significant? |
|---|---|---|---|---|
| Apr 2, 2025 | Liberation Day reciprocal tariffs | -0.87% | -0.18 | No |
| Apr 9, 2025 | Tariffs on Chinese goods raised to 145% | +2.75% | 0.57 | No |
| Apr 15, 2025 | China orders Boeing delivery halt | +10.16% | 2.10 | Yes (5% level) |
| May 12, 2025 | Geneva truce; delivery ban lifted | +1.44% | 0.30 | No |
| Oct 10, 2025 | Trump 100% tariff threat (rare earths) | +0.84% | 0.17 | No |
| Oct 30, 2025 | Busan Summit truce | -5.67% | -1.17 | No |

**Multivariate OLS Model (Stata / `Boeing_V2.do`)**
- Baseline ITA beta: 1.1890 (t = 11.62, p = 0.000)
- China Delivery Ban dummy: +0.0155/day (t = 2.39, p = 0.017) → aggregate window impact ≈ 7 × 1.55% = +10.85%
- Joint significance test (all 6 event dummies): F(6, 493) = 1.14, p = 0.3363
- R² = 0.4021, N = 501

Please note that the Python and Stata code used in this study is AI-assisted.
