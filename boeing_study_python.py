import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf

ticker = "BA"
market = "ITA"

events = [
    {
        "date": "2025-04-02",
        "desc": "Liberation Day reciprocal tariffs signed",
    },
    {
        "date": "2025-04-09",
        "desc": "Reciprocal tariffs hiked to 145% total",
    },
    {
        "date": "2025-04-15",
        "desc": "China delivery ban & 125% retaliatory tariff",
    },
    {
        "date": "2025-05-12",
        "desc": "Geneva truce announced (Ban lifted)",
    },
    {
        "date": "2025-10-10",
        "desc": "Trump 100% tariff threat (Rare earth retaliation)",
    },
    {
        "date": "2025-10-30",
        "desc": "Busan Summit trade agreement signed",
    },
]

event_start_offset = -1
event_end_offset = 5
total_days = event_end_offset - event_start_offset + 1

print("Downloading market data from Yahoo Finance...")
df = yf.download([ticker, market], start="2024-01-01", end="2026-01-01", auto_adjust=False)[
    "Adj Close"
]
returns = df.pct_change().dropna()
returns.columns = ["Boeing", "Market"]

baseline_returns = returns.loc["2024-01-01":"2024-12-31"]
X_base = sm.add_constant(baseline_returns["Market"])
y_base = baseline_returns["Boeing"]

market_model = sm.OLS(y_base, X_base).fit()
alpha = market_model.params["const"]
beta = market_model.params["Market"]
resid_std = np.std(market_model.resid, ddof=2)

print("\n=== BASELINE MARKET MODEL (CY 2024) ===")
print(f"Alpha (Daily): {alpha:.6f} | Beta (vs {market}): {beta:.4f}")
print(f"Baseline Residual Volatility: {resid_std:.4%}\n")

summary_table = []
plt.figure(figsize=(12, 7))

timeline_x = np.arange(event_start_offset, event_end_offset + 1)

for ev in events:
    try:
        ev_date = pd.to_datetime(ev["date"])
        event_idx = returns.index.get_indexer([ev_date], method="pad")[0]

        window_data = returns.iloc[
            event_idx + event_start_offset : event_idx + event_end_offset + 1
        ].copy()

        window_data["Expected"] = alpha + beta * window_data["Market"]
        window_data["Abnormal"] = window_data["Boeing"] - window_data["Expected"]
        window_data["CAR"] = window_data["Abnormal"].cumsum()

        final_car = window_data["CAR"].iloc[-1]
        car_t_stat = final_car / (resid_std * np.sqrt(len(window_data)))

        sig = " "
        if abs(car_t_stat) > 2.576:
            sig = "***"
        elif abs(car_t_stat) > 1.96:
            sig = "**"
        elif abs(car_t_stat) > 1.645:
            sig = "*"

        summary_table.append({
            "Date": ev["date"],
            "Final CAR": f"{final_car:.2%}",
            "T-Stat": f"{car_t_stat:.2f}",
            "Sig": sig,
            "Description": ev["desc"],
        })

        plt.plot(
            timeline_x,
            window_data["CAR"] * 100,
            marker="o",
            linewidth=2,
            label=f"{ev['date']}: {ev['desc'][:30]}...",
        )

    except Exception as e:
        print(f"Skipping date {ev['date']}: {e}")

summary_df = pd.DataFrame(summary_table)
print("=== EVENT STUDY SUMMARY ===")
print(summary_df.to_string(index=False))
print("\nSignificance codes: *** p<0.01, ** p<0.05, * p<0.10")

plt.axvline(x=0, color="red", linestyle="--", alpha=0.7, label="Event Day (0)")
plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)
plt.title("Boeing (BA) Cumulative Abnormal Returns (CAR) Around 2025 Trade Events")
plt.xlabel("Days Relative to Event Announcement")
plt.ylabel("Cumulative Abnormal Return (%)")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()

chart_filename = "boeing_trade_war_study.png"
plt.savefig(chart_filename, dpi=300)
print(f"\n[Success] Visual analysis chart saved as: {os.path.abspath(chart_filename)}")

returns.to_csv("boeing_returns_data.csv")
print(
    f"[Data Export] Clean dataset saved for Stata at: {os.path.abspath('boeing_returns_data.csv')}"
)