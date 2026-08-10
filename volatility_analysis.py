import os

import numpy as np
import pandas as pd
import yfinance as yf

ticker = "BA"
market = "ITA"

events = [
    {"date": "2025-04-02", "desc": "Liberation Day reciprocal tariffs announced"},
    {"date": "2025-04-09", "desc": "Tariffs on Chinese goods raised to 145%"},
    {"date": "2025-04-15", "desc": "China delivery ban & 125% retaliatory tariff"},
    {"date": "2025-05-12", "desc": "Trade truce: Ban lifted (Mid-May)"},
    {"date": "2025-10-10", "desc": "Trump 100% rare earth tariff & BA parts threat"},
    {"date": "2025-10-30", "desc": "Busan Summit: US-China tariff & rare earth truce (Late Oct)"},
]

# realized vol windows around each event: pre-event run-up vs. post-event
PRE_WINDOW = (-10, -1)    # 10 trading days before event day
POST_WINDOW = (1, 10)     # 10 trading days after event day

DOWNLOAD_START = "2024-01-01"
DOWNLOAD_END = "2026-01-01"


def download_returns():
    raw = yf.download([ticker, market], start=DOWNLOAD_START, end=DOWNLOAD_END, auto_adjust=False)["Adj Close"]
    return raw.pct_change().dropna()


def realized_vol(returns, event_date, offset_range):
    start_off, end_off = offset_range
    event_idx = returns.index.get_indexer([event_date], method="pad")[0]
    window = returns.iloc[event_idx + start_off : event_idx + end_off + 1]
    daily_vol = window[ticker].std()
    annualized = daily_vol * np.sqrt(252)
    return daily_vol, annualized


def baseline_vol(returns):
    # full CY2024 baseline realized vol, same period used for the market model
    base = returns.loc["2024-01-01":"2024-12-31"][ticker]
    return base.std(), base.std() * np.sqrt(252)


def main():
    returns = download_returns()
    base_daily, base_ann = baseline_vol(returns)

    rows = []
    for ev in events:
        ev_date = pd.to_datetime(ev["date"])
        try:
            pre_daily, pre_ann = realized_vol(returns, ev_date, PRE_WINDOW)
            post_daily, post_ann = realized_vol(returns, ev_date, POST_WINDOW)
            rows.append({
                "Event": ev["date"],
                "Description": ev["desc"],
                "Pre-Event Vol (Ann.)": f"{pre_ann:.1%}",
                "Post-Event Vol (Ann.)": f"{post_ann:.1%}",
                "Baseline CY2024 Vol (Ann.)": f"{base_ann:.1%}",
                "Pre/Baseline Ratio": round(pre_ann / base_ann, 2),
                "Post/Baseline Ratio": round(post_ann / base_ann, 2),
                "Post/Pre Ratio": round(post_ann / pre_ann, 2),
            })
        except Exception as e:
            print(f"skipping {ev['date']}: {e}")

    results = pd.DataFrame(rows)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "volatility_results.csv")
    results.to_csv(out_path, index=False)
    print(f"saved: {out_path}")
    print(f"\nBaseline CY2024 annualized vol: {base_ann:.1%}\n")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
