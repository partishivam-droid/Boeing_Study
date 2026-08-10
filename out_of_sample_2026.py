import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf

ticker = "BA"
market = "ITA"

# out-of-sample events: none of these were used to build the original
# CY2024 baseline or informed the original event selection
events_2026 = [
    {
        "date": "2026-05-14",
        "desc": "Trump announces China will order 200 Boeing jets (post Trump-Xi summit, informal)",
    },
    {
        "date": "2026-05-20",
        "desc": "China Commerce Ministry formally confirms 200 Boeing aircraft order",
    },
]

event_start_offset = -1
event_end_offset = 5

DOWNLOAD_START = "2024-01-01"
DOWNLOAD_END = "2026-08-01"


def download_returns():
    raw = yf.download([ticker, market], start=DOWNLOAD_START, end=DOWNLOAD_END, auto_adjust=False)["Adj Close"]
    returns = raw.pct_change().dropna()
    returns.columns = [ticker, market] if list(returns.columns) != [ticker, market] else returns.columns
    return returns


def main():
    returns = download_returns()

    # same baseline as the original study: static CY2024, ITA benchmark
    baseline = returns.loc["2024-01-01":"2024-12-31"]
    X = sm.add_constant(baseline[market])
    y = baseline[ticker]
    model = sm.OLS(y, X).fit()
    alpha = model.params["const"]
    beta = model.params[market]
    resid_std = np.std(model.resid, ddof=2)

    print(f"baseline alpha={alpha:.6f} beta={beta:.4f} resid_std={resid_std:.4%}\n")

    rows = []
    for ev in events_2026:
        ev_date = pd.to_datetime(ev["date"])
        try:
            event_idx = returns.index.get_indexer([ev_date], method="pad")[0]
            window = returns.iloc[event_idx + event_start_offset : event_idx + event_end_offset + 1].copy()
            window["Expected"] = alpha + beta * window[market]
            window["Abnormal"] = window[ticker] - window["Expected"]
            window["CAR"] = window["Abnormal"].cumsum()
            car = window["CAR"].iloc[-1]
            t_stat = car / (resid_std * np.sqrt(len(window)))
            rows.append({
                "Event": ev["date"],
                "Description": ev["desc"],
                "CAR": f"{car:.2%}",
                "T-Stat": round(t_stat, 2),
                "Sig": "***" if abs(t_stat) > 2.576 else "**" if abs(t_stat) > 1.96 else "*" if abs(t_stat) > 1.645 else "",
            })
        except Exception as e:
            print(f"skipping {ev['date']}: {e}")

    results = pd.DataFrame(rows)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_of_sample_2026_results.csv")
    results.to_csv(out_path, index=False)
    print(f"saved: {out_path}\n")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
