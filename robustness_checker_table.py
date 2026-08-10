import itertools
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

ticker = "BA"

events = [
    {"date": "2025-04-02", "desc": "Liberation Day reciprocal tariffs announced"},
    {"date": "2025-04-09", "desc": "Tariffs on Chinese goods raised to 145%"},
    {"date": "2025-04-15", "desc": "China delivery ban & 125% retaliatory tariff"},
    {"date": "2025-05-12", "desc": "Trade truce: Ban lifted (Mid-May)"},
    {"date": "2025-10-10", "desc": "Trump 100% rare earth tariff & BA parts threat"},
    {"date": "2025-10-30", "desc": "Busan Summit: US-China tariff & rare earth truce"},
]

BENCHMARKS = ["ITA", "^GSPC"]
ESTIMATION_WINDOWS = ["static_2024", "rolling_250d"]
EVENT_WINDOWS = [(-1, 3), (-1, 5), (-5, 10)]


def load_returns():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "boeing_returns_data.csv")
    
    returns = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    returns.columns = ["BA", "ITA"]
    
    if "^GSPC" not in returns.columns:
        import yfinance as yf
        sp500 = yf.download("^GSPC", start="2023-12-20", end="2026-01-01", auto_adjust=False)["Adj Close"]
        sp500_ret = np.log(sp500 / sp500.shift(1)).dropna()
        returns["^GSPC"] = sp500_ret
        
    return returns.dropna()


def estimate_baseline(returns, benchmark, event_date, estimation_window):
    if estimation_window == "static_2024":
        base = returns.loc["2024-01-01":"2024-12-31"]
    elif estimation_window == "rolling_250d":
        anchor_idx = returns.index.get_indexer([event_date], method="pad")[0]
        end_idx = max(anchor_idx - 30, 250)
        start_idx = max(end_idx - 250, 0)
        base = returns.iloc[start_idx:end_idx]
    else:
        raise ValueError(estimation_window)

    X = sm.add_constant(base[benchmark])
    y = base[ticker]
    model = sm.OLS(y, X).fit()
    alpha = model.params["const"]
    beta = model.params[benchmark]
    resid_std = np.std(model.resid, ddof=2)
    return alpha, beta, resid_std


def compute_car(returns, benchmark, alpha, beta, resid_std, event_date, window):
    start_off, end_off = window
    event_idx = returns.index.get_indexer([event_date], method="pad")[0]
    window_data = returns.iloc[event_idx + start_off : event_idx + end_off + 1].copy()

    expected = alpha + beta * window_data[benchmark]
    abnormal = window_data[ticker] - expected
    car = abnormal.cumsum().iloc[-1]
    t_stat = car / (resid_std * np.sqrt(len(window_data)))
    return car, t_stat


def sig_stars(t_stat):
    t = abs(t_stat)
    if t > 2.576:
        return "***"
    elif t > 1.96:
        return "**"
    elif t > 1.645:
        return "*"
    return " "


def main():
    returns = load_returns()

    rows = []
    for ev, benchmark, est_win, event_win in itertools.product(
        events, BENCHMARKS, ESTIMATION_WINDOWS, EVENT_WINDOWS
    ):
        ev_date = pd.to_datetime(ev["date"])
        try:
            alpha, beta, resid_std = estimate_baseline(returns, benchmark, ev_date, est_win)
            car, t_stat = compute_car(returns, benchmark, alpha, beta, resid_std, ev_date, event_win)
            rows.append({
                "Event": ev["date"],
                "Description": ev["desc"],
                "Benchmark": benchmark,
                "Estimation Window": est_win,
                "Event Window": f"[{event_win[0]:+d},{event_win[1]:+d}]",
                "CAR": f"{car:.2%}",
                "T-Stat": round(t_stat, 2),
                "Sig": sig_stars(t_stat),
            })
        except Exception as e:
            print(f"skipping {ev['date']} / {benchmark} / {est_win} / {event_win}: {e}")

    results = pd.DataFrame(rows)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, "robustness_battery_results.csv")
    results.to_csv(out_path, index=False)
    print(f"saved: {out_path}")

    apr15 = results[results["Event"] == "2025-04-15"]
    print("\nApril 15 (delivery ban) across all specs:")
    print(apr15.to_string(index=False))

    n_sig = (apr15["Sig"].str.strip() != "").sum()
    print(f"\nsignificant in {n_sig} / {len(apr15)} specifications")


if __name__ == "__main__":
    main()
    