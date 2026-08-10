import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import yfinance as yf

market = "ITA"

# Boeing plus comparison firms. Airbus is the placebo (non-US, not
# subject to the delivery ban). GE Aerospace / Honeywell / Spirit are
# mechanism-linked suppliers rather than generic "aerospace" names.
firms = {
    "BA": "Boeing",
    "EADSY": "Airbus",
    "GE": "GE Aerospace",
    "HON": "Honeywell",
    "RTX": "RTX (Collins Aerospace / Pratt & Whitney)",
}

events = [
    {"date": "2025-04-02", "desc": "Liberation Day reciprocal tariffs announced"},
    {"date": "2025-04-09", "desc": "Tariffs on Chinese goods raised to 145%"},
    {"date": "2025-04-15", "desc": "China delivery ban & 125% retaliatory tariff"},
    {"date": "2025-05-12", "desc": "Trade truce: Ban lifted (Mid-May)"},
    {"date": "2025-10-10", "desc": "Trump 100% rare earth tariff & BA parts threat"},
    {"date": "2025-10-30", "desc": "Busan Summit: US-China tariff & rare earth truce"},
]

event_start_offset = -1
event_end_offset = 5

DOWNLOAD_START = "2024-01-01"
DOWNLOAD_END = "2026-01-01"


def download_returns():
    tickers = list(firms.keys()) + [market]
    raw = yf.download(tickers, start=DOWNLOAD_START, end=DOWNLOAD_END, auto_adjust=False)["Adj Close"]
    returns = raw.pct_change().dropna()
    missing = [t for t in tickers if t not in returns.columns or returns[t].isna().all()]
    if missing:
        print(f"dropping tickers with no data: {missing}")
        for t in missing:
            firms.pop(t, None)
    return returns


def estimate_baseline(returns, ticker):
    base = returns.loc["2024-01-01":"2024-12-31"]
    X = sm.add_constant(base[market])
    y = base[ticker]
    model = sm.OLS(y, X).fit()
    alpha = model.params["const"]
    beta = model.params[market]
    resid_std = np.std(model.resid, ddof=2)
    return alpha, beta, resid_std


def compute_car(returns, ticker, alpha, beta, resid_std, event_date):
    event_idx = returns.index.get_indexer([event_date], method="pad")[0]
    window = returns.iloc[event_idx + event_start_offset : event_idx + event_end_offset + 1].copy()
    expected = alpha + beta * window[market]
    abnormal = window[ticker] - expected
    car = abnormal.cumsum().iloc[-1]
    t_stat = car / (resid_std * np.sqrt(len(window)))
    return car, t_stat


def sig_stars(t_stat):
    t = abs(t_stat)
    if t > 2.576:
        return "***"
    elif t > 1.96:
        return "**"
    elif t > 1.645:
        return "*"
    return ""


def main():
    returns = download_returns()

    rows = []
    for ticker, name in firms.items():
        alpha, beta, resid_std = estimate_baseline(returns, ticker)
        for ev in events:
            ev_date = pd.to_datetime(ev["date"])
            try:
                car, t_stat = compute_car(returns, ticker, alpha, beta, resid_std, ev_date)
                rows.append({
                    "Ticker": ticker,
                    "Firm": name,
                    "Event": ev["date"],
                    "Description": ev["desc"],
                    "CAR": car,
                    "CAR_pct": f"{car:.2%}",
                    "T-Stat": round(t_stat, 2),
                    "Sig": sig_stars(t_stat),
                })
            except Exception as e:
                print(f"skipping {ticker} / {ev['date']}: {e}")

    panel = pd.DataFrame(rows)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cross_firm_panel_results.csv")
    panel.to_csv(out_path, index=False)
    print(f"saved: {out_path}")

    print("\nApril 15 (delivery ban) across firms:")
    print(panel[panel["Event"] == "2025-04-15"][["Firm", "CAR_pct", "T-Stat", "Sig"]].to_string(index=False))

    # pooled panel regression with firm and event fixed effects, plus
    # a firm x event-3 (delivery ban) interaction to test whether the
    # April 15 effect is Boeing-specific
    panel["is_boeing"] = (panel["Ticker"] == "BA").astype(int)
    panel["is_april15"] = (panel["Event"] == "2025-04-15").astype(int)
    panel["boeing_x_april15"] = panel["is_boeing"] * panel["is_april15"]

    reg = smf.ols("CAR ~ C(Ticker) + C(Event) + boeing_x_april15", data=panel).fit(
        cov_type="HAC", cov_kwds={"maxlags": 1}
    )
    print("\nPanel regression (firm + event fixed effects, Boeing x April15 interaction):")
    print(reg.summary())


if __name__ == "__main__":
    main()
