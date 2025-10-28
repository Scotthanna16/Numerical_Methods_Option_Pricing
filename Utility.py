import yfinance as yf
import numpy as np
import math as m
from scipy.stats import norm


def getMeanandVolatility(ticker, startdate, enddate):
    stock = yf.download(
        tickers=ticker, start=startdate, end=enddate, prepost=True, progress=False
    )
    returns = stock["Close"].pct_change()
    mu = returns.mean() * 250
    sigma = returns.std() * np.sqrt(250)
    return [mu, sigma]


def getGreeks(S0, K, r, delta, sigma, T, option: str):
    option = option.lower()
    d1 = (m.log(S0 / K) + (r - delta + 0.5 * sigma**2) * T) / (sigma * m.sqrt(T))
    d2 = d1 - sigma * m.sqrt(T)

    # Common factors
    Nd1 = norm.cdf(d1)
    nd1 = norm.pdf(d1)
    Nd2 = norm.cdf(d2)

    if "call" in option:
        Delta = m.exp(-delta * T) * Nd1
        Gamma = (m.exp(-delta * T) * nd1) / (S0 * sigma * m.sqrt(T))
        Vega = S0 * m.exp(-delta * T) * nd1 * m.sqrt(T)
        Theta = (
            -(S0 * nd1 * sigma * m.exp(-delta * T)) / (2 * m.sqrt(T))
            - r * K * m.exp(-r * T) * Nd2
            + delta * S0 * m.exp(-delta * T) * Nd1
        )
        Rho = K * T * m.exp(-r * T) * Nd2

    else:
        Delta = -m.exp(-delta * T) * norm.cdf(-d1)
        Gamma = (m.exp(-delta * T) * nd1) / (S0 * sigma * m.sqrt(T))
        Vega = S0 * m.exp(-delta * T) * nd1 * m.sqrt(T)
        Theta = (
            -(S0 * nd1 * sigma * m.exp(-delta * T)) / (2 * m.sqrt(T))
            + r * K * m.exp(-r * T) * norm.cdf(-d2)
            - delta * S0 * m.exp(-delta * T) * norm.cdf(-d1)
        )
        Rho = -K * T * m.exp(-r * T) * norm.cdf(-d2)

    return {
        "Delta": Delta,
        "Gamma": Gamma,
        "Vega": Vega,
        "Theta": Theta,
        "Rho": Rho,
    }


if __name__ == "__main__":
    mean, vol = getMeanandVolatility("AAPL", "2024-10-26", "2025-10-27")
    print(mean)
    print(vol)
    print(getGreeks(S0=100, K=110, r=0.07, delta=0.02, sigma=0.2, T=1, option="call"))
