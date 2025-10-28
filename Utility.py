import yfinance as yf
import numpy as np


def getMeanandVolatility(ticker, startdate, enddate):
    stock = yf.download(
        tickers=ticker, start=startdate, end=enddate, prepost=True, progress=False
    )
    returns = stock["Close"].pct_change()
    mu = returns.mean() * 250
    sigma = returns.std() * np.sqrt(250)
    return [mu, sigma]


# if __name__ == "__main__":
#     mean, vol = getMeanandVolatility("AAPL", "2024-10-26", "2025-10-27")
#     print(mean)
#     print(vol)
