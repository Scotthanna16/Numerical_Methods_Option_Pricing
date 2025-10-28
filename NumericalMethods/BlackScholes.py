import numpy as np
import math as m
from scipy.stats import norm
from NumericalMethodBase import NumericalMethod


class BlackScholes(NumericalMethod):
    def __init__(self, S0, K, r, delta, sigma, T):
        super().__init__(S0, r, sigma, T, steps=1)
        self.K = K
        self.delta = delta

    def run(self, option: str):
        option = option.lower()
        d1 = (
            m.log(self.S0 / self.K)
            + (self.r - self.delta + 0.5 * self.sigma**2) * self.T
        ) / (self.sigma * m.sqrt(self.T))
        d2 = d1 - self.sigma * m.sqrt(self.T)

        if "call" in option:
            price = self.S0 * m.exp(-self.delta * self.T) * norm.cdf(
                d1
            ) - self.K * m.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            price = self.K * m.exp(-self.r * self.T) * norm.cdf(-d2) - self.S0 * m.exp(
                -self.delta * self.T
            ) * norm.cdf(-d1)

        return price

    def getGreeks(self, option: str):
        option = option.lower()
        d1 = (
            m.log(self.S0 / self.K)
            + (self.r - self.delta + 0.5 * self.sigma**2) * self.T
        ) / (self.sigma * m.sqrt(self.T))
        d2 = d1 - self.sigma * m.sqrt(self.T)

        # Common factors
        Nd1 = norm.cdf(d1)
        nd1 = norm.pdf(d1)
        Nd2 = norm.cdf(d2)

        if "call" in option:
            Delta = m.exp(-self.delta * self.T) * Nd1
            Gamma = (m.exp(-self.delta * self.T) * nd1) / (
                self.S0 * self.sigma * m.sqrt(self.T)
            )
            Vega = self.S0 * m.exp(-self.delta * self.T) * nd1 * m.sqrt(self.T)
            Theta = (
                -(self.S0 * nd1 * self.sigma * m.exp(-self.delta * self.T))
                / (2 * m.sqrt(self.T))
                - self.r * self.K * m.exp(-self.r * self.T) * Nd2
                + self.delta * self.S0 * m.exp(-self.delta * self.T) * Nd1
            )
            Rho = self.K * self.T * m.exp(-self.r * self.T) * Nd2

        else:
            Delta = -m.exp(-self.delta * self.T) * norm.cdf(-d1)
            Gamma = (m.exp(-self.delta * self.T) * nd1) / (
                self.S0 * self.sigma * m.sqrt(self.T)
            )
            Vega = self.S0 * m.exp(-self.delta * self.T) * nd1 * m.sqrt(self.T)
            Theta = (
                -(self.S0 * nd1 * self.sigma * m.exp(-self.delta * self.T))
                / (2 * m.sqrt(self.T))
                + self.r * self.K * m.exp(-self.r * self.T) * norm.cdf(-d2)
                - self.delta * self.S0 * m.exp(-self.delta * self.T) * norm.cdf(-d1)
            )
            Rho = -self.K * self.T * m.exp(-self.r * self.T) * norm.cdf(-d2)

        return {
            "Delta": Delta,
            "Gamma": Gamma,
            "Vega": Vega,
            "Theta": Theta,
            "Rho": Rho,
        }


# if __name__ == "__main__":
#     model = BlackScholes(S0=100, K=110, r=0.07, delta=0.02, sigma=0.2, T=1)
#     price = model.run(option="call")
#     greeks = model.getGreeks(option="call")

#     print("Price:", price)
#     print("Greeks:", greeks)
