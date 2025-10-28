import math as m
from BlackScholesBase import BlackScholes
from scipy.stats import norm


class BlackScholesEuropean(BlackScholes):
    def __init__(self, S0, K, r, delta, sigma, T, option):
        super().__init__(S0, K, r, delta, sigma, T, option)

    def getprice(self) -> float:
        option = self.option.lower()
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


# if __name__ == "__main__":
#     model = BlackScholesEuropean(
#         S0=100, K=110, r=0.07, delta=0.02, sigma=0.2, T=1, option="call"
#     )
#     price = model.getprice()

#     print("Price:", price)
