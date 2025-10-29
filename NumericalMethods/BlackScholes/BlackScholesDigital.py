import math as m
from BlackScholesBase import BlackScholes
from scipy.stats import norm


class BlackScholesDigital(BlackScholes):
    def __init__(self, S0, K, r, delta, sigma, T, option):
        super().__init__(S0, K, r, delta, sigma, T, option)

    def getprice(self, payoff: str) -> float:
        option = self.option.lower()
        d1 = (
            m.log(self.S0 / self.K)
            + (self.r - self.delta + 0.5 * self.sigma**2) * self.T
        ) / (self.sigma * m.sqrt(self.T))
        d2 = d1 - self.sigma * m.sqrt(self.T)

        if payoff.lower() == "cash":
            if "call" in option:
                price = m.exp(-self.r * self.T) * norm.cdf(d2)

            else:
                price = m.exp(-self.r * self.T) * norm.cdf(-d2)

        else:

            if "call" in option:
                price = self.S0 * m.exp(-self.delta * self.T) * norm.cdf(d1)

            else:
                price = self.S0 * m.exp(-self.delta * self.T) * norm.cdf(-d1)

        return price


# if __name__ == "__main__":
#     model = BlackScholesDigital(
#         S0=100, K=110, r=0.07, delta=0.02, sigma=0.2, T=1, option="call"
#     )
#     price = model.getprice("cash")

#     print("Cash Call:", price)

#     model2 = BlackScholesDigital(
#         S0=100, K=110, r=0.07, delta=0.02, sigma=0.2, T=1, option="put"
#     )
#     price = model2.getprice("cash")

#     print("Cash Put:", price)

#     model3 = BlackScholesDigital(
#         S0=100, K=110, r=0.07, delta=0.02, sigma=0.2, T=1, option="call"
#     )
#     price = model3.getprice("asset")

#     print("Asset Call:", price)

#     model4 = BlackScholesDigital(
#         S0=100, K=110, r=0.07, delta=0.02, sigma=0.2, T=1, option="put"
#     )
#     price = model4.getprice("asset")

#     print("Asset Put:", price)
