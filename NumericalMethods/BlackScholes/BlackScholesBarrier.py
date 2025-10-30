import math as m
from BlackScholesBase import BlackScholes
from scipy.stats import norm
from BlackScholesEuropean import BlackScholesEuropean


class BlackScholesBarrier(BlackScholes):
    def __init__(self, S0, K, r, delta, sigma, T, option):
        super().__init__(S0, K, r, delta, sigma, T, option)

    def getprice(self, H, barrier_direction: str, barrier_action: str) -> float:

        lambda_ = (self.r - self.delta + 0.5 * self.sigma**2) / (self.sigma**2)

        HS_lambda = (H / self.S0) ** (2 * lambda_)
        HS_lambda2 = (H / self.S0) ** (2 * lambda_ - 2)

        if barrier_direction.lower() == "down":
            x1 = (
                m.log(S / H) / (self.sigma * m.sqrt(self.T))
            ) + lambda_ * self.sigma * m.sqrt(self.T)
            y1 = (
                m.log(H**2 / (self.S0 * self.K)) / (self.sigma * m.sqrt(self.T))
            ) + lambda_ * self.sigma * m.sqrt(self.T)
            x2 = x1 - self.sigma * m.sqrt(self.T)
            y2 = y1 - self.sigma * m.sqrt(self.T)

            if self.option.lower() == "call":
                out_price = self.S0 * m.exp(-self.delta * self.T) * (
                    norm.cdf(x1) - HS_lambda * norm.cdf(y1)
                ) - self.K * m.exp(-self.r * self.T) * (
                    norm.cdf(x2) - HS_lambda2 * norm.cdf(y2)
                )
            else:  # put
                out_price = self.K * m.exp(-self.r * self.T) * (
                    norm.cdf(-x2) - HS_lambda2 * norm.cdf(-y2)
                ) - self.S0 * m.exp(-self.delta * self.T) * (
                    norm.cdf(-x1) - HS_lambda * norm.cdf(-y1)
                )

        elif barrier_direction.lower() == "up":
            x1 = (
                m.log(self.S0 / H) / (self.sigma * m.sqrt(self.T))
            ) + lambda_ * self.sigma * m.sqrt(self.T)
            y1 = (
                m.log(H**2 / (self.S0 * self.K)) / (self.sigma * m.sqrt(self.T))
            ) + lambda_ * self.sigma * m.sqrt(self.T)
            x2 = x1 - self.sigma * m.sqrt(self.T)
            y2 = y1 - self.sigma * m.sqrt(self.T)

            if self.option.lower() == "call":
                out_price = self.S0 * m.exp(-self.delta * self.T) * (
                    norm.cdf(x1) - HS_lambda * norm.cdf(y1)
                ) - self.K * m.exp(-self.r * self.T) * (
                    norm.cdf(x2) - HS_lambda2 * norm.cdf(y2)
                )
            else:
                out_price = self.K * m.exp(-self.r * self.T) * (
                    norm.cdf(-x2) - HS_lambda2 * norm.cdf(-y2)
                ) - self.S0 * m.exp(-self.delta * self.T) * (
                    norm.cdf(-x1) - HS_lambda * norm.cdf(-y1)
                )

        if barrier_action.lower() == "out":
            return max(out_price, 0.0)
        elif barrier_action.lower() == "in":
            vanilla = BlackScholesEuropean(
                self.S0, self.K, self.r, self.delta, self.sigma, self.T, self.option
            )
            return max(vanilla.getprice() - out_price, 0.0)


if __name__ == "__main__":
    S, K, H = 100, 100, 90
    r, q, sigma, T = 0.05, 0.0, 0.2, 1.0
    option = BlackScholesBarrier(S, K, r, q, sigma, T, "call")
    price = option.getprice(H, "down", "out")
    print("Down-and-out Call:", price)

    option = BlackScholesBarrier(S, K, r, q, sigma, T, "call")
    price_in = option.getprice(H, "down", "in")
    print("Down-and-in Call:", price_in)
