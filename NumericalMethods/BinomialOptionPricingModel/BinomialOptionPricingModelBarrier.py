from BinomialOptionPricingModelBase import BinomialOptionPricingModel
from BinomialOptionPricingModelEuropean import BinomialOptionPricingModelEuropean
import numpy as np


class BinomialOptionPricingModelBarrier(BinomialOptionPricingModel):
    def __init__(self, S0, K, r, delta, sigma, T, steps, option):
        super().__init__(S0, K, r, delta, sigma, T, steps, option)

    def getprice(self, H, barrier_direction: str, barrier_action: str):
        u = np.exp(self.sigma * np.sqrt(self.h))
        d = 1.0 / u
        p = (np.exp((self.r - self.delta) * self.h) - d) / (u - d)
        p = np.clip(p, 0.0, 1.0)
        discount = np.exp(-self.r * self.h)

        # Stock prices at maturity
        k = np.arange(0, self.steps + 1)
        S_T = self.S0 * (u**k) * (d ** (self.steps - k))

        # Payoff
        if self.option.lower() == "call":
            V = np.maximum(S_T - self.K, 0)
        else:
            V = np.maximum(self.K - S_T, 0)

        # Barrier condition
        if "down" in barrier_direction:
            active = S_T > H
        else:  # "up"
            active = S_T < H

        if "out" in barrier_action:
            V = np.where(active, V, 0.0)

        # Backward induction
        for i in range(self.steps - 1, -1, -1):
            S_T = self.S0 * (d ** np.arange(i, -1, -1)) * (u ** np.arange(0, i + 1))
            V = discount * (p * V[1:] + (1 - p) * V[:-1])

            if "down" in barrier_direction:
                active = S_T > H
            else:
                active = S_T < H

            if "out" in barrier_action:
                V = np.where(active, V, 0.0)

        # Knock-in parity
        if "in" in barrier_action:
            out_price = self.getprice(H, barrier_direction, "out")
            vanilla_price = BinomialOptionPricingModelEuropean(
                self.S0,
                self.K,
                self.r,
                self.delta,
                self.sigma,
                self.T,
                self.steps,
                self.option,
            ).getprice()
            return vanilla_price - out_price

        return V[0]


# if __name__ == "__main__":
#     S, K, H = 100, 100, 90
#     r, q, sigma, T = 0.05, 0.0, 0.2, 1.0
#     option = BinomialOptionPricingModelBarrier(S, K, r, q, sigma, T, 1000, "call")
#     price = option.getprice(H, "down", "out")
#     print("Down-and-out Call:", price)

#     option = BinomialOptionPricingModelBarrier(S, K, r, q, sigma, T, 1000, "call")
#     price_in = option.getprice(H, "down", "in")
#     print("Down-and-in Call:", price_in)

#     option = BinomialOptionPricingModelBarrier(S, K, r, q, sigma, T, 1000, "call")
#     price = option.getprice(H, "up", "out")
#     print("Up-and-out Call:", price)

#     option = BinomialOptionPricingModelBarrier(S, K, r, q, sigma, T, 1000, "call")
#     price_in = option.getprice(H, "up", "in")
#     print("Up-and-in Call:", price_in)
