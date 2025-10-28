from NumericalMethodBase import NumericalMethod
import numpy as np
from scipy.stats import norm
import math


class BinomialOptionPricing(NumericalMethod):
    def __init__(self, S0, K, r, delta, sigma, T, steps):
        super().__init__(S0, r, sigma, T, steps)
        self.K = K
        self.delta = delta

    def run(self, option: str):
        option = option.lower()
        u = np.exp(self.sigma * np.sqrt(self.h))
        d = 1.0 / u
        p = (np.exp((self.r - self.delta) * self.h) - d) / (u - d)
        p = np.clip(p, 0.0, 1.0)
        q = 1.0 - p
        discount = np.exp(-self.r * self.h)

        k = np.arange(0, self.steps + 1)
        S_T = self.S0 * (u**k) * (d ** (self.steps - k))

        if "call" in option:
            Price = np.maximum(S_T - self.K, 0.0)
        else:
            Price = np.maximum(self.K - S_T, 0.0)

        # backward induction
        for i in range(self.steps - 1, -1, -1):
            Price = discount * (p * Price[1 : i + 2] + q * Price[0 : i + 1])
            # if American: apply early exercise (uncomment if needed)
            if "american" in option:
                S_i = (
                    self.S0
                    * (u ** np.arange(0, i + 1))
                    * (d ** (i - np.arange(0, i + 1)))
                )

                if "call" in option:
                    exercise = np.maximum(S_i - self.K, 0)
                else:
                    exercise = np.maximum(self.K - S_i, 0)

                Price = np.maximum(Price, exercise)

        return Price[0]


# if __name__ == "__main__":
#     m1 = BinomialOptionPricing(
#         S0=100, K=100, r=0.07, delta=0.2, sigma=0.2, T=1, steps=200
#     )
#     m2 = BinomialOptionPricing(
#         S0=100, K=100, r=0.07, delta=0.2, sigma=0.2, T=1, steps=200
#     )

#     print("European Call:", m1.run(option="european call"))
#     print("American Call:", m2.run(option="american call"))
