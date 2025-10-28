from NumericalMethodBase import NumericalMethod
import numpy as np
import random


class EulerMaruyamaMethod(NumericalMethod):
    def __init__(self, S0, r, sigma, T, steps):
        super().__init__(S0, r=r, sigma=sigma, T=T, steps=steps)

    def getprice(self) -> float:

        t = np.arange(0, self.T, self.h)
        prices = [self.S0] + [None] * (len(t) - 1)
        x = self.S0

        for i in range(1, len(t)):
            x = (
                x
                + x * self.r * self.h
                + x * self.sigma * random.gauss(0, np.sqrt(self.h))
            )
            prices[i] = x

        return prices[-1]


# if __name__ == "__main__":
#     random.seed(42)
#     m = EulerMaruyamaMethod(100, 0.02, 0.05, 1, 252)
#     prices = m.getprice()
#     print(prices)
