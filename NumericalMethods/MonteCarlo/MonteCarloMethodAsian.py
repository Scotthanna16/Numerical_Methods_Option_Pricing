from MonteCarloBase import MonteCarloMethod
import numpy as np


class MonteCarloMethodAsian(MonteCarloMethod):

    def __init__(self, S0, K, r, delta, sigma, T, steps, numSim, option):
        super().__init__(S0, K, r, delta, sigma, T, steps, numSim, option)

    def getprice(self) -> float:
        sim_steps = int(self.steps * self.T)
        St = np.zeros((sim_steps, self.numSim))
        St[0, :] = self.S0

        for i in range(1, sim_steps):
            z = np.random.randn(self.numSim)
            St[i, :] = St[i - 1, :] * np.exp(
                (self.r - self.delta - 0.5 * self.sigma**2) * self.h
                + self.sigma * np.sqrt(self.h) * z
            )

        # Arithmetic mean price for each step Asian Options
        mean_prices = np.mean(St, axis=1)
        if self.option.lower() == "call":
            discounted_payoffs = np.maximum(mean_prices - self.K, 0) * np.exp(
                -self.r * self.T
            )
        else:
            discounted_payoffs = np.maximum(-mean_prices + self.K, 0) * np.exp(
                -self.r * self.T
            )

        return np.mean(discounted_payoffs)


# if __name__ == "__main__":
#     mc = MonteCarloMethodAsian(
#         S0=100,
#         K=100,
#         r=0.05,
#         delta=0.02,
#         sigma=0.2,
#         T=1,
#         steps=250,
#         numSim=1000000,
#         option="call",
#     )
#     mc_prices = mc.getprice()
#     print("Monte Carlo (first 5 discounted payoffs):", mc_prices)
