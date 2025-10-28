from NumericalMethodBase import NumericalMethod
import numpy as np


class MonteCarloSimulation(NumericalMethod):
    def __init__(self, S0, K, r, div, sigma, T, steps, numSim):
        super().__init__(S0, r, sigma, T, steps)
        self.K = K
        self.div = div
        self.numSim = numSim

    def run(self) -> float:
        sim_steps = int(self.steps * self.T)
        St = np.zeros((sim_steps, self.numSim))
        St[0, :] = self.S0

        for i in range(1, sim_steps):
            z = np.random.randn(self.numSim)
            St[i, :] = St[i - 1, :] * np.exp(
                (self.r - self.div - 0.5 * self.sigma**2) * self.h
                + self.sigma * np.sqrt(self.h) * z
            )

        # Arithmetic mean price for each step Asian Options
        mean_prices = np.mean(St, axis=1)
        discounted_payoffs = np.maximum(mean_prices - self.K, 0) * np.exp(
            -self.r * self.T
        )

        return np.mean(discounted_payoffs)


# if __name__ == "__main__":
#     mc = MonteCarloSimulation(
#         S0=100, K=100, r=0.05, div=0.02, sigma=0.2, T=1, steps=250, numSim=1000000
#     )
#     mc_prices = mc.run()
#     print("Monte Carlo (first 5 discounted payoffs):", mc_prices)
