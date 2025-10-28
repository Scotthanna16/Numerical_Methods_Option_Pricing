from abc import ABC, abstractmethod


class FiniteDifference(ABC):
    def __init__(self, S0, K, r, delta, sigma, T, option, steps=100, Smax=5):
        self.S0 = S0
        self.K = K
        self.r = r
        self.delta = delta
        self.sigma = sigma
        self.T = T
        self.Smax = Smax * S0
        self.M = steps
        self.N = steps
        self.option = option

    @abstractmethod
    def getprice(self) -> float:
        pass
