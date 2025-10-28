from abc import abstractmethod, ABC


class MonteCarloMethod(ABC):
    def __init__(self, S0, K, r, delta, sigma, T, steps, numSim, option):
        self.S0 = S0
        self.K = K
        self.r = r
        self.delta = delta
        self.sigma = sigma
        self.T = T
        self.steps = steps
        self.h = T / steps
        self.numSim = numSim
        self.option = option

    @abstractmethod
    def getprice(self) -> float:
        pass
