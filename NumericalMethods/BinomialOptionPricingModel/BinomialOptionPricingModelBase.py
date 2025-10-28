from abc import abstractmethod, ABC


class BinomialOptionPricingModel(ABC):

    def __init__(self, S0, K, r, delta, sigma, T, steps, option):
        self.S0 = S0
        self.K = K
        self.r = r
        self.delta = delta
        self.sigma = sigma
        self.T = T
        self.steps = steps
        self.h = T / steps
        self.option = option

    @abstractmethod
    def getprice(self) -> float:
        pass
