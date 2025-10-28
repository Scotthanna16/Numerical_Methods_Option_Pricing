from abc import abstractmethod, ABC


class BlackScholes(ABC):

    def __init__(self, S0, K, r, delta, sigma, T, option):
        self.S0 = S0
        self.K = K
        self.r = r
        self.delta = delta
        self.sigma = sigma
        self.T = T
        self.option = option

    @abstractmethod
    def getprice(self) -> float:
        pass
