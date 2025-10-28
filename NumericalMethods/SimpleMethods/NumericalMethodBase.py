from abc import abstractmethod, ABC


class NumericalMethod(ABC):

    def __init__(self, S0, r, sigma, T, steps):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.steps = steps
        self.h = T / steps

    @abstractmethod
    def getprice(self) -> float:
        pass
