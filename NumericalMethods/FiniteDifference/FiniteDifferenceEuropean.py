from FiniteDifferenceBase import FiniteDifference
import numpy as np


class FiniteDifferenceEuropean(FiniteDifference):

    def __init__(self, S0, K, r, delta, sigma, T, option, steps=100, Smax=5):
        super().__init__(S0, K, r, delta, sigma, T, option, steps, Smax)

    def getprice(self) -> float:
        option = self.option.lower()
        # Discretization
        S = np.linspace(0, self.Smax, self.M + 1)
        dS = S[1] - S[0]
        dt = self.T / self.N

        # Payoff at maturity
        if "call" in option:
            V = np.maximum(S - self.K, 0)
        elif "put" in option:
            V = np.maximum(self.K - S, 0)
        else:
            raise ValueError("Option type must be 'call' or 'put'")

        # Coefficients (now use actual S, not index)
        S_inner = S[1:-1]
        a = (
            0.25
            * dt
            * (
                (self.sigma**2 * S_inner**2 / dS**2)
                - (self.r - self.delta) * S_inner / dS
            )
        )
        b = -0.5 * dt * ((self.sigma**2 * S_inner**2 / dS**2) + self.r)
        c = (
            0.25
            * dt
            * (
                (self.sigma**2 * S_inner**2 / dS**2)
                + (self.r - self.delta) * S_inner / dS
            )
        )

        # Matrices for Crank–Nicolson
        A = np.diag(1 - b) + np.diag(-a[1:], k=-1) + np.diag(-c[:-1], k=1)
        B = np.diag(1 + b) + np.diag(a[1:], k=-1) + np.diag(c[:-1], k=1)

        # Time stepping (backward in time)
        for n in range(self.N):
            rhs = B @ V[1:-1]

            # Boundary conditions
            if "call" in option:
                rhs[0] += a[0] * (0)  # V(0,t) = 0 for call
                rhs[-1] += c[-1] * (
                    self.Smax - self.K * np.exp(-self.r * (self.T - n * dt))
                )
            else:  # put
                rhs[0] += a[0] * (self.K * np.exp(-self.r * (self.T - n * dt)))
                rhs[-1] += c[-1] * 0  # V(Smax,t) = 0 for put

            # Solve tridiagonal system
            V[1:-1] = np.linalg.solve(A, rhs)

        # Interpolate to find V(S0)
        price = np.interp(self.S0, S, V)
        return price


# if __name__ == "__main__":
#     fd_call = FiniteDifferenceEuropean(
#         S0=100,
#         K=110,
#         r=0.07,
#         delta=0.02,
#         sigma=0.2,
#         T=1,
#         option="call",
#         steps=200,
#     )
#     fd_put = FiniteDifferenceEuropean(
#         S0=100,
#         K=110,
#         r=0.07,
#         delta=0.02,
#         sigma=0.2,
#         T=1,
#         option="put",
#         steps=200,
#     )

#     print("European Call:", fd_call.getprice())
#     print("European Put:", fd_put.getprice())
