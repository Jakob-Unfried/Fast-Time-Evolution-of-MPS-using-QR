import numpy as np
from scipy.linalg import expm


class ClockModel:

    def __init__(self, N: int, g: float = 0):
        """ H = -\sum_<i,j> (Z_i Z_j^† + hc) -g\sum_i (X_i + hc)"""
        self.N = N
        # clock matrix
        self.Z = np.diag([np.exp(2.j * np.pi * n / N) for n in range(N)])
        # shift matrix
        X = np.eye(N, k=1)
        X[-1, 0] = 1.
        self.X = X
        self.g = g

    def bond_hamiltonian(self):
        """[p1, p2, p1*, p2*]"""
        Z_Zhc = np.tensordot(self.Z, np.conj(self.Z.T), ((), ()))  # [p1, p1*, p2, p2*]
        interation = -(Z_Zhc + np.conj(np.transpose(Z_Zhc, (1, 0, 3, 2))))  # [p1, p1*, p2, p2*]
        Xphc = self.X + np.conj(self.X.T)  # [p,p*]
        Xphc_I = np.tensordot(Xphc, np.eye(self.N), ((), ()))
        I_Xphc = np.tensordot(np.eye(self.N), Xphc, ((), ()))
        field = -.5 * self.g * (Xphc_I + I_Xphc)  # factor .5 is for double counting "two bonds per site"
        res = interation + field
        return np.transpose(res, (0, 2, 1, 3))  # [p1, p2, p1*, p2*]

    def u_bond(self, dt: float, real_time=True):
        """e^{-i h_bond dt} for real_time, else e^{-h_bond dt}"""
        h = self.bond_hamiltonian()
        h = np.reshape(h, (self.N ** 2, self.N ** 2))  # [P,P*]
        factor = -1.j * dt if real_time else -dt
        u = expm(factor * h)
        return np.reshape(u, (self.N,) * 4)  # [p1, p2, p1*, p2*]

    def check_clock_algebra(self):
        assert self.X.shape == (self.N, self.N)
        assert self.Z.shape == (self.N, self.N)
        assert np.allclose(self.X @ self.Z, np.exp(2.j * np.pi / self.N) * self.Z @ self.X)
        assert np.allclose(matrix_power(self.X, self.N), np.eye(self.N))
        assert np.allclose(matrix_power(self.Z, self.N), np.eye(self.N))
        print('clock algebra is fulfilled')

    def check_eigenstates(self):
        for n in range(self.N):
            state = self.Z_eigenstate(n)
            assert np.allclose(self.Z @ state, self.Z_eigenvalue(n) * state)
        state = self.X_eigenstate()
        assert np.allclose(self.X @ state, state)
        print('eigensystem is correct')

    def Z_eigenstate(self, n):
        assert 0 <= n < self.N
        res = np.zeros((self.N,))
        res[n] = 1.
        return res

    def Z_eigenvalue(self, n):
        assert 0 <= n < self.N
        return np.exp(2.j * np.pi * n / self.N)

    def X_eigenstate(self):
        return np.ones((self.N,)) / np.sqrt(self.N)


def matrix_power(A, n: int):
    # not an optimal implementation.
    # in general possible to construct from fewer smaller powers
    if n == 0:
        return np.eye(*A.shape)
    if n == 1:
        return A
    if n == 2:
        return A @ A
    return matrix_power(A, n // 2) @ matrix_power(A, n - (n // 2))


if __name__ == '__main__':
    model = ClockModel(5)
    model.check_clock_algebra()
    model.check_eigenstates()
