from math import ceil
from typing import Union
from benchmark.backends import NumpyBackend, TorchBackend
from benchmark.truncator import SvdTruncator, QrTruncator, EigTruncator, QrTruncatorWithCBE


class iTEBD:
    """
    Conventions:
    MPS tensors, e.g. B have legs [vL, p, vR], i.e. left virtual, physical, right virtual
    S matrices either [vL, vR] for QR based TEBD (S is an actual matrix), or [v] for SVD based (S is diagonal)
    S_list[i] sits on the bond (i-1)----(i)
    """

    def __init__(self, B_list, S_list,
                 backend: Union[NumpyBackend, TorchBackend],
                 truncator: Union[SvdTruncator, QrTruncator, EigTruncator, QrTruncatorWithCBE],
                 cbe_increase_fraction: float = None,
                 ):
        self.B_list = [backend.asarray(B) for B in B_list]  # [vL, p, vR]
        self.L = len(B_list)  # length of unit cell
        self.S_list = [backend.asarray(S) for S in S_list]  # either [vL, vR] or [v]
        self.backend = backend
        self.truncator = truncator
        if isinstance(truncator, QrTruncatorWithCBE):
            assert cbe_increase_fraction is not None
        self.cbe_increase_fraction = cbe_increase_fraction

    def get_B(self, i):
        return self.B_list[i % self.L]

    def get_S(self, i):
        return self.S_list[i % self.L]

    def set_B(self, i, B):
        self.B_list[i % self.L] = B

    def set_S(self, i, S):
        self.S_list[i % self.L] = S

    def current_chi(self):
        return max(B.shape[0] for B in self.B_list)

    def sweep(self, bond_gates, chi_max, direction='R', num_qr_iters=1, Z_init_from_old_B=True, compute_err=True):
        """
        Perform a single TEBD sweep, i.e. apply every bond gate once, sequentially.
        bond_gates[i] is to be applied to sites (i, i+1) and has legs [p1, p2, p1*, p2*].
        returns a list of truncation errors
        """
        if direction == 'R':
            return self._right_sweep(bond_gates, chi_max, num_qr_iters=num_qr_iters,
                                     Z_init_from_old_B=Z_init_from_old_B, compute_err=compute_err)
        elif direction == 'L':
            self.mirror_mps()
            bond_gates = [self.backend.transpose(gate, (1, 0, 3, 2)) for gate in reversed(bond_gates)]
            res = self._right_sweep(bond_gates, chi_max, num_qr_iters=num_qr_iters,
                                     Z_init_from_old_B=Z_init_from_old_B, compute_err=compute_err)
            self.mirror_mps()
            return res
        else:
            raise ValueError

    def _right_sweep(self, bond_gates, chi_max, num_qr_iters=1, Z_init_from_old_B=True, compute_err=True):
        errs = []
        rel_norm_change = 0
        for i in range(self.L):
            C = self.backend.tensordot(self.get_B(i), self.get_B(i + 1), ((2,), (0,)))  # [vL, p1, p2, vR]
            C = self.backend.tensordot(C, bond_gates[i], ((1, 2), (2, 3)))  # [vL, vR, p1, p2]
            C = self.backend.transpose(C, (0, 2, 3, 1))  # [vL, p1, p2, vR]
            theta = self.truncator.apply_left_S(self.get_S(i), C)  # [vL, p1, p2, vR])

            if isinstance(self.truncator, QrTruncatorWithCBE):
                chi, d, *_ = theta.shape
                eta = max(100, ceil((1 + self.cbe_increase_fraction) * chi))
                if eta != chi:
                    Z_init_from_old_B = False
            else:
                eta = None

            X, Y, Z, N, trunc_err, theta_norm_sq = self.truncator.decompose(
                theta, chi_max=chi_max, num_iters=num_qr_iters,
                Z_init=self.get_B(i + 1) if Z_init_from_old_B else None,
                compute_err=compute_err, eta=eta
            )
            # <psi(t+dt)|psi(t+dt)> / <psi(t)|psi(t)> - 1
            rel_norm_change = theta_norm_sq - 1
            # inversion-free TEBD trick: want S^{-1} @ X @ Y here,
            # can use that C = S^{-1} @ theta = N * S^{-1} @ X @ Y @ Z ; and that Z is orthogonal
            new_B_i = self.backend.tensordot(C, self.backend.conj(Z), ((2, 3), (1, 2))) / N
            self.set_B(i, new_B_i)
            self.set_S(i + 1, Y)
            self.set_B(i + 1, Z)
            errs.append(trunc_err)
        if compute_err:
            err = sum(errs)
        else:
            err = None
        return err, rel_norm_change

    def mirror_mps(self):
        self.B_list = [self.backend.transpose(B, (2, 1, 0)) for B in reversed(self.B_list)]
        self.S_list = [self.truncator.mirror_S(S) for S in reversed(self.S_list)]

    def site_expvals(self, O_list):
        """
        expectation values of site operators. O_list[i] acts on site i and has legs [p, p*].
        does not assume hermiticity and computes complex values.
        """
        assert len(O_list) == self.L
        expvals = []
        for i in range(self.L):
            sB = self.truncator.apply_left_S(self.get_S(i), self.get_B(i))  # [vL, p, vR]
            C = self.backend.tensordot(sB, O_list[i], ((1,), (1,)))  # [vL, vR, p]
            val = self.backend.tensordot(C, self.backend.conj(sB), ((0, 1, 2), (0, 2, 1)))
            expvals.append(float(self.backend.real(val).item()))
        return expvals

    def bond_expvals(self, O_list):
        """expectation values of site operators. O_list[i] acts on sites i, i+1 and has legs [p1, p2, p1*, p2*]
        does not assume hermiticity and computes complex values."""
        assert len(O_list) == self.L
        expvals = []
        for i in range(self.L):
            sB = self.truncator.apply_left_S(self.get_S(i), self.get_B(i))  # [vL, p, vR]
            sBB = self.backend.tensordot(sB, self.get_B(i + 1), ((2,), (0,)))  # [vL, p1, p2, vR]
            C = self.backend.tensordot(sBB, O_list[i], ((1, 2), (2, 3)))  # [vL, vR, p1, p2]
            val = self.backend.tensordot(C, self.backend.conj(sBB), ((0, 2, 3, 1), (0, 1, 2, 3)))
            expvals.append(val.item())
        return expvals

    def entanglement_entropy(self, i=0):
        """half-chain von-Neumann entanglement entropy on bond (i-1)---(i)"""
        return self.truncator.vN_entropy(self.get_S(i))

    def mps_to_np(self):
        """
        Return the MPS in the following format:
        A tuple, the first entry is the list of B tensors as numpy arrays, the second the list of S tensors as np arrays
        """
        return [self.backend.to_np(B) for B in self.B_list], [self.backend.to_np(S) for S in self.S_list]
