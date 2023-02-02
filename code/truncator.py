from math import sqrt
from typing import Union
from .backends import NumpyBackend, TorchBackend


# noinspection PyMethodMayBeStatic
class SvdTruncator:
    """
    S are diagonal and we only store the diagonal entries as 1D arrays
    decomposition of theta is done via SVD
    """
    def __init__(self, backend: Union[NumpyBackend, TorchBackend]):
        self.backend = backend

    def apply_left_S(self, S, tens):
        """S[v]  ;  tens[vL,...]"""
        idx = (slice(None, None, None),) + (None,) * (len(tens.shape) - 1)
        return S[idx] * tens

    def apply_right_S(self, tens, S):
        """tens[..., vR]  ;  S[v]"""
        idx = (None,) * (len(tens.shape) - 1) + (slice(None, None, None),)
        return tens * S[idx]

    def decompose(self, theta, chi_max: int, threshold: float = 1e-14, **kwargs):
        """
        return an (approximate) decomposition theta ~= N * X @ Y @ Z
        
              -------theta-------  ~=   N  *   ----X----Y----Z----
                     |   |                         |         |
        
        where X is left-isometric, Y is diagonal and Z is right-isometric
        and the dimension chi of the new bonds is at most chi_max.

        return X, Y, Z, N, trunc_err, theta_norm_sq
        where trunc_err = norm(theta - approximation) / norm(theta)
        """
        try:
            chiL, d1, d2, chiR = theta.shape
        except ValueError as e:
            raise ValueError(f'Expected 4 axes. Got theta.shape={theta.shape}.') from e
        X, Y, Z = self.backend.svd(self.backend.reshape(theta, (chiL * d1, d2 * chiR)))
        chi = min(sum(Y > threshold), chi_max)
        theta_norm_sq = self.backend.norm_sq(Y)

        trunc_err = self.backend.norm_sq(Y[chi:]) / theta_norm_sq
        trunc_err = sqrt(float(trunc_err))
        # truncate
        Y = Y[:chi]
        X = self.backend.reshape(X[:, :chi], (chiL, d1, chi))
        Z = self.backend.reshape(Z[:chi, :], (chi, d2, chiR))

        N = self.backend.norm(Y)
        Y = Y / N

        return X, Y, Z, N, trunc_err, theta_norm_sq

    def vN_entropy(self, S):
        """von-Neumann entropy of the Schmidt-values, given an S-matrix"""
        return self.backend.vN_entropy(S)

    def mirror_S(self, S):
        """return the transpose of an S matrix, used when spatially mirroring an MPS"""
        return S


class EigTruncator(SvdTruncator):
    # replaces SVD by eig-based svd. S are still diagonal, stored as 1D arrays

    def decompose(self, theta, chi_max: int, threshold: float = 1e-14, **kwargs):
        try:
            chiL, d1, d2, chiR = theta.shape
        except ValueError as e:
            raise ValueError(f'Expected 4 axes. Got theta.shape={theta.shape}.') from e
        Y, Z = self.backend.eig_based_svd_Vh(self.backend.reshape(theta, (chiL * d1, d2 * chiR)))
        chi = min(sum(Y > threshold), chi_max)
        theta_norm_sq = self.backend.norm_sq(Y)
        trunc_err = self.backend.norm_sq(Y[chi:]) / theta_norm_sq
        trunc_err = sqrt(float(trunc_err))
        Y = Y[:chi]
        X = None  # iTEBD with inversion-free trick does not actually need X
        Z = self.backend.reshape(Z[:chi, :], (chi, d2, chiR))
        N = self.backend.norm(Y)
        Y = Y / N
        return X, Y, Z, N, trunc_err, theta_norm_sq


class QrTruncator:
    """
    S are lower triangular matrices
    decomposition of theta is done via iterative QRs
    """
    def __init__(self, backend: Union[NumpyBackend, TorchBackend]):
        self.backend = backend

    def apply_left_S(self, S, tens):
        """S[vL, vR]  ;  tens[vL, ...]"""
        return self.backend.tensordot(S, tens, ((1,), (0,)))

    def apply_right_S(self, tens, S):
        """tens[...,vR]  ;  S[vL,vR]"""
        return self.backend.tensordot(S, tens, ((-1,), (0,)))

    # noinspection PyUnboundLocalVariable
    def decompose(self, theta, chi_max: int, num_iters: int = 1, Z_init=None,
                  compute_err: bool = True, skip_initial_LQ: bool = True, **k):
        """
        return an (approximate) decomposition theta ~= N * X @ Y @ Z
        
              -------theta-------  ~=   N  *   ----X----Y----Z----
                     |   |                         |         |
        
        where X is left-isometric, Y is normalized and upper triangular and Z is right-isometric
        and the dimension chi of the new bonds is at most chi_max.

        Z_init, if given, is assumed to be isometric (i.e. right-orthogonal or B-form)

        return X, Y, Z, N, trunc_err

        where trunc_err is the relative truncation error
            norm(theta - approximation) / norm(theta)
        """
        try:
            chiL, d1, d2, chiR = theta.shape
        except ValueError as e:
            raise ValueError(f'Expected 4 axes. Got theta.shape={theta.shape}.') from e
        chi = min(chiL * d1, chi_max, d2 * chiR)
        theta = self.backend.reshape(theta, (chiL * d1, d2 * chiR))  # [L,R]

        if Z_init is None:
            Z = theta[:chi, :]
            if not skip_initial_LQ:
                _, Z = self.backend.lq(theta[:chi, :])  # [a,R]
        else:
            _chi, _, _ = Z_init.shape
            Z_init = self.backend.reshape(Z_init, (_chi, d2 * chiR))
            if _chi < chi:
                Z = self.backend.copy(theta[:chi, :])
                Z[:_chi] = Z_init
                if not skip_initial_LQ:
                    _, Z = self.backend.lq(Z)
            else:  # i.e. _chi >= chi
                Z = Z_init[:chi]

        for n in range(num_iters):
            M = self.backend.tensordot(theta, self.backend.conj(Z), ((1,), (1,)))
            X, Y = self.backend.qr(M)
            M = self.backend.tensordot(self.backend.conj(X), theta, ((0,), (0,)))
            Y, Z = self.backend.lq(M)

        theta_norm_sq = self.backend.norm_sq(theta)
        if compute_err:
            approximation = X @ Y @ Z
            trunc_err = self.backend.norm_sq(theta - approximation) / theta_norm_sq
            trunc_err = sqrt(float(trunc_err))
        else:
            trunc_err = None

        X = self.backend.reshape(X, (chiL, d1, chi))
        N = self.backend.norm(Y)
        Y = Y / N
        Z = self.backend.reshape(Z, (chi, d2, chiR))

        return X, Y, Z, N, trunc_err, theta_norm_sq

    def vN_entropy(self, S):
        """von-Neumann entropy of a state, given an S-matrix"""
        S = self.backend.eig_based_singvals(S)
        return self.backend.vN_entropy(S)

    def mirror_S(self, S):
        """return the transpose of an S matrix, used when spatially mirroring an MPS"""
        return self.backend.transpose(S, (1, 0))


# noinspection PyMethodMayBeStatic
class QrTruncatorWithCBE:
    """
    S are diagonal
    decomposition of theta is done via iterative QRs,
    followed by an eigh-powered SVD of the bond-tensor (Xi)
    """
    def __init__(self, backend: Union[NumpyBackend, TorchBackend]):
        self.backend = backend

    def apply_left_S(self, S, tens):
        """S[v]  ;  tens[vL,...]"""
        idx = (slice(None, None, None),) + (None,) * (len(tens.shape) - 1)
        return S[idx] * tens

    def apply_right_S(self, tens, S):
        """tens[..., vR]  ;  S[v]"""
        idx = (None,) * (len(tens.shape) - 1) + (slice(None, None, None),)
        return tens * S[idx]

    # noinspection PyUnboundLocalVariable
    def decompose(self, theta, chi_max: int, num_iters: int = 1, eta=None,
                  Z_init=None, compute_err: bool = True, skip_initial_LQ: bool = True,
                  threshold: float = 1e-14):
        try:
            chiL, d1, d2, chiR = theta.shape
        except ValueError as e:
            raise ValueError(f'Expected 4 axes. Got theta.shape={theta.shape}.') from e
        chi = min(chiL * d1, chi_max, d2 * chiR)
        theta = self.backend.reshape(theta, (chiL * d1, d2 * chiR))  # [L,R]

        if (eta is not None and eta != chi) or Z_init is None:
            if eta is None:
                eta = 1.1 * chi
            Z = theta[:eta, :]
            if not skip_initial_LQ:
                _, Z = self.backend.lq(theta[:chi, :])  # [a,R]
        else:
            _chi, _, _ = Z_init.shape
            Z_init = self.backend.reshape(Z_init, (_chi, d2 * chiR))
            if _chi < chi:
                Z = self.backend.copy(theta[:chi, :])
                Z[:_chi] = Z_init
                if not skip_initial_LQ:
                    _, Z = self.backend.lq(Z)
            else:  # i.e. _chi >= chi
                Z = Z_init[:chi]

        for n in range(num_iters):
            M = self.backend.tensordot(theta, self.backend.conj(Z), ((1,), (1,)))
            X, Y = self.backend.qr(M)
            M = self.backend.tensordot(self.backend.conj(X), theta, ((0,), (0,)))
            Y, Z = self.backend.lq(M)

        theta_norm_sq = self.backend.norm_sq(theta)
        if compute_err:
            approximation = X @ Y @ Z
            err_sq = self.backend.norm_sq(theta - approximation)
        else:
            err_sq = None

        S, V = self.backend.eig_based_svd_Vh(Y)
        Z = V @ Z

        chi = min(sum(S > threshold), chi_max)
        S = S[:chi]
        if compute_err:
            err_sq += self.backend.norm_sq(S[chi:])
            trunc_err = sqrt(float(err_sq / theta_norm_sq))
        else:
            trunc_err = None
        X = None  # not needed bc we use inversion-free trick
        Z = self.backend.reshape(Z[:chi, :], (chi, d2, chiR))

        N = self.backend.norm(S)
        S = S / N

        return X, S, Z, N, trunc_err, theta_norm_sq

    def vN_entropy(self, S):
        """von-Neumann entropy of the Schmidt-values, given an S-matrix"""
        return self.backend.vN_entropy(S)

    def mirror_S(self, S):
        """return the transpose of an S matrix, used when spatially mirroring an MPS"""
        return S


class PolarTruncator:

    def truncated_polar(self, M, delta=1e-12):
        """
        computes a truncated polar decomposition

        M = U' H'

        where if M is m x n we have an m x k isometry U' and a k x n matrix H'
        the rank k of the approximation is given by the number of singular values of M greater than delta.

        An equivalent decomposition could be achieved via svd, i.e.


        """