import numpy as np
import scipy.linalg

try:
    import torch
except ModuleNotFoundError:
    print('failed to import torch.')


# noinspection PyMethodMayBeStatic
class NumpyBackend:
    def __init__(self, dtype='complex128'):
        if isinstance(dtype, str):
            if dtype == 'complex128':
                dtype = np.complex128
            elif dtype == 'complex64':
                dtype = np.complex64
            else:
                raise ValueError(f'Could not parse numpy dtype from "{dtype}"')
        self.dtype = dtype

    def asarray(self, a):
        return np.asarray(a, self.dtype)

    def tensordot(self, a, b, axes, out=None):
        return np.tensordot(a, b, axes)

    def transpose(self, a, perm):
        return np.transpose(a, perm)

    def conj(self, a):
        return np.conj(a)

    def reshape(self, a, shape):
        return np.reshape(a, shape)

    def svd(self, a, compute_uv=True):
        return np.linalg.svd(a, full_matrices=False, compute_uv=compute_uv)

    def eig_based_svd_Vh(self, a):
        """
        - compute the eigensystem of hc(a) @ a = hc(Z) @ E @ Z
        - permute columns to make the diagonal of E in descending order (like singular values would be)
        - return sqrt(E) and Z, this means there is an X, which we do not compute, such that X, sqrt(E), Z
          is an SVD of a
        """
        ahc_dot_a = np.tensordot(np.conj(a), a, ((0,), (0,)))
        try:
            w, v = np.linalg.eigh(ahc_dot_a)
        except np.linalg.LinAlgError as e:
            print(f'error ignored, falling back to eig over eigh: {e}')
            w, v = np.linalg.eig(ahc_dot_a)
        # now np.allclose(w @ (E[:, None] * np.conj(w.T)), ahc_dot_a) == True
        w = np.abs(w)
        order = np.argsort(w)[::-1]
        S = np.sqrt(w[order])
        Vh = np.conj(v.T)[order, :]
        return S, Vh

    def eig_based_svd_U(self, a):
        """
        - compute the eigensystem of hc(a) @ a = hc(Z) @ E @ Z
        - permute columns to make the diagonal of E in descending order (like singular values would be)
        - return sqrt(E) and Z, this means there is an X, which we do not compute, such that X, sqrt(E), Z
          is an SVD of a
        """
        ahc_dot_a = np.tensordot(np.conj(a), a, ((0,), (0,)))
        try:
            w, v = np.linalg.eigh(ahc_dot_a)
        except np.linalg.LinAlgError as e:
            print(f'error ignored, falling back to eig over eigh: {e}')
            w, v = np.linalg.eig(ahc_dot_a)
        # now np.allclose(w @ (E[:, None] * np.conj(w.T)), ahc_dot_a) == True
        w = np.abs(w)
        order = np.argsort(w)[::-1]
        S = np.sqrt(w[order])
        U = v[order, :]
        return U, S

    def eig_based_svd_full(self, a):
        U, S = self.eig_based_svd_U(a)
        _, Vh = self.eig_based_svd_Vh(a)
        return U, S, Vh

    def eig_based_singvals(self, a):
        ahc_dot_a = np.tensordot(np.conj(a), a, ((0,), (0,)))
        try:
            w = np.linalg.eigvalsh(ahc_dot_a)
        except np.linalg.LinAlgError as e:
            print(f'error ignored, falling back to eig over eigh: {e}')
            w = np.linalg.eigvals(ahc_dot_a)
        w = np.abs(w)
        order = np.argsort(w)[::-1]
        S = np.sqrt(w[order])
        return S

    def qr(self, a):
        return np.linalg.qr(a)

    def lq(self, a):
        q, r = np.linalg.qr(a.T)
        return r.T, q.T

    def norm(self, a):
        return np.linalg.norm(a)

    def norm_sq(self, a):
        return np.real(np.sum(a * np.conj(a)))

    def real(self, a):
        return np.real(a)

    def vN_entropy(self, a):
        """von Neumann entropy of a state, given its schmidt-values as 1D array"""
        a = a[a > 1e-10] ** 2
        return -np.real(np.dot(np.log(a), a))

    def copy(self, a):
        return np.copy(a)

    def to_np(self, a):
        return a

    def synchronize(self):
        # wait for GPU computations to finish
        pass

    def polar(self, a, side='right'):
        return scipy.linalg.polar(a, side=side)


# noinspection PyMethodMayBeStatic
class TorchBackend:
    def __init__(self, device='cpu', dtype='complex128'):  # for gpu: device='cuda'

        if isinstance(dtype, str):
            if dtype == 'complex128':
                dtype = torch.complex128
            elif dtype == 'complex64':
                dtype = torch.complex64
            else:
                raise ValueError(f'Could not parse numpy dtype from "{dtype}"')
        self.dtype = dtype
        self.device = device

    def asarray(self, a):
        return torch.tensor(a, device=self.device, dtype=self.dtype)
    
    def tensordot(self, a, b, axes, out=None):
        return torch.tensordot(a, b, axes, out=out)

    def transpose(self, a, perm):
        return torch.permute(a, perm)

    def conj(self, a):
        return torch.conj(a)

    def reshape(self, a, shape):
        return torch.reshape(a, shape)

    def svd(self, a, compute_uv=True):
        if compute_uv:
            return torch.linalg.svd(a, full_matrices=False)
        else:
            return torch.linalg.svdvals(a)

    def eig_based_svd_Vh(self, a):
        """
        - compute the eigensystem of hc(a) @ a = hc(Z) @ E @ Z
        - permute columns to make the diagonal of E in descending order (like singular values would be)
        - return sqrt(E) and Z, this means there is an X, which we do not compute, such that X, sqrt(E), Z
          is an SVD of a
        """
        ahc_dot_a = torch.tensordot(torch.conj(a), a, ([0], [0]))
        # noinspection PyUnresolvedReferences
        try:
            w, v = torch.linalg.eigh(ahc_dot_a)
        except torch._C._LinAlgError as e:
            print(f'error ignored, falling back to eig over eigh: {e}')
            w, v = torch.linalg.eig(ahc_dot_a)
        w = torch.abs(w)
        order = torch.flip(torch.argsort(w), (0,))
        S = torch.sqrt(w[order])
        Vh = torch.conj(v.T)[order, :]
        return S, Vh

    # noinspection PyTypeChecker,PyUnresolvedReferences
    def eig_based_svd_U(self, a):
        """
        - compute the eigensystem of hc(a) @ a = hc(Z) @ E @ Z
        - permute columns to make the diagonal of E in descending order (like singular values would be)
        - return sqrt(E) and Z, this means there is an X, which we do not compute, such that X, sqrt(E), Z
          is an SVD of a
        """
        a_dot_ahc = torch.tensordot(a, torch.conj(a), ((1,), (1,)))
        try:
            w, u = torch.linalg.eigh(a_dot_ahc)
        except torch._C._LinAlgError as e:
            print(f'error ignored, falling back to eig over eigh: {e}')
            w, u = torch.linalg.eig(a_dot_ahc)
        w = torch.abs(w)
        order = torch.flip(torch.argsort(w), (0,))
        S = torch.sqrt(w[order])
        u = u[:, order]
        return u, S

    def eig_based_svd_full(self, a):
        U, S = self.eig_based_svd_U(a)
        _, Vh = self.eig_based_svd_Vh(a)
        return U, S, Vh

    def eig_based_singvals(self, a):
        ahc_dot_a = torch.tensordot(torch.conj(a), a, ([0], [0]))
        # noinspection PyUnresolvedReferences
        try:
            w = torch.linalg.eigvalsh(ahc_dot_a)
        except torch._C._LinAlgError as e:
            print(f'error ignored, falling back to eig over eigh: {e}')
            w = torch.linalg.eigvals(ahc_dot_a)
        w = torch.abs(w)
        order = torch.flip(torch.argsort(w), (0,))
        S = torch.sqrt(w[order])
        return S

    def qr(self, a):
        return torch.linalg.qr(a)

    def lq(self, a):
        q, r = torch.linalg.qr(a.T)
        return r.T, q.T

    def norm(self, a):
        return torch.linalg.norm(a)

    def norm_sq(self, a):
        return torch.real(torch.sum(a * torch.conj(a)))

    def real(self, a):
        return torch.real(a)

    def vN_entropy(self, a):
        """von Neumann entropy of a state, given its schmidt-values as 1D array"""
        a = torch.real(a)
        a = a[a > 1e-10] ** 2
        return -torch.dot(torch.log(a), a)

    def copy(self, a):
        return torch.clone(a)

    def to_np(self, a):
        return np.array(a.cpu())

    def synchronize(self):
        # wait for GPU computations to finish
        torch.cuda.synchronize(self.device)

    def eye(self, n, m=None):
        return torch.eye(n, m=m, dtype=self.dtype, device=self.device)

    def polar(self, a, side='right'):
        # polar decomposition of an m x n matrix a = U H with m >= n, (or if side is "left", a = H U).
        # where U is m x n isometric matrix and H an n x n hermitian matrix
        # iteration described in https://arxiv.org/pdf/2204.05693.pdf

        if side == 'right':
            pass
        elif side == 'left':
            # a.T = U @ H  ->  a = H.T @ U.T
            U, H = self.polar(a.T.conj())
            U = U.T.conj()
            return U, H
        else:
            raise ValueError(f'Illegal side. Expected "left" or "right". Got {side}.')

        norm_a = self.norm(a)
        U = a / norm_a
        Uhc_U = U.T.conj() @ U
        converged = False
        eye_n = self.eye(a.shape[1])
        for n in range(100):
            U = 1.5 * U - .5 * U @ Uhc_U
            Uhc_U = U.T.conj() @ U
            if self.norm(Uhc_U - eye_n) < 1e-10:  # TODO figure out good threshold
                break  # converged
        else:  # (no break)
            raise RuntimeError('polar did not converge')
        H = U.T.conj() @ a
        return U, H
