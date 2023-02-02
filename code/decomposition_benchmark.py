import os
threads = 64
os.environ['MKL_DYNAMIC'] = 'False'
os.environ['MKL_NUM_THREADS'] = str(threads)
os.environ['OMP_NUM_THREADS'] = str(threads)

import argparse
import pickle
import time
from pathlib import Path
from typing import Union

from .backends import NumpyBackend, TorchBackend
from .clock_model import ClockModel
from .truncator import SvdTruncator, QrTruncator, EigTruncator, QrTruncatorWithCBE

g = 2
dt = 0.05
root = Path('/space/ga96vet/tebd-qr-and-gpu-powered/benchmark/data')


def main(N: int, chi: int, truncator: str = 'qr', backend: str = 'numpy',
         dtype: str = 'complex128', file_suffix: str = '', num_threads: int = 64):
    assert root.exists()
    dtype_str = '_' + dtype if dtype != 'complex128' else ''
    outfolder = root.joinpath(f'decomposition_benchmark/{truncator}_{backend}_{N}_{chi}{dtype_str}{file_suffix}')
    if outfolder.exists():
        raise FileExistsError(f'already exists: {outfolder}!')
    outfolder.mkdir(parents=True, exist_ok=False)

    mps_file = Path(f'/scratch/ga96vet/data/L_2_g_2_dt_0.05/qr_torch_cuda/N_{N}/chi_{chi}_save_mps/representative_mps.pkl')
    if not mps_file.exists():
        # try loading from qr_numpy instead
        mps_file = Path(f'/scratch/ga96vet/data/L_2_g_2_dt_0.05/qr_numpy/N_{N}/chi_{chi}_save_mps/representative_mps.pkl')

    with open(mps_file, 'rb') as f:
        try:
            mps_list = pickle.load(f)
        except Exception as e:
            raise type(e)(f'failed to unpickle {f}. error_msg: {e}') from e

    runtime_benchmark(mps_list, N=N, chi=chi, outfolder=outfolder, truncator_str=truncator, backend_str=backend,
                      dtype=dtype)


def runtime_benchmark(mps_list, N, chi, outfolder: Path, truncator_str, backend_str, dtype):

    if backend_str == 'numpy':
        backend = NumpyBackend(dtype=dtype)
    elif backend_str == 'torch':
        backend = TorchBackend(device='cuda', dtype=dtype)
    else:
        raise ValueError

    if truncator_str == 'svd':
        truncator = SvdTruncator(backend=backend)
    elif truncator_str == 'qr':
        truncator = QrTruncator(backend=backend)
    elif truncator_str == 'qr_bond':
        truncator = QrTruncatorWithCBE(backend=backend)
    elif truncator_str == 'eig':
        truncator = EigTruncator(backend=backend)
    else:
        raise ValueError

    model = ClockModel(N=N, g=g)
    U = backend.asarray(model.u_bond(dt=dt, real_time=True))

    times = []
    for B_list, S_list in mps_list:
        B_list = [backend.asarray(B) for B in B_list]
        S_list = [backend.asarray(S) for S in S_list]
        t = single_run(B_list, S_list, U, backend, truncator)
        times.append(float(t))
        print(t)
        with open(outfolder.joinpath('times.txt'), 'a') as f:
            print(t, file=f)

    times = times[1:]  # discard warm-up run, cuda needs to load cuBLAS on the first call, which takes a short while.
    res = dict(times=times, g=g, dt=dt, N=N, backend=backend_str, truncator=truncator_str, chi=chi, threads=threads)
    with open(outfolder.joinpath('results.pkl'), 'wb') as f:
        pickle.dump(res, f)


def single_run(B_list, S_list, gate,
               backend: Union[NumpyBackend, TorchBackend],
               truncator: Union[SvdTruncator, QrTruncator, EigTruncator]):
    if isinstance(truncator, (SvdTruncator, QrTruncatorWithCBE)):
        # MPS is from QrTruncator, i.e. it has S as matrices, not as 1D arrays
        #
        # bond (1) -- (0)
        U, S, Vh = backend.eig_based_svd_full(S_list[0])
        B_list[1] = backend.tensordot(B_list[1], U, ((2,), (0,)))
        S_list[0] = S
        B_list[0] = backend.tensordot(Vh, B_list[0], ((1,), (0,)))

        # bond (0) -- (1)
        U, S, Vh = backend.eig_based_svd_full(S_list[1])
        B_list[0] = backend.tensordot(B_list[0], U, ((2,), (0,)))
        S_list[1] = S
        B_list[1] = backend.tensordot(Vh, B_list[1], ((1,), (0,)))

    start = time.time()
    chi1, d, chi2 = B_list[0].shape
    chi = max(chi1, chi2)
    C = backend.tensordot(B_list[0], B_list[1], ((2,), (0,)))  # [vL, p1, p2, vR]
    C = backend.tensordot(C, gate, ((1, 2), (2, 3)))  # [vL, vR, p1, p2]
    C = backend.transpose(C, (0, 2, 3, 1))  # [vL, p1, p2, vR]
    theta = truncator.apply_left_S(S_list[1], C)  # [vL, p1, p2, vR]
    X, Y, Z, N, trunc_err, theta_norm_sq = truncator.decompose(
        theta, chi_max=chi, num_iters=1, Z_init=B_list[1], compute_err=False, eta=int(round(1.1 * chi)),
    )
    new_B_i = backend.tensordot(C, backend.conj(Z), ((2, 3), (1, 2))) / N
    backend.synchronize()  # wait for asynchronous GPU computations to finish
    return time.time() - start


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--chi', type=int, default=128, help='maximum MPS bond dimension')
    parser.add_argument('--dim', type=int, default=5, help='local Hilbert space dimension')
    parser.add_argument('--backend', type=str, default='numpy', choices=['numpy', 'torch'], help=' ')
    parser.add_argument('--truncator', type=str, default='qr', choices=['qr', 'qr_bond', 'svd', 'eig'], help=' ')
    parser.add_argument('--dtype', type=str, default='complex128', choices=['complex128', 'complex64'])
    parser.add_argument('--fileSuffix', type=str, default='', help='appended to outfolder name')
    parser.add_argument('--numThreads', type=int, default=64, help='number of threads / CPU cores')
    args = parser.parse_args()
    os.environ['MKL_NUM_THREADS'] = str(args.numThreads)
    os.environ['OMP_NUM_THREADS'] = str(args.numThreads)
    main(N=args.dim, chi=args.chi, truncator=args.truncator, backend=args.backend, dtype=args.dtype,
         file_suffix=args.fileSuffix, num_threads=args.numThreads)
