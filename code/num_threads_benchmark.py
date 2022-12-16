import os

from benchmark.backends import NumpyBackend, TorchBackend
from benchmark.clock_model import ClockModel
from benchmark.run import prepare_product_state
from benchmark.truncator import SvdTruncator, QrTruncator, QrTruncatorWithCBE, EigTruncator

os.environ['MKL_DYNAMIC'] = 'False'
os.environ['MKL_NUM_THREADS'] = '64'
os.environ['OMP_NUM_THREADS'] = '64'

import argparse
import pickle
import time
from pathlib import Path
from benchmark.tebd import iTEBD

g = 2
L = 2
N = 5
dt = 0.05
root = Path('/space/ga96vet/tebd-qr-and-gpu-powered/benchmark/data')


def main(chi: int, truncator: str = 'qr', backend: str = 'numpy', num_threads: int = 64):
    assert root.exists()
    outfile = root.joinpath(f'num_threads_benchmark/{truncator}_{backend}_{N}_{chi}_{num_threads}.txt')
    if outfile.exists():
        raise FileExistsError(f'already exists: {outfile}!')
    outfile.parent.mkdir(parents=True, exist_ok=True)

    mps_file = Path(f'/scratch/ga96vet/data/L_2_g_2_dt_0.05/qr_torch_cuda/N_{N}/chi_{chi}_save_mps/representative_mps.pkl')
    if not mps_file.exists():
        # try loading from qr_numpy instead
        mps_file = Path(f'/scratch/ga96vet/data/L_2_g_2_dt_0.05/qr_numpy/N_{N}/chi_{chi}_save_mps/representative_mps.pkl')

    with open(mps_file, 'rb') as f:
        try:
            mps_list = pickle.load(f)
        except Exception as e:
            raise type(e)(f'failed to unpickle {f}. error_msg: {e}') from e

    backend_in = backend
    if backend == 'numpy':
        backend = NumpyBackend()
    elif backend == 'torch':
        backend = TorchBackend(device='cuda')
    else:
        raise ValueError

    truncator_in = truncator
    if truncator == 'svd':
        truncator = SvdTruncator(backend=backend)
    elif truncator == 'qr':
        truncator = QrTruncator(backend=backend)
    elif truncator == 'qr_bond':
        truncator = QrTruncatorWithCBE(backend=backend)
    elif truncator == 'eig':
        truncator = EigTruncator(backend=backend)
    else:
        raise ValueError

    model = ClockModel(N=N, g=g)
    B_list, S_list = prepare_product_state(model.Z_eigenstate(0), L=L, backend=backend, truncator=truncator)

    B_list, S_list = mps_list[-1]
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
    #
    engine = iTEBD(B_list, S_list, backend=backend, truncator=truncator)

    # use large time step to quickly generate entanglement
    u_bond = backend.asarray(model.u_bond(dt=10 * dt, real_time=True))
    bond_gates = [u_bond] * L
    while engine.current_chi() < chi:
        print('increasing chi...')
        engine.sweep(bond_gates, chi_max=chi)

    u_bond = backend.asarray(model.u_bond(dt=dt, real_time=True))
    bond_gates = [u_bond] * L
    start = time.time()
    num = 10
    for n in range(10):
        print(f'starting step [{n + 1} / {num}]', end='\r')
        engine.sweep(bond_gates, chi_max=chi)
    runtime = time.time() - start
    print(runtime)
    with open(outfile, 'w') as f:
        print(str(runtime), file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--chi', type=int, default=128, help='maximum MPS bond dimension')
    parser.add_argument('--backend', type=str, default='numpy', choices=['numpy', 'torch'], help=' ')
    parser.add_argument('--truncator', type=str, default='qr', choices=['qr', 'qr_bond', 'svd', 'eig'], help=' ')
    parser.add_argument('--numThreads', type=int, default=64)
    args = parser.parse_args()
    os.environ['MKL_NUM_THREADS'] = str(args.numThreads)
    os.environ['OMP_NUM_THREADS'] = str(args.numThreads)
    main(chi=args.chi, truncator=args.truncator, backend=args.backend, num_threads=args.numThreads)
