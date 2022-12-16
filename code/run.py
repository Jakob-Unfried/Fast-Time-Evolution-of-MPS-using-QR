import os
threads = 64
os.environ['MKL_DYNAMIC'] = 'False'
os.environ['MKL_NUM_THREADS'] = str(threads)
os.environ['OMP_NUM_THREADS'] = str(threads)

import argparse
import logging
import pickle
import sys
import numpy as np
from pathlib import Path
import time
from typing import Union

from benchmark.backends import NumpyBackend, TorchBackend
from benchmark.clock_model import ClockModel
from benchmark.tebd import iTEBD
from benchmark.truncator import QrTruncator, SvdTruncator, EigTruncator, QrTruncatorWithCBE

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
stdout_handler = logging.StreamHandler(stream=sys.stdout)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)


if Path('/space').exists():
    output_root = '/space/ga96vet/tebd-qr-and-gpu-powered/benchmark/data/'
else:
    output_root = '~/Desktop/tebd_debugging_runs/'
logger.info(f'output_root={output_root}')


def run_tebd(truncator='qr', use_old_B=True, backend='numpy', device='cpu',
             N=5, g=2, L=2, dt=0.05, num_steps=200, chi_max=128, save_mps=False,
             eval_z=0, dtype='complex128', file_suffix='',
             cbe_increase_fraction: float = .1, num_threads: int = 64):
    results = dict(
        truncator=truncator, backend=backend, device=device, N=N, g=g, L=L, dt=dt, chi_max=chi_max,
        threads=num_threads,
        trunc_err=[], norm_change=[], times=[], entropy=[], run_times=[], z_expvals=[],
        cbe_increase_fraction=cbe_increase_fraction
    )

    backend_str = 'numpy' if backend == 'numpy' else f'{backend}_{device}'
    dtype_str = '_' + dtype if dtype != 'complex128' else ''
    if truncator == 'qr_bond':
        cbe_string = f'_cbe_{round(cbe_increase_fraction, 3)}'
    else:
        cbe_string = ''
    outfolder = Path(output_root).expanduser().joinpath(
        f'L_{L}_g_{g}_dt_{dt}/{truncator}_{backend_str}/N_{N}/chi_{chi_max}_eval_{eval_z}{dtype_str}{cbe_string}{file_suffix}'
    )
    if outfolder.exists():
        raise FileExistsError(f'Folder already exists: {outfolder}')
    outfolder.mkdir(parents=True, exist_ok=False)
    file_handler = logging.FileHandler(outfolder.joinpath('simulation.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f'truncator={truncator}, backend={backend}, device={device}')
    logger.info(f'N={N}, g={g:.4f}, L={L}')
    logger.info(f'dt={dt:.4f}, chi_max={chi_max}')
    logger.info(f'cbe_increase_fraction={cbe_increase_fraction:.4f}')

    backend_in = backend
    if backend == 'numpy':
        backend = NumpyBackend(dtype=dtype)
    elif backend == 'torch':
        backend = TorchBackend(device=device, dtype=dtype)
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
    u_bond = backend.asarray(model.u_bond(dt=dt, real_time=True))
    bond_gates = [u_bond] * L
    Z = backend.asarray(model.Z)

    engine = iTEBD(B_list, S_list, backend=backend, truncator=truncator, cbe_increase_fraction=cbe_increase_fraction)

    results['times'].append(0)
    entropy = float(engine.entanglement_entropy())
    results['entropy'].append(entropy)
    if eval_z > 0:
        results['z_expvals'].append(engine.site_expvals([Z, Z]))

    logger.info(f'initial; chi={engine.current_chi()}, entropy={entropy:.4f}')
    start_time = time.time()

    saved_mps = []

    for n in range(num_steps):
        # for QrTruncator, evaluating truncation error and entropy is not free. only do it every 10-th step
        err, rel_norm_change = engine.sweep(bond_gates, chi_max=chi_max, num_qr_iters=1, Z_init_from_old_B=use_old_B,
                                            compute_err=True)

        if err is not None:
            err = float(err)

        entropy = float(engine.entanglement_entropy())

        if eval_z > 0 and n % eval_z == 0:
            z_vals = engine.site_expvals([Z, Z])
        else:
            z_vals = None

        run_time = time.time() - start_time

        results['trunc_err'].append(err)
        results['norm_change'].append(float(rel_norm_change))
        results['times'].append(float((n + 1) * dt))
        results['entropy'].append(entropy)
        results['z_expvals'].append(z_vals)
        results['run_times'].append(run_time)

        entropy_str = '  None' if entropy is None else f'{entropy:.4f}'
        err_str = '        None' if err is None else f'{err:e}'
        logger.info(f'{results["truncator"]}, N={N}, n={n+1:>3}, chi={engine.current_chi():>5}, '
                    f'S_vN={entropy_str}, err={err_str}, time: {format_time(run_time)}')

        if save_mps and n in [100, 105, 110, 115, 120, 125, 130, 135, 140, 145]:
            saved_mps.append(engine.mps_to_np())

        with open(outfolder.joinpath('results.pkl'), 'wb') as f:
            pickle.dump(results, f)

    if save_mps:
        file = Path('/scratch/ga96vet/data').joinpath(
            f'L_{L}_g_{g}_dt_{dt}/{truncator_in}_{backend_str}/N_{N}/chi_{chi_max}_save_mps/representative_mps.pkl'
        )
        file.parent.mkdir(parents=True, exist_ok=True)
        with open(file, 'wb') as f:
            pickle.dump(saved_mps, f)


def prepare_product_state(state, L, backend: Union[NumpyBackend, TorchBackend],
                          truncator: Union[SvdTruncator, QrTruncator, EigTruncator]):
    d = len(state)
    state = state / np.linalg.norm(state)
    B = np.reshape(np.array(state), (1, d, 1))
    if isinstance(truncator, QrTruncator):
        S = np.array([[1.]])
    else:
        S = np.array([1.])
    return [B] * L, [S] * L


def new_subfolder(folder: Path) -> Path:
    folder = folder.expanduser()
    for n in range(1000):
        trial = folder.joinpath(f'{n:03d}')
        if not trial.exists():
            break
    else:  # loop was not broken
        logger.error(f'Clean up directory {folder}!')
        raise RuntimeError
    trial.mkdir(parents=True, exist_ok=False)
    return trial


def format_time(seconds):
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours}:{minutes:02d}:{seconds:02d}'


def _str_to_bool(msg):
    if str(msg).lower() in ['false', 'no', 'f', 'n']:
        return False
    if str(msg).lower() in ['true', 'yes', 't', 'y']:
        return True
    raise ValueError(f'Could not convert to bool: {msg}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--chi', type=int, default=128, help='maximum MPS bond dimension')
    parser.add_argument('--dim', type=int, default=5, help='local Hilbert space dimension')
    parser.add_argument('--backend', type=str, default='numpy', choices=['numpy', 'torch'], help=' ')
    parser.add_argument('--truncator', type=str, default='qr', choices=['qr', 'qr_bond', 'svd', 'eig'], help=' ')
    parser.add_argument('--saveMps', type=_str_to_bool, default=False, help=' ')
    parser.add_argument('--evalZ', type=int, default=0, help='How often to evaluate Z expectation values (0 = never)')
    parser.add_argument('--dtype', type=str, default='complex128', choices=['complex128', 'complex64'])
    parser.add_argument('--fileSuffix', type=str, default='', help='appended to outfolders name')
    parser.add_argument('--cbeFrac', type=float, default=0.1, help='fraction by which eta increases in CBE')
    parser.add_argument('--numThreads', type=int, default=64)
    args = parser.parse_args()
    os.environ['MKL_NUM_THREADS'] = str(args.numThreads)
    os.environ['OMP_NUM_THREADS'] = str(args.numThreads)
    run_tebd(truncator=args.truncator, backend=args.backend, device='cuda', N=args.dim, chi_max=args.chi,
             save_mps=args.saveMps, eval_z=args.evalZ, dtype=args.dtype, file_suffix=args.fileSuffix,
             cbe_increase_fraction=args.cbeFrac, num_threads=args.numThreads)
