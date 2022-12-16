import os
os.environ['MKL_DYNAMIC'] = 'False'
os.environ['MKL_NUM_THREADS'] = '64'
os.environ['OMP_NUM_THREADS'] = '64'

import argparse
import logging
import numpy as np
import sys
import time
from pathlib import Path
import pickle

from tenpy.algorithms.tebd import TEBDEngine
from tenpy.linalg import np_conserved as npc
from tenpy.models.lattice import Chain
from tenpy.models.model import CouplingMPOModel, NearestNeighborModel
from tenpy.networks.mps import MPS
from tenpy.networks.site import Site

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


class ClockSite(Site):
    def __init__(self, N: int = 2, conserve=None):
        if not conserve:
            conserve = None
        if conserve is not None:
            raise NotImplementedError

        assert 0 < N == np.rint(N)
        self.N = N
        X = np.eye(N, k=1)
        X[-1, 0] = 1.
        Xphc = X + X.T.conj()
        Z = np.diag(np.exp(2.j * np.pi * np.arange(N) / N))
        ops = dict(X=X, Xphc=Xphc, Z=Z, Zhc=Z.T.conj())
        leg = npc.LegCharge.from_trivial(N)
        self.conserve = conserve
        names = [str(i) for i in range(N)]
        Site.__init__(self, leg=leg, state_labels=names, sort_charge=False, **ops)
        # noinspection PyTypeChecker
        self.state_labels['up'] = self.state_labels[names[0]]


class ClockModel(CouplingMPOModel, NearestNeighborModel):
    default_lattice = Chain
    force_default_lattice = True

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', None)
        if conserve is not None:
            raise NotImplementedError('Have not implemented charge conservation for clock model')

        N = model_params.get('N', 2)
        site = ClockSite(N=N, conserve=None)
        return site

    def init_terms(self, model_params):
        J = np.asarray(model_params.get('J', 1.))
        g = np.asarray(model_params.get('g', 1.))
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-g, u, 'Xphc')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-J, u1, 'Z', u2, 'Zhc', dx)
            self.add_coupling(-J, u1, 'Zhc', u2, 'Z', dx)


def tebd_run(chi: int = 512, N: int = 2, g: float = 2, dt: float = 0.05, num_steps: int = 200,
             L=2):
    outfolder = Path(output_root).expanduser().joinpath(
        f'L_{L}_g_{g}_dt_{dt}/tenpy/N_{N}/chi_{chi}'
    )
    if outfolder.exists():
        raise FileExistsError(f'Folder already exists: {outfolder}')
    outfolder.mkdir(parents=True, exist_ok=False)
    file_handler = logging.FileHandler(outfolder.joinpath('simulation.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    model_params = dict(N=N, L=2, J=1., g=g, bc_MPS='infinite', conserve=None)
    model = ClockModel(model_params)
    product_state = ['up'] * model.lat.N_sites
    psi = MPS.from_product_state(model.lat.mps_sites(), product_state, bc=model.lat.bc_MPS)
    tebd_params = dict(
        order=2, dt=dt, N_steps=1, trunc_params=dict(chi_max=chi, svd_min=1e-14)
    )
    engine = TEBDEngine(psi, model, tebd_params)

    results = dict(
        truncator='svd', backend='tenpy', device=None, N=N, g=g, L=L, dt=dt, chi_max=chi,
        trunc_err=[], times=[], entropy=[], run_times=[], z_expvals=[],
    )

    results['times'].append(0)
    results['entropy'].append(psi.entanglement_entropy())
    results['z_expvals'].append(psi.expectation_value('Z'))

    start_time = time.time()

    for n in range(num_steps):
        x = engine.run()
        err = np.sqrt(engine.trunc_err.eps)
        entropy = psi.entanglement_entropy()
        z_vals = psi.expectation_value('Z')
        run_time = time.time() - start_time

        results['trunc_err'].append(err)
        results['times'].append(float((n + 1) * dt))
        results['entropy'].append(entropy)
        results['z_expvals'].append(z_vals)
        results['run_times'].append(run_time)

        logger.info(f'{results["truncator"]}, N={N}, n={n+1:>3}, chi={max(psi.chi):>5}, '
                    f'S_vN={entropy}, err={err}, time: {format_time(run_time)}')

        with open(outfolder.joinpath('results.pkl'), 'wb') as f:
            pickle.dump(results, f)


def format_time(seconds):
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours}:{minutes:02d}:{seconds:02d}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--chi', type=int, default=128, help='maximum MPS bond dimension')
    parser.add_argument('--dim', type=int, default=10, help='local Hilbert space dimension')
    args = parser.parse_args()
    tebd_run(chi=args.chi, N=args.dim, g=2, dt=.05, num_steps=200)
