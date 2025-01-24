import numpy as np
from tqdm import tqdm
import yastn
import yastn.tn.mps as mps

import sys
from pathlib import Path
path_to_source = (Path(__file__).absolute().parent / '../').resolve()
sys.path.append(str(path_to_source))
from src.confluence import vectorI, Lv, calculate_currents, identity, LXXZ, calculate_occupations, calculate_currents_XXZ


def run_test_RWA_evol(N, vS, gamma, U, ref_curr):
    print(" evol RWA ")
    ops = yastn.operators.SpinlessFermions(sym='U1')
    I = vectorI(N, ops)
    H = Lv(t=0, N=N, vS=vS/2, Delta=0, w=0, gamma=gamma, U=U, ops=ops)
    #
    psi = I.shallow_copy()
    opts_expmv = {'hermitian': False, 'tol': 1e-12}
    opts_svd = {"D_total": 64, 'tol': 1e-14}
    #
    U = identity(N, ops)
    dt0 = 0.01
    U = U - (dt0 * H) + (dt0**2 / 2) * (H @ H) - (dt0**3 / 6) * (H @ H @ H)
    U.canonize_(to='last').truncate_(to='first', opts_svd=opts_svd)
    #
    next(mps.tdvp_(U, H, times=(dt0, 1), dt=0.01,
                   u=1, method='12site',
                   opts_svd=opts_svd, opts_expmv=opts_expmv,
                   progressbar=True))
    #
    opts_svd = {"D_total" : 16, 'tol': 1e-12}
    for _ in tqdm(range(200)):
        psi0 = mps.zipper(U, psi, opts_svd=opts_svd)
        mps.compression_(psi0, (U, psi), method='1site', max_sweeps=10)
        psi = psi0

    currs = calculate_currents(psi, ops, vS / 2, gamma)
    print(currs)
    print([x - ref_curr for x in currs])
    assert all(abs(x-ref_curr) < 1e-3 for x in currs)



def run_test_LXXZ_evol(N, V, F, gamma, mu, D):
    print(" evol XXZ ")
    ops = yastn.operators.SpinlessFermions(sym='U1')
    I = vectorI(N, ops)
    H = LXXZ(N=N, V=V, F=F, gamma=gamma, mu=mu, ops=ops)
    #
    psi = I.shallow_copy()
    opts_expmv = {'hermitian': False, 'tol': 1e-12}
    opts_svd = {"D_total": 64, 'tol': 1e-14}
    #
    U = identity(N, ops)
    dt0 = 0.01
    U = U - (dt0 * H) + (dt0**2 / 2) * (H @ H) - (dt0**3 / 6) * (H @ H @ H)
    U.canonize_(to='last').truncate_(to='first', opts_svd=opts_svd)
    #
    next(mps.tdvp_(U, H, times=(dt0, 0.1), dt=0.01,
                   u=1, method='12site',
                   opts_svd=opts_svd, opts_expmv=opts_expmv,
                   progressbar=True))
    #
    opts_svd = {"D_total" : D, 'tol': 1e-12}
    for _ in tqdm(range(200)):
        psi0 = mps.zipper(U, psi, opts_svd=opts_svd)
        mps.compression_(psi0, (U, psi), method='1site', max_sweeps=2)
        psi = psi0

    currs = calculate_currents_XXZ(psi, ops, mu)
    print("currs = ", currs)
    occs = calculate_occupations(psi, ops)
    occs = [(x - 0.5) /mu for x in occs]
    print("occs = ", occs)


if __name__ == "__main__":
    run_test_LXXZ_evol(N=8, V=3, F=0.8, gamma=1, mu=0.01, D=16)

# if __name__ == "__main__":
#     #
#     N, vS, gamma = 4, 0.5, 1.0
#     # for non-interacting within RWA
#     # current is gamma/2 * vs**2 / (vs**2 + gamma**2)
#     # ref_curr = (gamma/2) * vS**2 / (vS**2 + gamma**2)
#     Ucurr = {0: 0.1,
#              1: 0.01341176470588,
#              2: 0.00325739833936,
#              3: 0.00141905064152,
#              4: 0.00141905064152}
#     for U in [0, 1, 2]:
#         print("U = ", U)
#         # run_test_RWA_tdvp(N, vS, gamma, U, Ucurr[U])
#         run_test_RWA_evol(N, vS, gamma, U, Ucurr[U])
#         # run_test_RWA_dmrg(N, vS, gamma, U, Ucurr[U])
