import numpy as np
import csv
import os.path
from pathlib import Path
import ray
from tqdm import tqdm
import yastn
import yastn.tn.mps as mps
import time
from confluence_old import vectorI, Lv, identity, calculate_current_period


def fname_evol(N, vS, gamma, U, parts_evol, D_evol, steps_evol, Schmidt_tol, order, method):
    path = Path(f"./results2/U={U:0.2f}/vS={vS:0.4f}/gamma={gamma:0.4f}/N={N}")
    path.mkdir(parents=True, exist_ok=True)
    fname = f"parts={parts_evol}_D_evol={D_evol}_st={Schmidt_tol:0.1e}_steps_evol={steps_evol}_{order}_{method}.npy"
    return path / fname


def fname_ness(N, vS, gamma, U, parts_evol, D_evol, steps_evol, Schmidt_tol, order, method, steps_ness):
    path = Path(f"./results2/U={U:0.2f}/vS={vS:0.4f}/gamma={gamma:0.4f}/N={N}")
    path.mkdir(parents=True, exist_ok=True)
    fname = f"parts={parts_evol}_D_evol={D_evol}_st={Schmidt_tol:0.1e}_steps_evol={steps_evol}_{order}_{method}_steps_ness={steps_ness}.npy"
    return path / fname


def fname_curr_ness(N, vS, gamma, U):
    path = Path(f"./results2/U={U:0.2f}/vS={vS:0.4f}/gamma={gamma:0.4f}/N={N}/currents_evol.csv")
    return path


def export_curr_ness(N, vS, gamma, U, parts_evol, D_evol, steps_evol, Schmidt_tol, order, method, D_ness, steps_ness, entropy_evol, entropy_ness, tm, currs):
    fname = fname_curr_ness(N, vS, gamma, U)

    fieldnames = ["parts", "D_evol", "steps_evol", "Schmidt_tol", "order", "method", "entropy_evol",
                  "D_ness", "steps_ness", "entropy_ness", "time", "mean_curr", "std_curr"]
    out = {"parts" : parts_evol,
           "D_evol" : D_evol,
           "steps_evol": steps_evol,
           "Schmidt_tol": Schmidt_tol,
           "order": order,
           "method": method,
           "entropy_evol": entropy_evol,
           "D_ness": D_ness,
           "steps_ness": steps_ness,
           "entropy_ness": max(entropy_ness),
           "time": tm,
           "mean_curr": np.mean(currs),
           "std_curr": np.std(currs)}
    fieldnames.append("currLS")
    out["currLS"] = currs[0]
    for n in range(1, N):
        name = f"curr_{n}"
        fieldnames.append(name)
        out[name] = currs[n]
    fieldnames.append("currSR")
    out["currSR"] = currs[-1]

    file_exists = os.path.isfile(fname)
    with open(fname, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
        if not file_exists:
            writer.writeheader()
        writer.writerow(out)


@ray.remote
def run_evol(N, vS, gamma, U, parts_evol, D_evol, steps_evol, Schmidt_tol, order, method):
    """ time-evolve Floque operator """
    ops = yastn.operators.SpinlessFermions(sym='U1')
    w = 1
    H = lambda t: Lv(t, N, vS, Delta=w, w=w, gamma=gamma, U=U, ops=ops)
    #
    opts_expmv = {'hermitian': False, 'tol': 1e-12}
    opts_svd = {"D_total": D_evol, 'tol': Schmidt_tol}
    #
    tf = (2 * np.pi / w)
    dtf = tf / parts_evol
    dt0 = dtf / steps_evol
    #
    evolUs = []
    t = 0
    time0 = time.time()
    for _ in range(parts_evol):
        H0 = H(t + dt0 / 2)
        #
        evolU = identity(N, ops) - (dt0 * H0) + (dt0 ** 2 / 2) * (H0 @ H0) - (dt0 ** 3 / 6) * (H0 @ H0 @ H0)
        evolU_ref = evolU.shallow_copy()
        evolU.canonize_(to='last').truncate_(to='first', opts_svd=opts_svd)
        mps.compression_(evolU, evolU_ref, method='1site', max_sweeps=50, Schmidt_tol=Schmidt_tol)
        #
        gen = mps.tdvp_(evolU, H, times=(t + dt0, t + dtf), dt=dt0,
                        u=1, method=method, order=order,
                        opts_svd=opts_svd, opts_expmv=opts_expmv, progressbar=True)
        next(gen)
        evolUs.append(evolU)
        t += dtf
    time1 = time.time()
    #
    data = {}
    data["evolUs"] = [evolU.save_to_dict() for evolU in evolUs]
    data["bds"] = [evolU.get_bond_dimensions() for evolU in evolUs]
    data["entropys"] = [evolU.get_entropy() for evolU in evolUs]
    schs = [evolU.get_Schmidt_values() for evolU in evolUs]
    data["schmidt"] = [[min(x._data) for x in sch] for sch in schs]
    data["time"] = time1 - time0
    fname = fname_evol(N, vS, gamma, U, parts_evol, D_evol, steps_evol, Schmidt_tol, order, method)
    np.save(fname, data, allow_pickle=True)


# @ray.remote
def run_ness(N, vS, gamma, U, parts_evol, D_evol, steps_evol, Schmidt_tol, order, method, Ds_ness, steps_ness):
    ops = yastn.operators.SpinlessFermions(sym='U1')
    #
    try:
        fname = fname_evol(N, vS, gamma, U, parts_evol, D_evol, steps_evol, Schmidt_tol, order, method)
        data = np.load(fname, allow_pickle=True).item()
        evolUs = [mps.load_from_dict(ops.config, evolU) for evolU in data["evolUs"]]
        entropy_evol = max(max(entropy for entropy in data["entropys"]))

    except FileNotFoundError:
        return None
    #
    data = {'psi': {}, 'bd': {},  'entropy': {}, 'currs': {}, 'std_currs': {}, 'sweeps': {}, 'time': {}, 'diffs': {}}
    if isinstance(Ds_ness, int):
        Ds_ness = [Ds_ness]

    psi = vectorI(N, ops)

    for D_ness in Ds_ness:
        t0 = time.time()
        opts_svd = {"D_total" : D_ness, 'tol': 1e-12}
        for _ in range(4):
            for evolU in evolUs:
                psi0 = mps.zipper(evolU, psi, opts_svd=opts_svd)
                mps.compression_(psi0, (evolU, psi), method='1site', max_sweeps=1)
                psi = psi0
        #
        diffs = []
        for step in tqdm(range(steps_ness)):
            psi_ref = psi.shallow_copy()
            for evolU in evolUs:
                psi0 = psi.shallow_copy()
                mps.compression_(psi0, (evolU, psi), method='1site', max_sweeps=1)
                psi = psi0
            diffs.append((psi - psi_ref).norm())
            if diffs[-1] < 1e-10:
                break

        print("steps = ", step, "diff = ", diffs[-1])
        w = 1
        H = lambda t: Lv(t, N, vS, Delta=w, w=w, gamma=gamma, U=U, ops=ops)
        currs = calculate_current_period(psi, H, ops, vS, gamma, w, opts_svd, steps=parts_evol * steps_evol)
        currs = np.mean(currs, axis=0)
        t1 = time.time()
        data["psi"][D_ness] = psi.save_to_dict()
        data["bd"][D_ness] = psi.get_bond_dimensions()
        data["entropy"][D_ness] = psi.get_entropy()
        data["currs"][D_ness] = currs
        data["std_currs"][D_ness] = np.std(currs)
        data["sweeps"][D_ness] = step
        data["time"][D_ness] = t1-t0
        data["diffs"][D_ness] = diffs
        #
        fname = fname_ness(N, vS, gamma, U, parts_evol, D_evol, steps_evol, Schmidt_tol, order, method, steps_ness)
        np.save(fname, data, allow_pickle=True)
        export_curr_ness(N, vS, gamma, U, parts_evol, D_evol, steps_evol, Schmidt_tol, order, method, D_ness, steps_ness, entropy_evol, data["entropy"][D_ness], t1-t0, currs)




if __name__ == "__main__":
    ray.init()
    #
    refs = []
    vS, gamma, U = 0.1, 0.1, 2.0

    for N in [25, ]:
        D_evol = 64
        parts_evol = 8
        steps_evol = 25
        Schmidt_tol = 1e-10
        order = '2nd'
        method = '12site'

        # job = run_evol.remote(N, vS, gamma, U, parts_evol, D_evol, steps_evol, Schmidt_tol, order, method)
        # refs.append(job)
        # ray.get(refs)

        Ds_ness = [64, 128, 256, 512, 1024]
        steps_ness = 1000
        run_ness(N, vS, gamma, U, parts_evol, D_evol, steps_evol, Schmidt_tol, order, method, Ds_ness, steps_ness)

