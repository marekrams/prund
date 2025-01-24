import numpy as np
import csv
import os.path
from pathlib import Path
import time
import ray
import yastn
import yastn.tn.mps as mps
from src.confluence import vectorI, identity, LXXZ, calculate_occupations, calculate_currents_XXZ

def fname_XXZ_evol(N, V, F, mu, D_evol, dt0, iter_evol):
    path = Path(f"./results_XXZ/V={V:0.2f}/F={F:0.4f}/mu={mu:0.4f}/N={N}")
    path.mkdir(parents=True, exist_ok=True)
    fname = f"D_evol={D_evol}_dt0={dt0:0.4f}_steps={iter_evol}.npy"
    return path / fname

def fname_XXZ_ness(N, V, F, mu, D_evol, dt0, iter_evol):
    path = Path(f"./results_XXZ/V={V:0.2f}/F={F:0.4f}/mu={mu:0.4f}/N={N}")
    path.mkdir(parents=True, exist_ok=True)
    fname = f"D_evol={D_evol}_dt0={dt0:0.4f}_steps={iter_evol}_ness.npy"
    return path / fname

def export_XXZ_curr_ness(N, V, F, mu, D_evol, dt0, iter_evol, D_ness, steps_ness, entropy_evol, error_evol, entropy_ness, diff, tm, currs, occs):
    fname = Path(f"./results_XXZ/V={V:0.2f}/F={F:0.4f}/mu={mu:0.4f}/N={N}/currents_ness.csv")

    fieldnames = ["D_evol", "dt0", "iter_evol", "entropy_evol", "error_evol",
                  "D_ness", "steps_ness", "entropy_ness", "diff", "time", "mean_curr", "std_curr"]
    out = {"D_evol" : D_evol,
           "dt0": dt0,
           "iter_evol": iter_evol,
           "entropy_evol": max(entropy_evol),
           "error_evol": error_evol,
           "D_ness": D_ness,
           "steps_ness": steps_ness,
           "entropy_ness": max(entropy_ness),
           "diff": diff,
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
    for n in range(0, N):
        name = f"occ_{n}"
        fieldnames.append(name)
        out[name] = occs[n]

    file_exists = os.path.isfile(fname)
    with open(fname, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
        if not file_exists:
            writer.writeheader()
        writer.writerow(out)


@ray.remote(num_cpus=8)
def run_XXZ_evol(N, V, F, mu, D_evol, dt0, iter_evol):
    """ time-evolve Floque operator """
    ops = yastn.operators.SpinlessFermions(sym='U1')
    w = 1
    H = LXXZ(N, V, F, 1, mu, ops=ops)
    #
    opts_svd = {"D_total": D_evol}
    #
    evolU = identity(N, ops) - (dt0 * H) + (dt0 ** 2 / 2) * (H @ H) - (dt0 ** 3 / 6) * (H @ H @ H)
    evolU_ref = evolU.copy()
    err = evolU.canonize_(to='last').truncate_(to='first', opts_svd=opts_svd, )
    print(err)
    mps.compression_(evolU, evolU_ref, method='1site', max_sweeps=10, Schmidt_tol=1e-10)
    total_error = err
    #
    for step in range(iter_evol):
        tmp = evolU @ evolU
        err = tmp.canonize_(to='last').truncate_(to='first', opts_svd=opts_svd, )
        total_error = 2 * total_error + err
        print(step, err)
        mps.compression_(tmp, [evolU, evolU], method='1site', max_sweeps=20, Schmidt_tol=1e-10, opts_svd=opts_svd,)
        evolU = tmp
    #
    data = {}
    data["evolU"] = evolU.save_to_dict()
    data["bd"] = evolU.get_bond_dimensions()
    data["error"] = total_error
    data["entropy"] = evolU.get_entropy()
    sch = evolU.get_Schmidt_values()
    data["schmidt"] = [x.to_numpy() for x in sch]
    fname = fname_XXZ_evol(N, V, F, mu, D_evol, dt0, iter_evol)
    np.save(fname, data, allow_pickle=True)
    print("N=", N, "V=", V, "F=", F, 'mu=', mu, "total error", total_error)
    return total_error


@ray.remote(num_cpus=8)
def run_XXZ_ness(N, V, F, mu, D_evol, dt0, iter_evol, Ds_ness, steps_ness):
    ops = yastn.operators.SpinlessFermions(sym='U1')
    #
    try:
        fname = fname_XXZ_evol(N, V, F, mu, D_evol, dt0, iter_evol)
        data = np.load(fname, allow_pickle=True).item()
        evolU = mps.load_from_dict(ops.config, data["evolU"])
        entropy_evol = data["entropy"]
        error_evol = data["error"]
    except FileNotFoundError:
        return None
    #
    fname = fname_XXZ_ness(N, V, F, mu, D_evol, dt0, iter_evol)
    try:
        data = np.load(fname, allow_pickle=True).item()
    except FileNotFoundError:
        data = {'psi': {}, 'bd': {}, 'entropy': {}, 'currs': {}, 'std_currs': {}, 'sweeps': {}, 'time': {}, 'diffs': {}}

    if isinstance(Ds_ness, int):
        Ds_ness = [Ds_ness]

    for D_ness in Ds_ness:
        if any(k <= D_ness for k in data["psi"]):
            k = max(k for k in data["psi"] if k <= D_ness)
            psi = mps.load_from_dict(ops.config, data["psi"][k])
        else:
            psi = vectorI(N, ops)

        opts_svd = {"D_total": D_ness, 'tol': 1e-12}
        t0 = time.time()
        for _ in range(4):
            psi0 = mps.zipper(evolU, psi, opts_svd=opts_svd)
            mps.compression_(psi0, (evolU, psi), method='1site', max_sweeps=4)
            psi = psi0
        #
        diffs = []
        for step in range(steps_ness):
            psi0 = psi.copy()
            mps.compression_(psi0, (evolU, psi), method='1site', max_sweeps=1)
            diffs.append((psi - psi0).norm())
            psi = psi0
            if diffs[-1] < 1e-10:
                break
        currs = calculate_currents_XXZ(psi, ops, mu)
        occs = calculate_occupations(psi, ops)

        print("steps =", step, "diff =", diffs[-1], "std_curr =", np.std(currs), "error_evol =", error_evol)
        t1 = time.time()
        #
        data["psi"][D_ness] = psi.save_to_dict()
        data["bd"][D_ness] = psi.get_bond_dimensions()
        data["entropy"][D_ness] = psi.get_entropy()
        data["currs"][D_ness] = currs
        data["std_currs"][D_ness] = np.std(currs)
        data["diffs"][D_ness] = diffs
        if D_ness in data["sweeps"]:
            data["sweeps"][D_ness] += step
        else:
            data["sweeps"][D_ness] = step
        if D_ness in data["time"]:
            data["time"][D_ness] += t1-t0
        else:
            data["time"][D_ness] = t1-t0
        #
        fname = fname_XXZ_ness(N, V, F, mu, D_evol, dt0, iter_evol)
        np.save(fname, data, allow_pickle=True)
        export_XXZ_curr_ness(N, V, F, mu, D_evol, dt0, iter_evol, D_ness, data["sweeps"][D_ness], entropy_evol, error_evol, data["entropy"][D_ness], diffs[-1], data["time"][D_ness], currs, occs)



if __name__ == "__main__":
    #
    refs = []
    V = 3.0
    F = 0.8
    mu = 0.01
    D_evol = 96
    dt0 = 1 / 1024
    iter_evol = 6
    #
    # for N in [20, 50]:
    #     job = run_XXZ_evol.remote(N, V, F, mu, D_evol, dt0, iter_evol)
    #     refs.append(job)
    # ray.get(refs)
    # #
    # refs = []
    for N in [50,]:
        Ds_ness = [16, 32, 64, 128, 256, 512]
        steps_ness = 2048
        job = run_XXZ_ness.remote(N, V, F, mu, D_evol, dt0, iter_evol, Ds_ness, steps_ness)
        refs.append(job)
    ray.get(refs)

