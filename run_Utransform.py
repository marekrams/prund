import numpy as np
import csv
import os.path
from pathlib import Path
import time
import ray
import yastn
import yastn.tn.mps as mps
from confluence_old import vectorI, L_Utransform_v, identity, calculate_currents_Utransform_RWA

def fname_URWA_evol(N, vS, gamma, U, D_evol, dt0, iter_evol):
    path = Path(f"./results_URWA/U={U:0.2f}/vS={vS:0.4f}/gamma={gamma:0.4f}/N={N}")
    path.mkdir(parents=True, exist_ok=True)
    fname = f"D_evol={D_evol}_dt0={dt0:0.4f}_steps={iter_evol}.npy"
    return path / fname

def fname_URWA_ness(N, vS, gamma, U, D_evol, dt0, iter_evol):
    path = Path(f"./results_URWA/U={U:0.2f}/vS={vS:0.4f}/gamma={gamma:0.4f}/N={N}")
    path.mkdir(parents=True, exist_ok=True)
    fname = f"D_evol={D_evol}_dt0={dt0:0.4f}_steps={iter_evol}_ness.npy"
    return path / fname

def export_URWA_curr_ness(N, vS, gamma, U, D_evol, dt0, iter_evol, D_ness, steps_ness, entropy_evol, error_evol, entropy_ness, diff, tm, currs):
    fname = Path(f"./results_URWA/U={U:0.2f}/vS={vS:0.4f}/gamma={gamma:0.4f}/N={N}/currents_ness.csv")

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

    file_exists = os.path.isfile(fname)
    with open(fname, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
        if not file_exists:
            writer.writeheader()
        writer.writerow(out)


@ray.remote(num_cpus=7)
def run_URWA_evol(N, vS, gamma, U, D_evol, dt0, iter_evol):
    """ time-evolve Floque operator """
    ops = yastn.operators.SpinlessFermions(sym='U1')
    w = 1
    H = L_Utransform_v(N, vS, Delta=w, w=w, gamma=gamma, U=U, ops=ops)
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
    fname = fname_URWA_evol(N, vS, gamma, U, D_evol, dt0, iter_evol)
    np.save(fname, data, allow_pickle=True)
    print("N =", N, "gamma =", gamma, "vS =", vS, "total error", total_error)
    return total_error


@ray.remote(num_cpus=2)
def run_URWA_ness(N, vS, gamma, U, D_evol, dt0, iter_evol, Ds_ness, steps_ness):
    ops = yastn.operators.SpinlessFermions(sym='U1')
    #
    try:
        fname = fname_URWA_evol(N, vS, gamma, U, D_evol, dt0, iter_evol)
        data = np.load(fname, allow_pickle=True).item()
        evolU = mps.load_from_dict(ops.config, data["evolU"])
        entropy_evol = data["entropy"]
        error_evol = data["error"]
    except FileNotFoundError:
        return None
    #
    fname = fname_URWA_ness(N, vS, gamma, U, D_evol, dt0, iter_evol)
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
        currs = calculate_currents_Utransform_RWA(psi, ops, vS / 2, gamma)
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
        fname = fname_URWA_ness(N, vS, gamma, U, D_evol, dt0, iter_evol)
        np.save(fname, data, allow_pickle=True)
        export_URWA_curr_ness(N, vS, gamma, U, D_evol, dt0, iter_evol, D_ness, data["sweeps"][D_ness], entropy_evol, error_evol, data["entropy"][D_ness], diffs[-1], data["time"][D_ness], currs)



if __name__ == "__main__":
    #
    refs = []
    U = 2.0
    for gamma in [0.1,]:
        for vS in [10 ** -1.5]:  # [10 ** -2, 10 ** -1.5, 10 ** -1, 10 ** -0.5, 10 ** 0]:
            D_evol = 64
            dt0 = 1 / 1024
            iter_evol = 8
            #
            # for N in [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25]:
            #     job = run_URWA_evol.remote(N, vS, gamma, U, D_evol, dt0, iter_evol)
            #     refs.append(job)
            #
            for N in [14, 16, 18, 20, 22]:
                for D_evol in [64]:
                    for iter_evol in [8]:
                        Ds_ness = [512]
                        steps_ness = 2 ** (19 - iter_evol - int(2 * np.log10(vS)))
                        job = run_URWA_ness.remote(N, vS, gamma, U, D_evol, dt0, iter_evol, Ds_ness, steps_ness)
                        refs.append(job)

    #
    # refs = []
    # for N in Ns:
    #     Ds_ness = [16, 32, 64, 128, 256, 512, 1024]
    #     steps_ness = 1000
    #     job = run_URWA_ness.remote(N, vS, gamma, U, D_evol, tf, steps_evol, Schmidt_tol, order, method, Ds_ness, steps_ness)
    #     refs.append(job)
    ray.get(refs)






#
# @ray.remote
# def run_URWA_ness(N, vS, gamma, U, D_evol, tf, steps_evol, Schmidt_tol, order, method, Ds_ness, steps_ness):
#     ops = yastn.operators.SpinlessFermions(sym='U1')
#     #
#     try:
#         fname = fname_URWA_evol(N, vS, gamma, U, D_evol, tf, steps_evol, Schmidt_tol, order, method)
#         data = np.load(fname, allow_pickle=True).item()
#         evolU = mps.load_from_dict(ops.config, data["evolU"])
#         entropy_evol = data["entropy"]
#
#     except FileNotFoundError:
#         return None
#     #
#     psi = vectorI(N, ops)
#     if isinstance(Ds_ness, int):
#         Ds_ness = [Ds_ness]
#     data = {'psi': {}, 'bd': {}, 'H2': {},  'entropy': {}, 'currs': {}, 'std_currs': {}, 'sweeps': {}, 'time': {}, 'diffs': {}}
#
#     for D_ness in Ds_ness:
#         opts_svd = {"D_total": D_ness, 'tol': 1e-12}
#         t0 = time.time()
#         for _ in range(4):
#             psi0 = mps.zipper(evolU, psi, opts_svd=opts_svd)
#             mps.compression_(psi0, (evolU, psi), method='1site', max_sweeps=2)
#             psi = psi0
#         #
#         diffs = []
#         for step in range(steps_ness):
#             psi0 = psi.copy()
#             mps.compression_(psi0, (evolU, psi), method='1site', max_sweeps=1)
#             diffs.append((psi - psi0).norm())
#             psi = psi0
#             if diffs[-1] < 1e-10:
#                 break
#         currs = calculate_currents_Utransform_RWA(psi, ops, vS / 2, gamma)
#         print("steps =", step, "diff =", diffs[-1], "std_curr =", np.std(currs))
#         t1 = time.time()
#         #
#         # data["psi"][D_ness] = psi.save_to_dict()
#         data["bd"][D_ness] = psi.get_bond_dimensions()
#         data["entropy"][D_ness] = psi.get_entropy()
#         data["currs"][D_ness] = currs
#         data["std_currs"][D_ness] = np.std(currs)
#         data["sweeps"][D_ness] = step
#         data["time"][D_ness] = t1-t0
#         data["diffs"][D_ness] = diffs
#         #
#         fname = fname_URWA_ness(N, vS, gamma, U, D_evol, tf, steps_evol, Schmidt_tol, order, method, steps_ness)
#         np.save(fname, data, allow_pickle=True)
#         export_URWA_curr_ness(N, vS, gamma, U, D_evol, tf, steps_evol, Schmidt_tol, order, method, D_ness, steps_ness, entropy_evol, data["entropy"][D_ness], t1-t0, currs)
#
# @ray.remote
# def run_URWA_dmrg(N, vS, gamma, U, Ds_ness, max_sweeps):
#     """ time-evolve Floque operator """
#     ops = yastn.operators.SpinlessFermions(sym='U1')
#     I = vectorI(N, ops)
#     w = 1
#     H = L_Utransform_v(N, vS, Delta=w, w=w, gamma=gamma, U=U, ops=ops)
#     #
#     H2 = H.H @ H
#     H2.canonize_(to='last')
#     H2.truncate_(to='first', opts_svd={'tol': 1e-12})
#     #
#     psi = I.shallow_copy()
#     if isinstance(Ds_ness, int):
#         Ds_ness = [Ds_ness]
#     data = {'psi': {}, 'bd': {}, 'H2': {},  'entropy': {}, 'currs': {}, 'std_currs': {}, 'sweeps': {}, 'time': {}}
#
#     for D_ness in Ds_ness:
#
#         t0 = time.time()
#         opts_svd = {"D_total": D_ness, 'tol': 1e-12}
#         step = mps.dmrg_(psi, H2, method='2site', max_sweeps=5, opts_svd=opts_svd)
#         step = mps.dmrg_(psi, H2, method='1site', max_sweeps=max_sweeps, energy_tol=1e-14)
#         currs = calculate_currents_Utransform_RWA(psi, ops, vS / 2, gamma)
#         t1 = time.time()
#         print("DMRG; std_curr = ", np.std(currs))
#         # data["psi"][D_ness] = psi.save_to_dict()
#         data["bd"][D_ness] = psi.get_bond_dimensions()
#         data["H2"][D_ness] = step.energy.real
#         data["entropy"][D_ness] = psi.get_entropy()
#         data["currs"][D_ness] = currs
#         data["std_currs"][D_ness] = np.std(currs)
#         data["sweeps"][D_ness] = step.sweeps
#         data["time"][D_ness] = t1-t0
#
#         fname = fname_URWA_dmrg(N, vS, gamma, U, max_sweeps)
#         np.save(fname, data, allow_pickle=True)
#         export_URWA_curr_dmrg(N, vS, gamma, U, max_sweeps, D_ness, data["entropy"][D_ness], data["H2"][D_ness], t1-t0, currs)
#
# @ray.remote
# def run_URWA_evol(N, vS, gamma, U, D_evol, tf, steps_evol, Schmidt_tol, order, method):
#     """ time-evolve Floque operator """
#     ops = yastn.operators.SpinlessFermions(sym='U1')
#     w = 1
#     H = L_Utransform_v(N, vS, Delta=w, w=w, gamma=gamma, U=U, ops=ops)
#     #
#     opts_expmv = {'hermitian': False, 'tol': 1e-12}
#     opts_svd = {"D_total": D_evol, 'tol': Schmidt_tol}
#     #
#     dt0 = tf / steps_evol
#     print(dt0)
#     #
#     print("H0 bd = ", H.get_bond_dimensions())
#     evolU = identity(N, ops) - (dt0 * H) + (dt0 ** 2 / 2) * (H @ H) - (dt0 ** 3 / 6) * (H @ H @ H)
#     print("evolU bd = ", evolU.get_bond_dimensions())
#
#     evolU_ref = evolU.copy()
#     evolU.canonize_(to='last').truncate_(to='first', opts_svd=opts_svd)
#     mps.compression_(evolU, evolU_ref, method='1site', max_sweeps=50, Schmidt_tol=Schmidt_tol)
#     print("evolU bd = ", evolU.get_bond_dimensions())
#     #
#     gen = mps.tdvp_(evolU, H, times=(dt0, tf), dt=dt0,
#                     u=1, method=method, order=order,
#                     opts_svd=opts_svd, opts_expmv=opts_expmv, progressbar=True)
#     next(gen)
#     #
#     print("evolU bd = ", evolU.get_bond_dimensions())
#     #
#     data = {}
#     data["evolU"] = evolU.save_to_dict()
#     data["bd"] = evolU.get_bond_dimensions()
#     data["entropy"] = evolU.get_entropy()
#     sch = evolU.get_Schmidt_values()
#     data["schmidt"] = [x.to_numpy() for x in sch]
#     fname = fname_URWA_evol(N, vS, gamma, U, D_evol, tf, steps_evol, Schmidt_tol, order, method)
#     np.save(fname, data, allow_pickle=True)
#
# def fname_URWA_dmrg(N, vS, gamma, U, max_sweeps):
#     path = Path(f"./results_URWA/U={U:0.2f}/vS={vS:0.4f}/gamma={gamma:0.4f}/N={N}")
#     path.mkdir(parents=True, exist_ok=True)
#     fname = f"dmrg_max_sweeps={max_sweeps}.npy"
#     return path / fname
#
# def fname_URWA_curr_dmrg(N, vS, gamma, U):
#     return Path(f"./results_URWA/U={U:0.2f}/vS={vS:0.4f}/gamma={gamma:0.4f}/N={N}/currents_dmrg.csv")
#
# def export_URWA_curr_dmrg(N, vS, gamma, U, max_sweeps, D_ness, entropy, H2_energy, tm, currs):
#     fname = fname_URWA_curr_dmrg(N, vS, gamma, U)
#
#     fieldnames = ["D_ness", "max_sweeps", "entropy_ness", "H2_ness", "time", "mean_curr", "std_curr"]
#     out = {"D_ness" : D_ness,
#            "max_sweeps": max_sweeps,
#            "entropy_ness": max(entropy),
#            "H2_ness": H2_energy,
#            "time": tm,
#            "mean_curr": np.mean(currs),
#            "std_curr": np.std(currs)}
#     fieldnames.append("currLS")
#     out["currLS"] = currs[0]
#     for n in range(1, N):
#         name = f"curr_{n}"
#         fieldnames.append(name)
#         out[name] = currs[n]
#     fieldnames.append("currSR")
#     out["currSR"] = currs[-1]
#
#     file_exists = os.path.isfile(fname)
#     with open(fname, 'a', newline='') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
#         if not file_exists:
#             writer.writeheader()
#         writer.writerow(out)
#
# def fname_URWA_evol(N, vS, gamma, U, D_evol, tf, steps_evol, Schmidt_tol, order, method):
#     path = Path(f"./results_URWA/U={U:0.2f}/vS={vS:0.4f}/gamma={gamma:0.4f}/N={N}")
#     path.mkdir(parents=True, exist_ok=True)
#     fname = f"D_evol={D_evol}_st={Schmidt_tol:0.1e}_tf={tf:0.4f}_steps_evol={steps_evol}_{order}_{method}.npy"
#     return path / fname
#
# def export_URWA_curr_ness(N, vS, gamma, U, D_evol, tf, steps_evol, Schmidt_tol, order, method, D_ness, steps_ness, entropy_evol, entropy_ness, tm, currs):
#     fname = fname_URWA_curr_ness(N, vS, gamma, U)
#
#     fieldnames = ["D_evol", "tf", "steps_evol", "Schmidt_tol", "order", "method", "entropy_evol",
#                   "D_ness", "steps_ness", "entropy_ness", "time", "mean_curr", "std_curr"]
#     out = {"D_evol" : D_evol,
#            "tf": tf,
#            "steps_evol": steps_evol,
#            "Schmidt_tol": Schmidt_tol,
#            "order": order,
#            "method": method,
#            "entropy_evol": max(entropy_evol),
#            "D_ness": D_ness,
#            "steps_ness": steps_ness,
#            "entropy_ness": max(entropy_ness),
#            "time": tm,
#            "mean_curr": np.mean(currs),
#            "std_curr": np.std(currs)}
#     fieldnames.append("currLS")
#     out["currLS"] = currs[0]
#     for n in range(1, N):
#         name = f"curr_{n}"
#         fieldnames.append(name)
#         out[name] = currs[n]
#     fieldnames.append("currSR")
#     out["currSR"] = currs[-1]

#     file_exists = os.path.isfile(fname)
#     with open(fname, 'a', newline='') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
#         if not file_exists:
#             writer.writeheader()
#         writer.writerow(out)
