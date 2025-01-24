import numpy as np
import tqdm
import yastn
import yastn.tn.mps as mps


def s(n):  # system
    return f"s{n}"


def a(n):  # ancila
    return f"a{n}"


def Lv(t, N, vS, Delta, w, gamma, U, ops):
    sc, scp, sn = ops.c(), ops.cp(), ops.n()
    ac, acp, an = ops.c().conj(), ops.cp().conj(), ops.n().conj()
    #
    sites = []  # order system and ancila sites
    for j in range(1, N+1):
        sites.append(s(j))
        sites.append(a(j))
    s2i = {s: i for i, s in enumerate(sites)}

    terms = []
    # d |rho> = -1 * [1j H ro -1j ro H - gamma (cp1 ro c1 + cN ro cpN) + gamma/2 (c1 cp1 ro + ro c1 cp1 + cpN cN ro + ro cpN cN)
    #
    # 1j H rho
    for j in range(1, N):
        terms.append((1j * vS * np.cos(w * t), [s(j), s(j+1)], [scp, sc]))
        terms.append((1j * vS * np.cos(w * t), [s(j+1), s(j)], [scp, sc]))
        terms.append((1j * U,  [s(j), s(j+1)], [sn, sn]))
    for j in range(1, N + 1):
        terms.append((1j * j * Delta, [s(j)], [sn]))
    #
    # -1j rho H
    for j in range(1, N):
        terms.append((-1j * vS * np.cos(w * t), [a(j), a(j+1)], [acp, ac]))
        terms.append((-1j * vS * np.cos(w * t), [a(j+1), a(j)], [acp, ac]))
        terms.append((-1j * U,  [a(j), a(j+1)], [an, an]))
    for j in range(1, N + 1):
        terms.append((-1j * j * Delta, [a(j)], [an]))
    #
    # injection
    terms.append((-gamma, [s(1), a(1)], [scp, acp]))
    terms.append((-gamma / 2, [s(1)], [sn]))
    terms.append((-gamma / 2, [a(1)], [an]))
    #
    # depletion
    terms.append((gamma, [s(N), a(N)], [sc, ac]))
    terms.append((gamma / 2, [s(N)], [sn]))
    terms.append((gamma / 2, [a(N)], [an]))
    #
    terms = [mps.Hterm(amp, [s2i[p] for p in pos], oprs) for amp, pos, oprs in terms]
    #
    one = mps.product_mpo([ops.I(), ops.I().conj()], 2 * N)
    return merge_sa(mps.generate_mpo(one, terms))


def LXXZ(N, V, F, gamma, mu, ops):
    sc, scp, sn = ops.c(), ops.cp(), ops.n()
    ac, acp, an = ops.c().conj(), ops.cp().conj(), ops.n().conj()
    snt = sn - ops.I() / 2
    ant = an - ops.I().conj() / 2
    #
    sites = []  # order system and ancilla sites
    for j in range(1, N+1):
        sites.append(s(j))
        sites.append(a(j))
    s2i = {s: i for i, s in enumerate(sites)}

    terms = []
    # d |rho> = -1 * [1j H ro -1j ro H - gamma (cp1 ro c1 + cN ro cpN) + gamma/2 (c1 cp1 ro + ro c1 cp1 + cpN cN ro + ro cpN cN)
    #
    # 1j H rho
    for j in range(1, N):
        terms.append((1j, [s(j), s(j+1)], [scp, sc]))
        terms.append((1j, [s(j+1), s(j)], [scp, sc]))
        terms.append((1j * V,  [s(j), s(j+1)], [snt, snt]))
    for j in range(1, N + 1):
        terms.append((1j * (j - N / 2) * F, [s(j)], [snt]))
    #
    # -1j rho H
    for j in range(1, N):
        terms.append((-1j, [a(j), a(j+1)], [acp, ac]))
        terms.append((-1j, [a(j+1), a(j)], [acp, ac]))
        terms.append((-1j * V,  [a(j), a(j+1)], [ant, ant]))
    for j in range(1, N + 1):
        terms.append((-1j * (j - N / 2) * F, [a(j)], [ant]))
    #
    # injection
    terms.append((-gamma * (1+mu), [s(1), a(1)], [scp, acp]))
    terms.append((-gamma * (1+mu) / 2, [s(1)], [sn]))
    terms.append((-gamma * (1+mu) / 2, [a(1)], [an]))
    #
    terms.append((-gamma * (1-mu), [s(N), a(N)], [scp, acp]))
    terms.append((-gamma * (1-mu) / 2, [s(N)], [sn]))
    terms.append((-gamma * (1-mu) / 2, [a(N)], [an]))
    #
    # depletion
    terms.append((gamma * (1+mu), [s(N), a(N)], [sc, ac]))
    terms.append((gamma * (1+mu) / 2, [s(N)], [sn]))
    terms.append((gamma * (1+mu) / 2, [a(N)], [an]))
    #
    terms.append((gamma * (1-mu), [s(1), a(1)], [sc, ac]))
    terms.append((gamma * (1-mu) / 2, [s(1)], [sn]))
    terms.append((gamma * (1-mu) / 2, [a(1)], [an]))
    #
    terms = [mps.Hterm(amp, [s2i[p] for p in pos], oprs) for amp, pos, oprs in terms]
    #
    one = mps.product_mpo([ops.I(), ops.I().conj()], 2 * N)
    return merge_sa(mps.generate_mpo(one, terms))


def vectorI(N, ops):
    I2 = mps.product_mps([ops.vec_n(0), ops.vec_n(0).conj()])
    cc = mps.product_mpo([ops.cp(), ops.cp().conj()])
    psi2 = (1 / np.sqrt(2)) * (I2 + cc @ I2)
    psi = mps.Mps(2 * N)
    for n in range(N):
        psi[2*n]   = psi2[0]
        psi[2*n+1] = psi2[1]
    return merge_sa(psi).canonize_(to='first')


def merge_sa(psi):
    N = psi.N // 2
    phi = mps.MpsMpoOBC(N, psi.nr_phys)
    axes = (0, (1, 2), 3) if psi.nr_phys == 1 else (0, (1, 3), 4, (2, 5))
    for n in range(N):
        tmp = yastn.tensordot(psi[2*n], psi[2*n+1], axes=(2, 0))
        phi[n] = tmp.fuse_legs(axes=axes)
    return phi


def occupation(j, N, ops):
    tmp = [ops.I(), ops.I().conj()] * (j - 1) + [ops.n(), ops.I().conj()] + [ops.I(), ops.I().conj()] * (N - j)
    return merge_sa(mps.product_mpo(tmp))


def identity(N, ops):
    tmp = [ops.I(), ops.I().conj()] * N
    return merge_sa(mps.product_mpo(tmp))


def corr_nn(j, N, ops):
    term = mps.Hterm(1, [2 * j - 2, 2 * j], [ops.cp(), ops.c()])
    one = mps.product_mpo([ops.I(), ops.I().conj()], 2 * N)
    return merge_sa(mps.generate_mpo(one, [term]))


def calculate_currents(psi, ops, vS, gamma):
    N = len(psi)
    I = vectorI(N, ops)
    O1 = occupation(1, N, ops)
    ON = occupation(N, N, ops)
    cc = [corr_nn(i, N, ops) for i in range(1, N)]
    norm = mps.vdot(I, psi)
    currs = [gamma * (1 - (mps.vdot(I, O1 @ psi) / norm).real)]
    for op in cc:
        currs.append(-2 * vS * (mps.vdot(I, op @ psi) / norm).imag)
    currs.append(gamma * (mps.vdot(I, ON @ psi) / norm).real)
    return currs


def calculate_currents_XXZ(psi, ops, mu):
    N = len(psi)
    I = vectorI(N, ops)
    O1 = occupation(1, N, ops)
    ON = occupation(N, N, ops)
    cc = [corr_nn(i, N, ops) for i in range(1, N)]
    norm = mps.vdot(I, psi)
    currs = [mu + (1 - 2 * (mps.vdot(I, O1 @ psi) / norm).real)]
    for op in cc:
        currs.append(-2 * (mps.vdot(I, op @ psi) / norm).imag)
    currs.append(mu + (2 * (mps.vdot(I, ON @ psi) / norm).real - 1))
    return currs


def calculate_occupations(psi, ops):
    N = len(psi)
    I = vectorI(N, ops)
    occs = []
    norm = mps.vdot(I, psi)
    for n in range(1, N+1):
        O = occupation(n, N, ops)
        occs.append((mps.vdot(I, O @ psi) / norm).real)
    return occs
