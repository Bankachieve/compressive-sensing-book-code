# -*- coding: utf-8 -*-
"""สร้างรูปประกอบและผลการทดลองเชิงตัวเลขทั้งหมดของบทที่ 5 -> figures/fig05-*.pdf
ขั้นตอนวิธีทุกตัว (MP, OMP, CoSaMP, SP, IHT, NIHT, HTP, AMP, SOMP, MOD, K-SVD)
เขียนขึ้นเองด้วย NumPy/SciPy ตามสมการและรหัสเทียมในบท เพื่อให้ผู้อ่านตรวจสอบและทำซ้ำได้
ค่าตัวเลขทุกค่าที่อ้างในคำบรรยายภาพและเนื้อหาพิมพ์ออกทางจอเมื่อรันสคริปต์นี้
เลือกสร้างเฉพาะบางรูปได้ เช่น `python fig05_generate.py omp-trace amp`"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle  # กริดพื้นหลังและตำแหน่ง legend แบบเดียวกันทั้งเล่ม
from scipy.optimize import linprog
import os, sys, time

RUN = set(sys.argv[1:])
def want(name):
    return (not RUN) or (name in RUN)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT, exist_ok=True)
INK, ACC, GRY, RED, GRN, ORG = "#1F2A37", "#1D4ED8", "#9AA3AF", "#B91C1C", "#047857", "#B45309"
plt.rcParams.update({
    "font.size": 9, "axes.linewidth": 0.6, "axes.edgecolor": INK,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.labelcolor": INK, "lines.linewidth": 1.4,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf")
    plt.close(fig)
    print("saved", name)

def gaussian(m, n, rng):
    """เมทริกซ์แบบเกาส์ที่สมาชิกมีความแปรปรวน 1/m: E||phi_j||^2 = 1 แต่ไม่ได้ปรับนอร์มของแต่ละสดมภ์ให้เท่ากับ 1"""
    return rng.standard_normal((m, n)) / np.sqrt(m)

def sparse_signal(n, k, rng, kind="gauss"):
    x = np.zeros(n)
    S = rng.choice(n, k, replace=False)
    x[S] = rng.standard_normal(k) if kind == "gauss" else rng.choice([-1.0, 1.0], k)
    return x, np.sort(S)

def rel_err(xh, x):
    return np.linalg.norm(xh - x) / np.linalg.norm(x)

def ls_on(Phi, y, S):
    """กำลังสองน้อยที่สุดบนตัวรองรับ S: คืนเวกเตอร์ขนาด n ที่นอกตัวรองรับเป็นศูนย์"""
    x = np.zeros(Phi.shape[1])
    if len(S):
        x[S] = np.linalg.lstsq(Phi[:, S], y, rcond=None)[0]
    return x

def top_k(v, k):
    return np.argsort(np.abs(v))[::-1][:k]

def soft(z, t):
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)

# ------------------------------------------------------------
# ขั้นตอนวิธีละโมบ (หัวข้อ 5.2–5.3)
# ------------------------------------------------------------
def mp(Phi, y, iters):
    """matching pursuit (Mallat--Zhang 1993); คืนผลเฉลยและประวัติของ ||r||_2"""
    x = np.zeros(Phi.shape[1]); r = y.copy(); hist = [np.linalg.norm(r)]
    norms2 = np.sum(Phi ** 2, axis=0)
    for _ in range(iters):
        c = Phi.T @ r
        j = np.argmax(np.abs(c) / np.sqrt(norms2))
        a = c[j] / norms2[j]
        x[j] += a; r = r - a * Phi[:, j]; hist.append(np.linalg.norm(r))
    return x, np.array(hist)

def omp(Phi, y, k=None, tol=None, return_trace=False):
    """orthogonal matching pursuit (Pati และคณะ 1993) หยุดเมื่อครบ k รอบ หรือ ||r||_2 <= tol"""
    m, n = Phi.shape
    kmax = k if k is not None else m
    S = []; r = y.copy(); x = np.zeros(n); hist = [np.linalg.norm(r)]; picks = []
    for _ in range(kmax):
        if tol is not None and np.linalg.norm(r) <= tol:
            break
        j = int(np.argmax(np.abs(Phi.T @ r)))
        S.append(j); picks.append(j)
        x = ls_on(Phi, y, S)
        r = y - Phi @ x; hist.append(np.linalg.norm(r))
    if return_trace:
        return x, np.array(hist), picks
    return x

def cosamp(Phi, y, k, iters=50, tol=1e-10):
    """CoSaMP (Needell--Tropp 2009)"""
    n = Phi.shape[1]; x = np.zeros(n); r = y.copy()
    for _ in range(iters):
        Omega = top_k(Phi.T @ r, 2 * k)
        T = np.union1d(Omega, np.flatnonzero(x))
        b = ls_on(Phi, y, T)
        S = top_k(b, k); x_new = np.zeros(n); x_new[S] = b[S]
        r = y - Phi @ x_new
        if np.linalg.norm(x_new - x) <= tol * max(np.linalg.norm(x_new), 1e-300):
            x = x_new; break
        x = x_new
        if np.linalg.norm(r) <= tol * np.linalg.norm(y):
            break
    return x

def subspace_pursuit(Phi, y, k, iters=50):
    """subspace pursuit (Dai--Milenkovic 2009)"""
    S = top_k(Phi.T @ y, k)
    x = ls_on(Phi, y, S); r = y - Phi @ x
    for _ in range(iters):
        T = np.union1d(S, top_k(Phi.T @ r, k))
        b = ls_on(Phi, y, T)
        S_new = T[top_k(b[T], k)]
        x_new = ls_on(Phi, y, S_new); r_new = y - Phi @ x_new
        if np.linalg.norm(r_new) >= np.linalg.norm(r):
            break
        S, x, r = S_new, x_new, r_new
    return x

# ------------------------------------------------------------
# ขั้นตอนวิธีขีดเริ่มเปลี่ยน (หัวข้อ 5.4)
# ------------------------------------------------------------
def iht(Phi, y, k, alpha=1.0, iters=300):
    """IHT (Blumensath--Davies 2009): x <- H_k(x + alpha Phi^T (y - Phi x))"""
    x = np.zeros(Phi.shape[1])
    for _ in range(iters):
        v = x + alpha * (Phi.T @ (y - Phi @ x))
        x = np.zeros_like(v); S = top_k(v, k); x[S] = v[S]
    return x

def niht(Phi, y, k, iters=300, c=0.01, kappa=2.0, tol=1e-12, return_hist=False):
    """IHT แบบปรับขนาดก้าว (normalized IHT; Blumensath--Davies 2010)"""
    n = Phi.shape[1]; x = np.zeros(n); S = top_k(Phi.T @ y, k); hist = []
    for _ in range(iters):
        g = Phi.T @ (y - Phi @ x)
        gS = g[S]; den = np.linalg.norm(Phi[:, S] @ gS) ** 2
        alpha = (gS @ gS) / den if den > 0 else 1.0
        while True:
            v = x + alpha * g
            S_new = top_k(v, k); x_new = np.zeros(n); x_new[S_new] = v[S_new]
            if set(S_new) == set(S):
                break
            d = x_new - x; dd = np.linalg.norm(Phi @ d) ** 2
            omega = (1 - c) * (d @ d) / dd if dd > 0 else np.inf
            if alpha <= omega:
                break
            alpha = alpha / (kappa * (1 - c))
        step = np.linalg.norm(x_new - x)
        x, S = x_new, S_new
        hist.append(np.linalg.norm(y - Phi @ x))
        if step <= tol * max(np.linalg.norm(x), 1e-300):
            break
    return (x, np.array(hist)) if return_hist else x

def htp(Phi, y, k, iters=100):
    """hard thresholding pursuit (Foucart 2011): IHT แล้วแก้กำลังสองน้อยที่สุดบนตัวรองรับใหม่"""
    x = np.zeros(Phi.shape[1]); S_old = None
    for _ in range(iters):
        S = np.sort(top_k(x + Phi.T @ (y - Phi @ x), k))
        x = ls_on(Phi, y, S)
        if S_old is not None and np.array_equal(S, S_old):
            break
        S_old = S
    return x

def amp(Phi, y, iters=30, gamma=1.5, x_true=None):
    """approximate message passing (Donoho--Maleki--Montanari 2009) ด้วยขีดเริ่มเปลี่ยนแบบอ่อน
       ขีดเริ่มเปลี่ยน gamma * varsigma_t โดยประมาณ varsigma_t = ||z||_2 / sqrt(m)"""
    m, n = Phi.shape; x = np.zeros(n); z = y.copy(); mse = []; thr = []
    for _ in range(iters):
        varsigma = np.linalg.norm(z) / np.sqrt(m)
        x_new = soft(x + Phi.T @ z, gamma * varsigma)
        b = np.count_nonzero(x_new) / m
        z = y - Phi @ x_new + b * z
        x = x_new; thr.append(gamma * varsigma)
        if x_true is not None:
            mse.append(np.mean((x - x_true) ** 2))
    return x, np.array(mse), np.array(thr)

def fista(Phi, y, lam, iters, x_true=None):
    L = np.linalg.norm(Phi, 2) ** 2
    x = np.zeros(Phi.shape[1]); v = x.copy(); tau = 1.0; mse = []
    for _ in range(iters):
        x_new = soft(v - Phi.T @ (Phi @ v - y) / L, lam / L)
        tau_new = (1 + np.sqrt(1 + 4 * tau ** 2)) / 2
        v = x_new + ((tau - 1) / tau_new) * (x_new - x)
        x, tau = x_new, tau_new
        if x_true is not None:
            mse.append(np.mean((x - x_true) ** 2))
    return x, np.array(mse)

def ista(Phi, y, lam, iters, x_true=None):
    L = np.linalg.norm(Phi, 2) ** 2
    x = np.zeros(Phi.shape[1]); mse = []
    for _ in range(iters):
        x = soft(x - Phi.T @ (Phi @ x - y) / L, lam / L)
        if x_true is not None:
            mse.append(np.mean((x - x_true) ** 2))
    return x, np.array(mse)

def soft_homotopy(Phi, y, k, gamma, nmax=100, use_mpi=True, mpi_init=True, stop=2, pinv=None):
    """ตระกูลโฮโมโทปีแบบอ่อน (Tausiesakul & Asavaskulkiet, IEEE TIM 2023)
       x <- S_lam(x + B (y - Phi x)), lam <- gamma lam, ทำต่อขณะที่ ||x||_0 <= stop*k และไม่เกิน nmax รอบ
       SHA      : B = Phi^T,  lam_0 = ||Phi^T y||_inf, stop = 2
       SHA-MPI  : B = Phi^+,  lam_0 = ||Phi^T y||_inf, stop = 2      SMHA-MPI : เหมือนกันแต่ stop = 1
       STA-MPI  : B = Phi^+,  lam_0 = ||Phi^+ y||_inf, stop = 2      SMTA-MPI : เหมือนกันแต่ stop = 1"""
    m, n = Phi.shape
    if use_mpi:
        B = pinv if pinv is not None else Phi.T @ np.linalg.inv(Phi @ Phi.T)
    else:
        B = Phi.T
    lam = np.max(np.abs((B if (use_mpi and mpi_init) else Phi.T) @ y))
    x = np.zeros(n); i = 0
    with np.errstate(all="ignore"):
        while np.count_nonzero(x) <= stop * k and i <= nmax:              # เงื่อนไขตามบทความ: i <= N_max
            i += 1
            x = soft(x + B @ (y - Phi @ x), lam); lam *= gamma
            if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 1e12:      # ลู่ออก
                return np.full(n, np.inf)
    return x

SH_VARIANTS = {"SHA": dict(use_mpi=False, mpi_init=False, stop=2),
               "SHA-MPI": dict(use_mpi=True, mpi_init=False, stop=2),
               "SMHA-MPI": dict(use_mpi=True, mpi_init=False, stop=1),
               "STA-MPI": dict(use_mpi=True, mpi_init=True, stop=2),
               "SMTA-MPI": dict(use_mpi=True, mpi_init=True, stop=1)}

def iht_fnr(Phi, y, k, alpha, lam=0.0, p=0.9, init="zero", imax=100, tol=1e-6):
    """IHT ด้วยเลขชี้กำลังของนอร์ม (Tausiesakul & Asavaskulkiet, LNEE 2024)
       x <- H_k(x + alpha [Phi^T (y - Phi x) - lam p sign(x)|x|^{p-1}])   (บทความใช้ rho = 2 lam)
       lam = 0 คือ IHT ดั้งเดิม; init = 'lls' เริ่มจากผลเฉลยนอร์ม l2 น้อยที่สุด"""
    n = Phi.shape[1]
    x = np.zeros(n) if init == "zero" else Phi.T @ np.linalg.solve(Phi @ Phi.T, y)
    for _ in range(imax):
        g = Phi.T @ (y - Phi @ x)
        if lam > 0:
            ax = np.abs(x)
            g = g - np.where(ax > 0, lam * p * np.sign(x) * np.maximum(ax, 1e-300) ** (p - 1), 0.0)
        v = x + alpha * g
        x_new = np.zeros(n); S = top_k(v, k); x_new[S] = v[S]
        d = np.linalg.norm(x); e = np.linalg.norm(x_new - x) / d if d > 0 else np.inf
        x = x_new
        if e <= tol:
            break
    return x

def basis_pursuit(Phi, y):
    m, n = Phi.shape
    res = linprog(np.ones(2 * n), A_eq=np.hstack([Phi, -Phi]), b_eq=y, bounds=(0, None), method="highs")
    if res.status != 0:                       # ตัวแก้ไม่สำเร็จ: คืน NaN เพื่อให้นับเป็นความล้มเหลว ไม่ใช่ผลเฉลย
        return np.full(n, np.nan)
    return res.x[:n] - res.x[n:]

# ------------------------------------------------------------
# ความเบาบางเชิงแถวและการเรียนรู้พจนานุกรม (หัวข้อ 5.5–5.6)
# ------------------------------------------------------------
def somp(Phi, Y, k):
    """simultaneous OMP (Tropp--Gilbert--Strauss 2006): เลือก argmax_j ||phi_j^T R||_2"""
    S = []; R = Y.copy()
    for _ in range(k):
        j = int(np.argmax(np.linalg.norm(Phi.T @ R, axis=1)))
        S.append(j)
        Xs = np.linalg.lstsq(Phi[:, S], Y, rcond=None)[0]
        R = Y - Phi[:, S] @ Xs
    X = np.zeros((Phi.shape[1], Y.shape[1])); X[S] = Xs
    return X, np.array(S)

def omp_batch(D, Y, k):
    """เข้ารหัสเบาบางทีละสดมภ์ด้วย OMP (ใช้ในขั้นเข้ารหัสของ MOD และ K-SVD)"""
    return np.column_stack([omp(D, Y[:, j], k=k) for j in range(Y.shape[1])])

def normalize_cols(D):
    return D / np.maximum(np.linalg.norm(D, axis=0), 1e-12)

def dictionary_learning(Y, d, k, iters, method, rng, D_true=None):
    """MOD (Engan และคณะ 1999) หรือ K-SVD (Aharon--Elad--Bruckstein 2006)
       คืนพจนานุกรมและสัดส่วนอะตอมที่กู้คืนได้ในแต่ละรอบ (|<psi_true, psi>| > 0.99)"""
    n, J = Y.shape
    D = normalize_cols(Y[:, rng.choice(J, d, replace=False)].copy())
    rate = []
    for _ in range(iters):
        S = omp_batch(D, Y, k)
        if method == "MOD":
            D = normalize_cols(Y @ np.linalg.pinv(S))
        else:
            for a in range(d):
                users = np.flatnonzero(S[a])
                if len(users) == 0:
                    D[:, a] = normalize_cols(Y[:, [rng.integers(J)]])[:, 0]; continue
                E = Y[:, users] - D @ S[:, users] + np.outer(D[:, a], S[a, users])
                U, sv, Vt = np.linalg.svd(E, full_matrices=False)
                D[:, a] = U[:, 0]; S[a, users] = sv[0] * Vt[0]
        if D_true is not None:
            G = np.abs(D_true.T @ D)
            rate.append(np.mean(G.max(axis=1) > 0.99))
    return D, np.array(rate)

# ============================================================
# fig05-omp-trace : (ก) สหสัมพันธ์ |Phi^T r| รอบแรก (ข) ||r||_2 ของ MP เทียบ OMP
# ============================================================
if want("omp-trace"):
    rng = np.random.default_rng(21)
    n, m, k = 128, 40, 5
    Phi = gaussian(m, n, rng); x, S = sparse_signal(n, k, rng, kind="sign")
    x[S] *= np.array([1.0, 0.8, 0.6, 0.45, 0.3])[np.argsort(rng.random(k))]
    y = Phi @ x
    xo, ho, picks = omp(Phi, y, k=k, return_trace=True)
    xm, hm = mp(Phi, y, 200)
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.6))
    c0 = np.abs(Phi.T @ y)
    axes[0].stem(np.arange(n), c0, linefmt=GRY, markerfmt=" ", basefmt=" ")
    axes[0].plot(S, c0[S], "o", color=RED, ms=4, label="true support")
    axes[0].plot(picks[0], c0[picks[0]], "s", mfc="none", mec=ACC, ms=7, mew=1.2, label="first pick")
    axes[0].set_xlabel("$j$"); axes[0].set_ylabel(r"$|\boldsymbol{\phi}_j^{\mathsf{T}}\mathbf{y}|$")
    axes[0].set_title("(a)", fontsize=9, loc="left"); axes[0].legend(frameon=False, fontsize=7.5)
    axes[1].semilogy(np.arange(len(hm)), hm / hm[0], color=GRY, label="MP")
    axes[1].semilogy(np.arange(len(ho)), np.maximum(ho / ho[0], 1e-16), "o-", color=ACC, ms=3.5, label="OMP")
    axes[1].set_xlabel("$t$"); axes[1].set_ylabel(r"$\|\mathbf{r}^{(t)}\|_2/\|\mathbf{y}\|_2$")
    axes[1].set_title("(b)", fontsize=9, loc="left"); axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_ylim(1e-16, 2)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig05-omp-trace")
    mp_hits = lambda tol: int(np.argmax(hm / hm[0] <= tol)) if np.any(hm / hm[0] <= tol) else None
    print(f"omp-trace: n,m,k={n},{m},{k}; support {list(S)}; OMP picks {picks}; all correct: {set(picks)==set(S)}; "
          f"OMP rel err {rel_err(xo, x):.1e}")
    print(f"   MP iterations to ||r||/||y|| <= 1e-2: {mp_hits(1e-2)}, <= 1e-4: {mp_hits(1e-4)}, after 200: {hm[-1]/hm[0]:.1e}; "
          f"distinct atoms used by MP: {np.count_nonzero(np.abs(xm) > 1e-12)}")
    print(f"   rank of first-pick correlation among all: true support correlations {np.round(c0[S],3)}, max off-support {c0[np.setdiff1d(np.arange(n),S)].max():.3f}")
    mu = np.max(np.abs(np.triu((normalize_cols(Phi).T @ normalize_cols(Phi)), 1)))
    print(f"   coherence mu = {mu:.3f}; coherence condition k < (1+1/mu)/2 = {(1+1/mu)/2:.2f}")

# ============================================================
# fig05-recovery : อัตราความสำเร็จเทียบกับ k (ก) สมาชิกแบบเกาส์ (ข) สมาชิก +-1
# ============================================================
if want("cosamp-stop"):
    # ตัวอย่างประกอบขั้นตอนวิธีที่ 5.2: ตัวรองรับเท่าเดิมสองรอบติดกัน แต่ค่าบนตัวรองรับยังปรับต่อได้
    # จึงไม่ควรใช้ "ตัวรองรับไม่เปลี่ยน" เป็นเกณฑ์หยุดเพียงอย่างเดียว (ฟังก์ชัน cosamp() ใช้การเปลี่ยนของตัววนซ้ำและส่วนเหลือ)
    m, n, k = 12, 20, 3; rng = np.random.default_rng(85)
    Phi = gaussian(m, n, rng); x = np.zeros(n); S = rng.choice(n, k, replace=False); x[S] = rng.standard_normal(k)
    y = Phi @ x; xt = np.zeros(n); r = y.copy(); print("cosamp-stop: m, n, k = 12, 20, 3, seed 85")
    for t in range(1, 6):
        T = np.union1d(top_k(Phi.T @ r, 2 * k), np.flatnonzero(xt)); b = ls_on(Phi, y, T)
        Sn = top_k(b, k); xn = np.zeros(n); xn[Sn] = b[Sn]
        print(f"   iteration {t}: support {sorted(Sn.tolist())}  relative error {rel_err(xn, x):.3e}")
        xt = xn; r = y - Phi @ xt
    print(f"   cosamp() with iterate/residual stopping: relative error {rel_err(cosamp(Phi, y, k), x):.3e}")

if want("recovery"):
    n, m, trials = 256, 96, 60
    ks = list(range(4, 53, 4))
    algs = {"BP": lambda P, y, k: basis_pursuit(P, y),
            "OMP": lambda P, y, k: omp(P, y, k=k),
            "CoSaMP": lambda P, y, k: cosamp(P, y, k),
            "SP": lambda P, y, k: subspace_pursuit(P, y, k),
            "NIHT": lambda P, y, k: niht(P, y, k),
            "HTP": lambda P, y, k: htp(P, y, k)}
    styles = {"BP": ("o-", INK), "OMP": ("s-", ACC), "CoSaMP": ("^--", RED),
              "SP": ("v--", ORG), "NIHT": ("d-.", GRN), "HTP": ("x:", GRY)}
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.7), sharey=True)
    for ax, kind, lab in zip(axes, ["gauss", "sign"], ["(a)", "(b)"]):
        rates = {a: [] for a in algs}
        for k in ks:
            hits = {a: 0 for a in algs}
            rng = np.random.default_rng(1000 + k)
            for _ in range(trials):
                Phi = gaussian(m, n, rng); x, S = sparse_signal(n, k, rng, kind=kind); y = Phi @ x
                for a, fn in algs.items():
                    hits[a] += rel_err(fn(Phi, y, k), x) < 1e-4
            for a in algs:
                rates[a].append(hits[a] / trials)
        for a in algs:
            ax.plot(ks, rates[a], styles[a][0], color=styles[a][1], ms=3, lw=1.1, label=a)
        ax.set_xlabel("$k$"); ax.set_title(lab, fontsize=9, loc="left")
        k50 = {}
        for a in algs:
            r = np.array(rates[a]); idx = np.flatnonzero(r < 0.5)
            if len(idx) and idx[0] > 0:
                i = idx[0]; k50[a] = ks[i - 1] + (ks[i] - ks[i - 1]) * (r[i - 1] - 0.5) / (r[i - 1] - r[i])
            else:
                k50[a] = np.nan
        print(f"recovery ({kind}) n,m={n},{m}, {trials} trials: k at 50% success: " + "  ".join(f"{a} {v:.1f}" for a, v in k50.items()))
        for i, k in enumerate(ks):
            print(f"   k={k:2d}: " + "  ".join(f"{a} {rates[a][i]:.2f}" for a in algs))
    axes[0].set_ylabel("success rate"); axes[1].legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig05-recovery")

# ============================================================
# fig05-noisy : ความคลาดเคลื่อนสัมพัทธ์เทียบกับ SNR
# ============================================================
if want("noisy"):
    n, m, k, trials = 256, 96, 12, 100
    snrs = [5, 10, 15, 20, 25, 30, 35, 40]
    names = ["Oracle LS", "OMP", "CoSaMP", "SP", "NIHT", "HTP", "AMP", "LASSO", "LASSO+LS"]
    res = {a: [] for a in names}
    for snr in snrs:
        errs = {a: [] for a in names}
        rng = np.random.default_rng(2000 + snr)
        for _ in range(trials):
            Phi = gaussian(m, n, rng); x, S = sparse_signal(n, k, rng)
            s0 = Phi @ x; sigma = np.sqrt(np.mean(s0 ** 2) * 10 ** (-snr / 10))
            y = s0 + sigma * rng.standard_normal(m)
            lam = sigma * np.sqrt(2 * np.log(n))
            est = {"Oracle LS": ls_on(Phi, y, S), "OMP": omp(Phi, y, k=k), "CoSaMP": cosamp(Phi, y, k),
                   "SP": subspace_pursuit(Phi, y, k), "NIHT": niht(Phi, y, k), "HTP": htp(Phi, y, k),
                   "AMP": amp(Phi, y, iters=50, gamma=1.5)[0], "LASSO": fista(Phi, y, lam, 500)[0]}
            xl = est["LASSO"]; Sl = np.flatnonzero(np.abs(xl) > 1e-6 * max(np.abs(xl).max(), 1e-300))
            est["LASSO+LS"] = ls_on(Phi, y, Sl) if len(Sl) <= m else xl   # กำลังสองน้อยที่สุดซ้ำบนตัวรองรับของ LASSO
            for a in names:
                errs[a].append(rel_err(est[a], x) ** 2)
        for a in names:
            res[a].append(np.sqrt(np.mean(errs[a])))
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    st = {"Oracle LS": ("-", GRY), "OMP": ("s-", ACC), "CoSaMP": ("^--", RED), "SP": ("v--", ORG),
          "NIHT": ("d-.", GRN), "HTP": ("x:", INK), "AMP": ("o-", "#7C3AED"), "LASSO": ("+-", "#0E7490"),
          "LASSO+LS": ("*--", "#0E7490")}
    for a in names:
        ax.semilogy(snrs, res[a], st[a][0], color=st[a][1], ms=3.2, lw=1.1, label=a)
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("RMSRE"); ax.legend(frameon=False, fontsize=7, ncol=2)
    save(fig, "fig05-noisy")
    print(f"noisy: n,m,k={n},{m},{k}, {trials} trials, LASSO lambda = sigma sqrt(2 log n), AMP gamma=1.5")
    for i, snr in enumerate(snrs):
        print(f"   SNR {snr:2d}: " + "  ".join(f"{a} {res[a][i]:.3f}" for a in names))

# ============================================================
# fig05-amp : (ก) AMP เทียบ ISTA/FISTA และวิวัฒน์สถานะ (ข) AMP บนเมทริกซ์เลขสภาวะสูง
# ============================================================
if want("amp"):
    rng = np.random.default_rng(31)
    n, m, k, gamma = 2000, 1000, 100, 1.5
    nu = m / n
    Phi = gaussian(m, n, rng); x, S = sparse_signal(n, k, rng)
    sigma = 0.01; y = Phi @ x + sigma * rng.standard_normal(m)
    T = 40
    xa, mse_amp, thr = amp(Phi, y, iters=T, gamma=gamma, x_true=x)
    # วิวัฒน์สถานะ: varsigma_{t+1}^2 = sigma^2 + (1/nu) E[(soft(X + varsigma_t Z; gamma varsigma_t) - X)^2]
    rngZ = np.random.default_rng(32); Z = rngZ.standard_normal((20, n))
    vs2 = [sigma ** 2 + np.mean(x ** 2) / nu]; mse_se = []
    for t in range(T):
        vs = np.sqrt(vs2[-1])
        e = np.mean((soft(x + vs * Z, gamma * vs) - x) ** 2)
        mse_se.append(e); vs2.append(sigma ** 2 + e / nu)
    # LASSO ที่สมนัยกับจุดตรึงของ AMP: lambda = theta (1 - ||x||_0 / m)
    lam = thr[-1] * (1 - np.count_nonzero(xa) / m)
    _, mse_ista = ista(Phi, y, lam, 3000, x_true=x)
    xf, mse_fista = fista(Phi, y, lam, 3000, x_true=x)
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.7))
    ax = axes[0]
    ax.semilogy(np.arange(1, T + 1), mse_amp, "o", color=ACC, ms=3, label="AMP")
    ax.semilogy(np.arange(1, T + 1), mse_se, "-", color=INK, lw=1.0, label="state evolution")
    ax.semilogy(np.arange(1, 3001), mse_fista, "--", color=RED, lw=1.0, label="FISTA")
    ax.semilogy(np.arange(1, 3001), mse_ista, ":", color=GRY, lw=1.2, label="ISTA")
    ax.set_xscale("log"); ax.set_xlabel("$t$"); ax.set_ylabel("MSE"); ax.set_title("(a)", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=7)
    print(f"amp: n,m,k={n},{m},{k}, sigma={sigma}, gamma={gamma}; calibrated LASSO lambda={lam:.4f}")
    for t in [1, 2, 5, 10, 20, 40]:
        print(f"   t={t:2d}: AMP {mse_amp[t-1]:.3e}  SE {mse_se[t-1]:.3e}")
    def first_within(h, target, f=1.05):
        idx = np.flatnonzero(h <= f * target); return int(idx[0]) + 1 if len(idx) else None
    tgt = mse_amp[-1]
    print(f"   iterations to within 5% of AMP final MSE {tgt:.3e}: AMP {first_within(mse_amp, tgt)}, FISTA {first_within(mse_fista, tgt)}, ISTA {first_within(mse_ista, tgt)}")
    print(f"   ||x_AMP - x_FISTA(3000)|| / ||x_FISTA|| = {np.linalg.norm(xa - xf)/np.linalg.norm(xf):.2e}")
    # (ข) เมทริกซ์ที่มีเลขสภาวะสูง
    U, _, Vt = np.linalg.svd(gaussian(m, n, np.random.default_rng(33)), full_matrices=False)
    res_b = {}
    for cond, col in [(1, ACC), (10, ORG), (100, RED)]:
        sv = np.geomspace(1, 1 / cond, m)
        P = normalize_cols(U @ np.diag(sv) @ Vt)
        yb = P @ x + sigma * rng.standard_normal(m)
        _, mse_b, _ = amp(P, yb, iters=T, gamma=gamma, x_true=x)
        mse_b = np.where(np.isfinite(mse_b), mse_b, 1e6)
        axes[1].semilogy(np.arange(1, T + 1), np.minimum(mse_b, 1e6), color=col, label=rf"cond $={cond}$")
        _, mf = fista(P, yb, lam, 3000, x_true=x)
        res_b[cond] = (mse_b[-1], mf[-1])
    axes[1].set_xlabel("$t$"); axes[1].set_ylabel("MSE (AMP)"); axes[1].set_title("(b)", fontsize=9, loc="left")
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig05-amp")
    for c, (a, f) in res_b.items():
        print(f"   cond {c:3d}: AMP MSE after {T} it {a:.3e}   FISTA after 3000 it {f:.3e}")

# ============================================================
# fig05-mmv : อัตราความสำเร็จของ SOMP เทียบกับ k สำหรับจำนวนเวกเตอร์การวัด C
# ============================================================
if want("mmv"):
    n, m, trials = 128, 32, 100
    ks = list(range(2, 31, 2)); Cs = [1, 2, 4, 8, 16]
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    cols = [GRY, ORG, RED, ACC, INK]
    for C, col in zip(Cs, cols):
        rates = []
        for k in ks:
            rng = np.random.default_rng(3000 + 31 * k + C); hit = 0
            for _ in range(trials):
                Phi = gaussian(m, n, rng); S = rng.choice(n, k, replace=False)
                X = np.zeros((n, C)); X[S] = rng.standard_normal((k, C)); Y = Phi @ X
                _, Sh = somp(Phi, Y, k); hit += set(Sh) == set(S)
            rates.append(hit / trials)
        ax.plot(ks, rates, "o-", color=col, ms=3, lw=1.1, label=f"$J={C}$")
        r = np.array(rates); idx = np.flatnonzero(r < 0.5)
        k50 = (ks[idx[0]-1] + (ks[idx[0]] - ks[idx[0]-1]) * (r[idx[0]-1] - 0.5) / (r[idx[0]-1] - r[idx[0]])) if len(idx) and idx[0] > 0 else np.nan
        print(f"mmv n,m={n},{m}: C={C:2d} k at 50% support recovery = {k50:.1f}; rates " + " ".join(f"{v:.2f}" for v in rates))
    ax.set_xlabel("$k$"); ax.set_ylabel("support recovery rate"); ax.legend(frameon=False, fontsize=7.5)
    save(fig, "fig05-mmv")

# ============================================================
# fig05-dictionary : สัดส่วนอะตอมที่กู้คืนได้ของ K-SVD เทียบ MOD
# ============================================================
if want("dictionary"):
    n, d, J, k, iters, runs = 20, 50, 1500, 3, 40, 3
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    finals = {"K-SVD": [], "MOD": []}; curves = {"K-SVD": [], "MOD": []}
    for run in range(runs):
        rng = np.random.default_rng(4000 + run)
        Dt = normalize_cols(rng.standard_normal((n, d)))
        Sx = np.zeros((d, J))
        for j in range(J):
            Sx[rng.choice(d, k, replace=False), j] = rng.standard_normal(k)
        X0 = Dt @ Sx
        noise = rng.standard_normal(X0.shape); noise *= np.linalg.norm(X0) / np.linalg.norm(noise) * 10 ** (-20 / 20)
        Y = X0 + noise
        for meth in ["K-SVD", "MOD"]:
            t0 = time.time()
            _, rate = dictionary_learning(Y, d, k, iters, meth, np.random.default_rng(5000 + run), D_true=Dt)
            curves[meth].append(rate); finals[meth].append(rate[-1])
            print(f"dictionary run {run}: {meth} final recovered {rate[-1]*100:.0f}%  (time {time.time()-t0:.1f} s); at it 10: {rate[9]*100:.0f}%, it 20: {rate[19]*100:.0f}%")
    for meth, col, ls in [("K-SVD", ACC, "-"), ("MOD", RED, "--")]:
        c = np.mean(curves[meth], axis=0) * 100
        ax.plot(np.arange(1, iters + 1), c, ls, color=col, label=meth)
    ax.set_xlabel("iteration"); ax.set_ylabel("recovered atoms (%)"); ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(0, 100)
    save(fig, "fig05-dictionary")
    print(f"dictionary n,d,J,k={n},{d},{J},{k}, SNR 20 dB, {runs} runs: mean final K-SVD {np.mean(finals['K-SVD'])*100:.1f}%  MOD {np.mean(finals['MOD'])*100:.1f}%")

# ============================================================
# fig05-author : การทำซ้ำอิสระของขั้นตอนวิธีในบทความของผู้แต่ง
#   ภาพ: RMSRE เทียบกับตัวประกอบการลดขีดเริ่มเปลี่ยน (ก) สดมภ์นอร์มหนึ่ง (ข) สดมภ์ที่มีพลังงาน 0.15
#   ตาราง/ตัวเลข: จำนวนรอบสูงสุด, ความไม่แปรเปลี่ยนต่อการแปลงของระบบ, กฎความเอนเอียง, กรณีมีสัญญาณรบกวน,
#                 และ IHT ด้วยเลขชี้กำลังของนอร์ม
# ============================================================
if want("author"):
    m, n, k = 128, 256, 6
    def draw(rng, kk=k, mm=m, nn=n):
        Phi = gaussian(mm, nn, rng)                                   # พลังงานคาดหมายของสดมภ์เป็น 1 (ไม่ normalize)
        x = np.zeros(nn); S = rng.choice(nn, kk, replace=False); x[S] = rng.standard_normal(kk) / np.sqrt(kk)
        return Phi, x
    def rmsre(e):
        e = np.array(e, dtype=float); e = np.where(np.isfinite(e), e, 1e6); return float(np.sqrt(np.mean(e ** 2)))
    def run_all(Phi, y, x, gam, nmax=100, names=None):
        P = Phi.T @ np.linalg.inv(Phi @ Phi.T)
        return {a: rel_err(soft_homotopy(Phi, y, k, gam[a] if isinstance(gam, dict) else gam, nmax=nmax, pinv=P, **kw), x)
                for a, kw in SH_VARIANTS.items() if names is None or a in names}
    show = ["SHA", "SHA-MPI", "STA-MPI", "SMTA-MPI"]
    st = {"SHA": ("o-", GRY), "SHA-MPI": ("s--", ORG), "SMHA-MPI": ("v:", GRN), "STA-MPI": ("^-", ACC), "SMTA-MPI": ("d-.", RED)}
    gammas = [0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.93, 0.95, 0.98]
    trials = 1000
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.75), sharey=True)
    for ax, s2, lab in [(axes[0], 1.0, "(a)"), (axes[1], 0.15, "(b)")]:
        res = {a: [] for a in SH_VARIANTS}; fail = {a: [] for a in SH_VARIANTS}
        for g in gammas:
            errs = {a: [] for a in SH_VARIANTS}; r2 = np.random.default_rng(7)      # เมล็ดเดียวกันทุกค่า: ระบบชุดเดียวกัน
            for _ in range(trials):
                Phi, x = draw(r2); Phi = np.sqrt(s2) * Phi; y = Phi @ x
                for a, e in run_all(Phi, y, x, g).items():
                    errs[a].append(e)
            for a in SH_VARIANTS:
                res[a].append(rmsre(errs[a])); fail[a].append(float(np.mean(np.array(errs[a]) > 0.1)))
        for a in show:
            ax.semilogy(gammas, np.minimum(res[a], 9.9), st[a][0], color=st[a][1], ms=3, lw=1.1, label=a)
        ax.set_xlabel(r"$\gamma_{\mathrm{h}}$"); ax.set_title(lab, fontsize=9, loc="left"); ax.set_ylim(1e-8, 5)
        print(f"author: soft homotopy family, RMSRE vs gamma_h (noiseless, m,n,k={m},{n},{k}, T=100, {trials} trials, column energy s^2={s2})")
        for i, g in enumerate(gammas):
            print(f"   gamma_h={g:.2f}: " + "  ".join(f"{a} {res[a][i]:.1e} [{fail[a][i]:.3f}]" for a in SH_VARIANTS))
        if s2 == 1.0:
            sta = dict(zip(gammas, res["STA-MPI"]))
            print(f"   bias law: RMSRE(0.95)/RMSRE(0.93) = {sta[0.95]/sta[0.93]:.2f} vs (0.95/0.93)^100 = {(0.95/0.93)**100:.2f};"
                  f"  RMSRE(0.93)/RMSRE(0.88) = {sta[0.93]/sta[0.88]:.0f} vs (0.93/0.88)^100 = {(0.93/0.88)**100:.0f};"
                  f"  RMSRE/gamma^T at 0.88, 0.93, 0.95 = {sta[0.88]/0.88**100:.2f}, {sta[0.93]/0.93**100:.2f}, {sta[0.95]/0.95**100:.2f}")
            sm = dict(zip(gammas, res["SHA"]))
            print(f"   SHA: RMSRE/gamma^T at 0.88, 0.93, 0.95 = {sm[0.88]/0.88**100:.2f}, {sm[0.93]/0.93**100:.2f}, {sm[0.95]/0.95**100:.2f}")
    axes[0].set_ylabel("RMSRE"); axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig05-author")
    # --- ความล้มเหลวที่เกิดน้อยครั้งทางซ้ายของค่าต่ำสุด (STA-MPI, สดมภ์นอร์มหนึ่ง, สุ่ม 8,000 ครั้ง)
    print("author: rare premature stops of STA-MPI, expected unit column energy, 8000 trials (RMSRE, fraction with rel. err > 0.1)")
    for g in [0.80, 0.85, 0.88]:
        r2 = np.random.default_rng(21); e = []
        for _ in range(8000):
            Phi, x = draw(r2); y = Phi @ x
            e.append(rel_err(soft_homotopy(Phi, y, k, g, **SH_VARIANTS["STA-MPI"]), x))
        e = np.array(e)
        print(f"   gamma_h={g:.2f}: RMSRE {rmsre(e):.1e}, fraction {np.mean(e > 0.1):.5f}, median {np.median(e):.1e}")
    # --- จำนวนรอบสูงสุด (สดมภ์นอร์มหนึ่ง ค่า gamma_h ตามที่บทความรายงานว่าเหมาะที่สุดสำหรับ K = 6)
    G = {"SHA": 0.95, "SHA-MPI": 0.90, "SMHA-MPI": 0.91, "STA-MPI": 0.88, "SMTA-MPI": 0.93}
    print("author: RMSRE vs max iterations T, expected unit column energy (gamma_h per paper: " + ", ".join(f"{a} {v}" for a, v in G.items()) + "), 300 trials")
    for N in [20, 40, 70, 100, 200]:
        errs = {a: [] for a in SH_VARIANTS}; r2 = np.random.default_rng(8)
        for _ in range(300):
            Phi, x = draw(r2); y = Phi @ x
            for a, e in run_all(Phi, y, x, G, nmax=N).items():
                errs[a].append(e)
        print(f"   T={N:3d}: " + "  ".join(f"{a} {rmsre(errs[a]):.1e}" for a in SH_VARIANTS))
    # --- ความไม่แปรเปลี่ยนต่อการแปลงผกผันได้ทางซ้ายของระบบ (มาตราส่วนและเลขสภาวะ)
    print("author: invariance test at gamma_h = 0.88, T = 100, 1000 trials (same signals, four versions of the same linear system)")
    U0, _, _ = np.linalg.svd(np.random.default_rng(11).standard_normal((m, m)))
    Bcond = U0 @ np.diag(np.geomspace(1, 1e-2, m)) @ U0.T              # เมทริกซ์ผกผันได้ เลขสภาวะ 100
    for lab, B in [("expected unit column energy (s^2 = 1)", np.eye(m)), ("scaled, s^2 = 0.15", np.sqrt(0.15) * np.eye(m)),
                   ("scaled, s^2 = sqrt(m)", m ** 0.25 * np.eye(m)), ("left-multiplied, cond(B) = 100", Bcond)]:
        errs = {a: [] for a in ["SHA", "SHA-MPI", "STA-MPI"]}; r2 = np.random.default_rng(7); cn = []
        for it in range(1000):
            Phi, x = draw(r2); P2 = B @ Phi; y2 = P2 @ x
            if it < 50:
                cn.append(np.linalg.cond(P2))
            for a, e in run_all(P2, y2, x, 0.88, names=errs.keys()).items():
                errs[a].append(e)
        print(f"   {lab:32s} cond(Phi) ~ {np.median(cn):7.1f}: " + "   ".join(f"{a} {rmsre(v):.1e}" for a, v in errs.items()))
    # --- กรณีมีสัญญาณรบกวน: การขยายสัญญาณรบกวนของตัวผกผันเทียม
    print(f"author: noisy case, expected unit column energy, 300 trials; heuristic ratio 1/sqrt(1-m/n) = {1/np.sqrt(1-m/n):.3f};"
          f" sqrt(m^2/(n(n-m-1))) = {np.sqrt(m*m/(n*(n-m-1))):.3f}")
    for snr in [10, 20, 30, 40, 60]:
        errs = {"SHA@0.95": [], "STA@0.88": [], "SHA@0.90": [], "STA@0.90": []}; r2 = np.random.default_rng(12); amp_ = []
        for _ in range(300):
            Phi, x = draw(r2); s0 = Phi @ x; e = np.sqrt(np.mean(s0 ** 2) * 10 ** (-snr / 10)) * r2.standard_normal(m); y2 = s0 + e
            P = Phi.T @ np.linalg.inv(Phi @ Phi.T)
            errs["SHA@0.95"].append(rel_err(soft_homotopy(Phi, y2, k, 0.95, **SH_VARIANTS["SHA"]), x))
            errs["STA@0.88"].append(rel_err(soft_homotopy(Phi, y2, k, 0.88, pinv=P, **SH_VARIANTS["STA-MPI"]), x))
            errs["SHA@0.90"].append(rel_err(soft_homotopy(Phi, y2, k, 0.90, **SH_VARIANTS["SHA"]), x))
            errs["STA@0.90"].append(rel_err(soft_homotopy(Phi, y2, k, 0.90, pinv=P, **SH_VARIANTS["STA-MPI"]), x))
            amp_.append(np.linalg.norm(P @ e) / np.linalg.norm(Phi.T @ e))
        R = {a: rmsre(v) for a, v in errs.items()}
        print(f"   SNR {snr:2d} dB: SHA(0.95) {R['SHA@0.95']:.3e}  STA-MPI(0.88) {R['STA@0.88']:.3e}  ratio {R['STA@0.88']/R['SHA@0.95']:.2f} |"
              f" SHA(0.90) {R['SHA@0.90']:.3e}  STA-MPI(0.90) {R['STA@0.90']:.3e}  ratio {R['STA@0.90']/R['SHA@0.90']:.2f} |"
              f" ||Phi^+ e||/||Phi^T e|| ~ {np.median(amp_):.3f}")
    # --- IHT ด้วยเลขชี้กำลังของนอร์ม
    m2, n2, k2 = 64, 128, 16
    def iht_trials(cfgs, trials, imax=100, seed=9):
        errs = {c: [] for c in cfgs}; r2 = np.random.default_rng(seed)
        for _ in range(trials):
            Phi, x = draw(r2, k2, m2, n2); y = Phi @ x; al = 1 / np.linalg.norm(Phi, 2) ** 2
            for c, kw in cfgs.items():
                if kw is None:
                    xl = Phi.T @ np.linalg.solve(Phi @ Phi.T, y); h = np.zeros(n2); S = top_k(xl, k2); h[S] = xl[S]
                    errs[c].append(rel_err(h, x))
                else:
                    errs[c].append(rel_err(iht_fnr(Phi, y, k2, al, imax=imax, **kw), x))
        return {c: (rmsre(v), float(np.mean(np.array(v) < 1e-3))) for c, v in errs.items()}
    cfgs = {"H_k(LLS)": None, "IHT (zero)": dict(), "IHT (LLS)": dict(init="lls")}
    for p in [0.5, 0.9]:
        for lam in [5e-7, 5e-5, 5e-4, 5e-3, 2e-2, 5e-2]:
            cfgs[f"IHT-FNR p={p} lam={lam:.0e}"] = dict(init="lls", lam=lam, p=p)
    print(f"author: IHT with norm exponent (m,n,k={m2},{n2},{k2}, alpha = 1/sigma_1^2, 400 trials); value = RMSRE (fraction with rel. err < 1e-3)")
    for imax, tr, keys in [(100, 400, list(cfgs)), (1000, 400, ["IHT (zero)", "IHT (LLS)", "IHT-FNR p=0.9 lam=5e-07", "IHT-FNR p=0.5 lam=5e-03", "IHT-FNR p=0.9 lam=2e-02"])]:
        out = iht_trials({c: cfgs[c] for c in keys}, tr, imax=imax)
        print(f"   {imax} iterations:")
        for c, (r, f_) in out.items():
            print(f"      {c:28s} {r:.4f} ({f_:.3f})")

# ============================================================
# fig05-runtime : เวลาคำนวณเทียบกับ n (m = n/4, k = n/32)
# ============================================================
if want("runtime"):
    rng = np.random.default_rng(13)
    ns = [256, 512, 1024, 2048, 4096]
    names = ["BP (LP)", "FISTA", "OMP", "CoSaMP", "NIHT", "AMP"]
    Tm = {a: [] for a in names}; Er = {a: [] for a in names}
    for n in ns:
        m, k = n // 4, n // 32
        Phi = gaussian(m, n, rng); x, S = sparse_signal(n, k, rng); y = Phi @ x
        fns = {"BP (LP)": lambda: basis_pursuit(Phi, y),
               "FISTA": lambda: fista(Phi, y, 1e-4 * np.max(np.abs(Phi.T @ y)), 500)[0],
               "OMP": lambda: omp(Phi, y, k=k), "CoSaMP": lambda: cosamp(Phi, y, k),
               "NIHT": lambda: niht(Phi, y, k), "AMP": lambda: amp(Phi, y, iters=30, gamma=1.5)[0]}
        for a in names:
            if a == "BP (LP)" and n > 2048:
                Tm[a].append(np.nan); Er[a].append(np.nan); continue
            t0 = time.time(); xh = fns[a](); Tm[a].append(time.time() - t0); Er[a].append(rel_err(xh, x))
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    st = {"BP (LP)": ("o-", INK), "FISTA": ("s-", GRY), "OMP": ("^-", ACC), "CoSaMP": ("v--", RED),
          "NIHT": ("d-.", GRN), "AMP": ("x:", "#7C3AED")}
    for a in names:
        ax.loglog(ns, Tm[a], st[a][0], color=st[a][1], ms=3.5, base=2, label=a)
    ax.set_xlabel("$n$"); ax.set_ylabel("time (s)"); ax.legend(frameon=False, fontsize=7, ncol=2)
    save(fig, "fig05-runtime")
    print("runtime (s) for n =", ns, "(m = n/4, k = n/32)")
    for a in names:
        t = np.array(Tm[a]); ok = np.isfinite(t)
        sl = np.polyfit(np.log2(np.array(ns)[ok]), np.log2(t[ok]), 1)[0]
        print(f"   {a:8s}", " ".join(f"{v:8.4f}" for v in t), f" slope(log2)={sl:.2f}  rel.err:", " ".join(f"{e:.0e}" for e in Er[a]))
