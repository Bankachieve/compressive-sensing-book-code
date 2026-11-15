# -*- coding: utf-8 -*-
"""สร้างรูปประกอบและผลการทดลองเชิงตัวเลขทั้งหมดของบทที่ 4 -> figures/fig04-*.pdf
ขั้นตอนวิธีทุกตัว (ISTA, FISTA, ADMM, IRLS, FOCUSS, การทำให้เป็นปรกติด้วยนอร์มเศษส่วน)
เขียนขึ้นเองด้วย NumPy/SciPy ตามสมการในบท เพื่อให้ผู้อ่านตรวจสอบและทำซ้ำได้
ค่าตัวเลขทุกค่าที่อ้างในคำบรรยายภาพและเนื้อหาพิมพ์ออกทางจอเมื่อรันสคริปต์นี้"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle  # กริดพื้นหลังและตำแหน่ง legend แบบเดียวกันทั้งเล่ม
from scipy.optimize import linprog
import os, time

import sys
RUN = set(sys.argv[1:])   # เช่น `python fig04_generate.py fnr focuss-mse` เพื่อสร้างเฉพาะบางรูป (ไม่ระบุ = ทั้งหมด)
def want(name):
    return (not RUN) or (name in RUN)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT, exist_ok=True)
INK, ACC, GRY, FILL = "#1F2A37", "#1D4ED8", "#9AA3AF", "#DBEAFE"
RED = "#B91C1C"
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
    return rng.standard_normal((m, n)) / np.sqrt(m)

def sparse_signal(n, k, rng, kind="gauss"):
    x = np.zeros(n)
    S = rng.choice(n, k, replace=False)
    x[S] = rng.standard_normal(k) if kind == "gauss" else rng.choice([-1.0, 1.0], k)
    return x, np.sort(S)

def rel_err(xh, x):
    return np.linalg.norm(xh - x) / np.linalg.norm(x)

# ------------------------------------------------------------
# ตัวดำเนินการและขั้นตอนวิธีตามสมการในบทที่ 4
# ------------------------------------------------------------
def soft(z, lam):
    """ตัวดำเนินการขีดเริ่มเปลี่ยนแบบอ่อน S_lambda = prox ของ lambda ||.||_1"""
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0.0)

def basis_pursuit(Phi, y):
    """(P1) ผ่านกำหนดการเชิงเส้น z = u - v, u,v >= 0 (ตัวแก้ HiGHS)"""
    m, n = Phi.shape
    res = linprog(np.ones(2 * n), A_eq=np.hstack([Phi, -Phi]), b_eq=y,
                  bounds=(0, None), method="highs")
    return None if res.status != 0 else res.x[:n] - res.x[n:]

def basis_pursuit_dual(Phi, y):
    """ตัวแปรคู่กันของ (P1): max y^T u s.t. ||Phi^T u||_inf <= 1"""
    m, n = Phi.shape
    # min -y^T u  s.t.  Phi^T u <= 1, -Phi^T u <= 1
    res = linprog(-y, A_ub=np.vstack([Phi.T, -Phi.T]), b_ub=np.ones(2 * n),
                  bounds=(None, None), method="highs")
    return None if res.status != 0 else res.x

def lasso_obj(Phi, y, x, lam):
    return 0.5 * np.sum((Phi @ x - y) ** 2) + lam * np.sum(np.abs(x))

def ista(Phi, y, lam, iters, x0=None, L=None):
    m, n = Phi.shape
    L = L or np.linalg.norm(Phi, 2) ** 2
    x = np.zeros(n) if x0 is None else x0.copy()
    hist = []
    PhiT = Phi.T
    for _ in range(iters):
        x = soft(x - (1.0 / L) * (PhiT @ (Phi @ x - y)), lam / L)
        hist.append(lasso_obj(Phi, y, x, lam))
    return x, np.array(hist)

def fista(Phi, y, lam, iters, L=None):
    m, n = Phi.shape
    L = L or np.linalg.norm(Phi, 2) ** 2
    x = np.zeros(n); w = x.copy(); t = 1.0
    hist = []
    PhiT = Phi.T
    for _ in range(iters):
        x_new = soft(w - (1.0 / L) * (PhiT @ (Phi @ w - y)), lam / L)
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        w = x_new + ((t - 1) / t_new) * (x_new - x)
        x, t = x_new, t_new
        hist.append(lasso_obj(Phi, y, x, lam))
    return x, np.array(hist)

def admm_lasso(Phi, y, lam, beta, iters):
    """ADMM สำหรับ LASSO: แยก x และ z โดยข้อจำกัด x = z (Boyd และคณะ 2011)"""
    m, n = Phi.shape
    # เมทริกซ์ผกผันของ (Phi^T Phi + beta I) ผ่านบทตั้งการผกผันเมทริกซ์เมื่อ m < n
    PhiT = Phi.T
    Minv = np.linalg.inv(np.eye(m) + Phi @ PhiT / beta)
    def solve(rhs):  # (Phi^T Phi + beta I)^{-1} rhs
        return rhs / beta - PhiT @ (Minv @ (Phi @ rhs)) / beta ** 2
    Phity = PhiT @ y
    x = np.zeros(n); z = np.zeros(n); u = np.zeros(n)
    hist = []
    for _ in range(iters):
        x = solve(Phity + beta * (z - u))
        z = soft(x + u, lam / beta)
        u = u + x - z
        hist.append(lasso_obj(Phi, y, z, lam))
    return z, np.array(hist)

def irls(Phi, y, p, k, iters=300, tol=1e-8):
    """IRLS แบบ Daubechies--DeVore--Fornasier--G\u00fcnt\u00fcrk (2010) สำหรับ min ||x||_p^p s.t. Phi x = y
    น้ำหนัก w_i = (x_i^2 + eps^2)^{p/2-1}; ขั้นกำลังสองน้อยที่สุดถ่วงน้ำหนักมีรูปแบบปิด
    x = W^{-1} Phi^T (Phi W^{-1} Phi^T)^{-1} y  และตารางลด eps_{t+1} = min(eps_t, r(x)_{k+1}/n)"""
    m, n = Phi.shape
    x = Phi.T @ np.linalg.solve(Phi @ Phi.T, y)   # เริ่มจากผลเฉลยนอร์มน้อยที่สุด
    eps = 1.0
    for _ in range(iters):
        w_inv = (x ** 2 + eps ** 2) ** (1 - p / 2)   # = 1/w_i
        G = (Phi * w_inv) @ Phi.T
        x_new = w_inv * (Phi.T @ np.linalg.solve(G, y))
        r = np.sort(np.abs(x_new))[::-1]
        eps = min(eps, r[k] / n)
        d = np.linalg.norm(x_new - x) / max(1.0, np.linalg.norm(x))
        x = x_new
        if d < tol and eps < 1e-9:
            break
    return x

def irls_cy(Phi, y, p, iters=300):
    """IRLS แบบ Chartrand--Yin (2008) สำหรับ p < 1: น้ำหนัก (x_i^2 + eps)^{p/2-1}
    เริ่ม eps = 1 และลด eps ลงสิบเท่าทุกครั้งที่การทำซ้ำลู่เข้า ณ ค่า eps ปัจจุบัน (การต่อเนื่อง)"""
    m, n = Phi.shape
    x = Phi.T @ np.linalg.solve(Phi @ Phi.T, y)
    eps = 1.0
    for _ in range(iters):
        w_inv = (x ** 2 + eps) ** (1 - p / 2)
        G = (Phi * w_inv) @ Phi.T
        x_new = w_inv * (Phi.T @ np.linalg.solve(G, y))
        if np.linalg.norm(x_new - x) < np.sqrt(eps) / 100:
            eps /= 10
        x = x_new
        if eps < 1e-9:
            break
    return x

def focuss(Phi, y, p, iters=40, lam=0.0, tol=1e-8):
    """FOCUSS (Gorodnitsky--Rao 1997) / FOCUSS แบบทำให้เป็นปรกติ (Rao และคณะ 2003)
    x_{t+1} = W_t Phi_t^+ y, W_t = diag(|x_t|^{1-p/2}), Phi_t = Phi W_t
    รูปแบบทำให้เป็นปรกติ: x_{t+1} = W_t^2 Phi^T (Phi W_t^2 Phi^T + lam I)^{-1} y"""
    m, n = Phi.shape
    if not np.any(y):                           # y = 0: ผลเฉลยคือเวกเตอร์ศูนย์ (น้ำหนักยุบเป็นศูนย์และ G เป็นเอกฐาน)
        return np.zeros(n)
    x = Phi.T @ np.linalg.solve(Phi @ Phi.T, y)
    for _ in range(iters):
        w2 = np.abs(x) ** (2 - p)
        G = (Phi * w2) @ Phi.T + lam * np.eye(m)
        try:
            x_new = w2 * (Phi.T @ np.linalg.solve(G, y))
        except np.linalg.LinAlgError:           # G เอกฐานเมื่อน้ำหนักเป็นศูนย์เกือบทั้งหมด: ใช้ผลเฉลยนอร์มน้อยที่สุด
            x_new = w2 * (Phi.T @ np.linalg.lstsq(G, y, rcond=None)[0])
        if np.linalg.norm(x_new - x) <= tol * max(1.0, np.linalg.norm(x)):
            x = x_new; break
        x = x_new
    return x

def _rel_change(x_new, x):
    d = np.linalg.norm(x)
    return np.linalg.norm(x_new - x) / d if d > 0 else np.inf

def irls_dr(Phi, y, p, xi=1e-6, nmax=100, tol=1e-6):
    """IRLS ด้วยการทำให้เป็นปรกติเชิงทแยง (Tausiesakul & Asavaskulkiet, JCSSE 2023 / JAIT 2023, ขั้นตอนวิธีที่ 2)
       x <- D Phi^T (Phi D Phi^T + xi I)^{-1} y,  D = diag(|x|^{2-p}),  x^(0) = 1
       xi = 0 คือ FOCUSS ดั้งเดิมที่เริ่มจากเวกเตอร์หนึ่ง"""
    m, n = Phi.shape
    x = np.ones(n)
    with np.errstate(all="ignore"):
        for _ in range(nmax):
            d = np.abs(x) ** (2 - p)
            G = (Phi * d) @ Phi.T + xi * np.eye(m)
            try:
                x_new = d * (Phi.T @ np.linalg.solve(G, y))
            except np.linalg.LinAlgError:
                x_new = d * (Phi.T @ (np.linalg.pinv(G) @ y))
            if not np.all(np.isfinite(x_new)):
                break
            e = _rel_change(x_new, x); x = x_new
            if e <= tol:
                break
    return x

def ip_fnr(Phi, y, p, lam=5e-7, eta=1e-6, nmax=100, tol=1e-6, form="mm"):
    """FNR แบบจุดตรึงพร้อมการก่อกวนตัวผกผัน (IP-FNR; MSSP 2023)
       ฟังก์ชันเป้าหมายของหนังสือ (1/2)||y - Phi x||^2 + lam ||x||_p^p  (บทความใช้ ||.||^2 + rho ||.||_p^p, rho = 2 lam)
       form='nn': รูปในบทความ  x <- (Phi^T Phi + lam p diag(1/(|x|^{2-p} + eta)))^{-1} Phi^T y   (ระบบ n x n)
       form='mm': รูปสมมูล     x <- D_eta Phi^T (Phi D_eta Phi^T + lam p I)^{-1} y, D_eta = diag(|x|^{2-p} + eta) (ระบบ m x m)"""
    m, n = Phi.shape
    x = np.ones(n)
    if form == "nn":
        PtP = Phi.T @ Phi; Pty = Phi.T @ y
    for _ in range(nmax):
        d = np.abs(x) ** (2 - p) + eta
        if form == "nn":
            x_new = np.linalg.solve(PtP + lam * p * np.diag(1.0 / d), Pty)
        else:
            x_new = d * (Phi.T @ np.linalg.solve((Phi * d) @ Phi.T + lam * p * np.eye(m), y))
        e = _rel_change(x_new, x); x = x_new
        if e <= tol:
            break
    return x

def tsvd_fnr(Phi, y, p, lam=5e-7, rmin=1e-15, nmax=100, tol=1e-6):
    """FNR แบบจุดตรึงที่หาตัวผกผันตัวแรกด้วยการแยกค่าเอกฐานแบบตัดทอน (TSVD-FNR; IEEE Access 2024, รูปแบบที่ 1)
       ตัวผกผันของ diag(|x|^{2-p}) เก็บเฉพาะค่าเอกฐานที่ >= rmin * ค่าเอกฐานมากที่สุด (ที่เหลือให้ส่วนกลับเป็นศูนย์)
       ตัวผกผันตัวที่สอง (n x n) คำนวณตรง"""
    m, n = Phi.shape
    x = np.ones(n); PtP = Phi.T @ Phi; Pty = Phi.T @ y
    for _ in range(nmax):
        d = np.abs(x) ** (2 - p)
        if d.max() == 0:                        # ตัววนซ้ำเป็นศูนย์ทั้งหมด: ไม่มีค่าเอกฐานให้ตัดทอน
            return np.zeros(n)
        keep = d >= rmin * d.max()
        w = np.zeros(n); w[keep] = 1.0 / d[keep]
        M = PtP + lam * p * np.diag(w)
        if keep.sum() >= n - m:
            x_new = np.linalg.solve(M, Pty)
        else:                                   # เมทริกซ์เอกฐาน: ใช้ผลเฉลยนอร์มน้อยที่สุด
            x_new = np.linalg.lstsq(M, Pty, rcond=None)[0]
        e = _rel_change(x_new, x); x = x_new
        if e <= tol:
            break
    return x

def frozen_weight_mse(Phi, x0, p, sigma2, delta=None):
    """การประมาณ MSE แบบตรึงน้ำหนักของ FOCUSS (JCSSE 2023, สมการ (20)) ในรูปความแปรปรวน + ความเอนเอียงกำลังสอง
       Q = D Phi^T (Phi D Phi^T)^+ (delta=None, ตัวผกผันเทียม) หรือ D Phi^T (Phi D Phi^T + delta I)^{-1}, D = diag(|x0|^{2-p})"""
    d = np.abs(x0) ** (2 - p); G = (Phi * d) @ Phi.T
    Gi = np.linalg.pinv(G, rcond=1e-12) if delta is None else np.linalg.inv(G + delta * np.eye(Phi.shape[0]))
    Q = (d[:, None] * Phi.T) @ Gi
    b = Q @ (Phi @ x0) - x0
    return sigma2 * np.sum(Q * Q), float(b @ b)

# ============================================================
# fig04-prox : ตัวดำเนินการใกล้เคียงของ lambda|x|^p สำหรับ p = 0, 1/2, 1
# ============================================================
if want("prox"):
    def prox_lp_scalar(v, lam, p, grid=None):
        """prox ของ lambda |x|^p ในหนึ่งมิติ หาด้วยการค้นบนกริดละเอียด (เฉพาะเพื่อวาดรูป)"""
        if p == 1:
            return np.sign(v) * max(abs(v) - lam, 0.0)
        if p == 0:
            return v if 0.5 * v * v > lam else 0.0
        xs = np.linspace(-abs(v) - 0.5, abs(v) + 0.5, 20001)
        f = 0.5 * (xs - v) ** 2 + lam * np.abs(xs) ** p
        return xs[np.argmin(f)]

    lam = 1.0
    v = np.linspace(-4, 4, 801)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.plot(v, v, color=GRY, lw=0.8, ls=":", label="identity")
    for p, ls, col in [(1.0, "-", INK), (0.5, "--", ACC), (0.0, "-.", RED)]:
        ax.plot(v, [prox_lp_scalar(t, lam, p) for t in v], ls=ls, color=col, label=f"$p={p:g}$")
    ax.set_xlabel("$v$"); ax.set_ylabel(r"$\mathrm{prox}_{\lambda|\cdot|^p}(v)$")
    ax.axhline(0, color=GRY, lw=0.4); ax.axvline(0, color=GRY, lw=0.4)
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "fig04-prox")
    # ค่าที่อ้างในเนื้อหา: เกณฑ์ของ p=0 คือ |v| > sqrt(2 lam)
    print(f"prox p=0 threshold sqrt(2*lam) = {np.sqrt(2*lam):.3f}; p=1/2 jump near v = {v[np.argmax([prox_lp_scalar(t,lam,0.5)>0 for t in v])]:.3f}")

# ============================================================
# fig04-certificate : ใบรับรองคู่กันของ basis pursuit
# ============================================================
if want("certificate"):
    rng = np.random.default_rng(52)
    n, m, k = 64, 32, 5
    Phi = gaussian(m, n, rng)
    x, S = sparse_signal(n, k, rng)
    y = Phi @ x
    xbp = basis_pursuit(Phi, y)
    # ใบรับรองคู่กันแบบนอร์มน้อยที่สุด (Fuchs 2004; Tropp 2006): u = Phi_S (Phi_S^T Phi_S)^{-1} sign(x_S)
    PhiS = Phi[:, S]
    u = PhiS @ np.linalg.solve(PhiS.T @ PhiS, np.sign(x[S]))
    eta_vec = Phi.T @ u
    u_lp = basis_pursuit_dual(Phi, y)
    print(f"certificate: BP rel.err = {rel_err(xbp, x):.2e}; ERC max |eta| off support = {np.max(np.abs(np.delete(eta_vec, S))):.3f}; "
          f"eta_S*sign(x_S) = {np.round(eta_vec[S]*np.sign(x[S]),3)}; LP duality gap = {np.sum(np.abs(xbp)) - y@u_lp:.2e}; "
          f"||x||_1 = {np.sum(np.abs(x)):.4f}, y^T u (ERC) = {y@u:.4f}")
    fig, axes = plt.subplots(2, 1, figsize=(5.2, 3.6), sharex=True)
    axes[0].stem(np.arange(n), x, linefmt=INK, markerfmt="o", basefmt=" ")
    axes[0].plot(np.arange(n), xbp, "x", color=ACC, ms=5, mew=1.2)
    axes[0].set_ylabel("$x_i$"); axes[0].set_title("(a)", fontsize=9, loc="left")
    axes[1].stem(np.arange(n), eta_vec, linefmt=GRY, markerfmt=".", basefmt=" ")
    axes[1].plot(S, eta_vec[S], "s", color=ACC, ms=5)
    axes[1].axhline(1, color=RED, lw=0.7, ls="--"); axes[1].axhline(-1, color=RED, lw=0.7, ls="--")
    axes[1].set_ylabel(r"$\eta_i = [\mathbf{\Phi}^{\mathsf{T}}\mathbf{u}]_i$"); axes[1].set_xlabel("$i$")
    axes[1].set_title("(b)", fontsize=9, loc="left"); axes[1].set_ylim(-1.3, 1.3)
    save(fig, "fig04-certificate")

# ============================================================
# fig04-lasso-path : เส้นทางผลเฉลยของ LASSO เทียบกับ lambda
# ============================================================
if want("lasso-path"):
    rng = np.random.default_rng(11)
    n, m, k = 40, 20, 4
    Phi = gaussian(m, n, rng)
    x, S = sparse_signal(n, k, rng)
    sigma = 0.05
    y = Phi @ x + sigma * rng.standard_normal(m)
    lam_max = np.max(np.abs(Phi.T @ y))
    lams = lam_max * np.logspace(0, -3, 60)
    L = np.linalg.norm(Phi, 2) ** 2
    path = np.zeros((len(lams), n)); xw = np.zeros(n)
    for i, lm in enumerate(lams):
        xw, _ = ista(Phi, y, lm, 3000, x0=xw, L=L)   # ต่อเนื่องจากผลเฉลยก่อนหน้า (warm start)
        path[i] = xw
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    for j in range(n):
        if j in S:
            ax.plot(lams / lam_max, path[:, j], color=ACC, lw=1.4)
        else:
            ax.plot(lams / lam_max, path[:, j], color=GRY, lw=0.7)
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_xlabel(r"$\lambda/\lambda_{\max}$"); ax.set_ylabel(r"$\hat{x}_i(\lambda)$")
    ax.axhline(0, color=INK, lw=0.4)
    save(fig, "fig04-lasso-path")
    first_false = None
    for i, lm in enumerate(lams):
        supp = np.where(np.abs(path[i]) > 1e-6)[0]
        if len(set(supp) - set(S)) > 0:
            first_false = lm / lam_max; break
    print(f"lasso path: lambda_max = {lam_max:.3f}; first spurious coefficient enters at lambda/lambda_max = {first_false:.3f}; "
          f"all true coefficients nonzero from lambda/lambda_max = "
          f"{(lams/lam_max)[np.argmax([np.all(np.abs(path[i][S])>1e-6) for i in range(len(lams))])]:.3f}")

# ============================================================
# fig04-convergence : ISTA, FISTA, ADMM บนปัญหา LASSO เดียวกัน
# ============================================================
if want("convergence"):
    rng = np.random.default_rng(7)
    n, m, k = 1024, 256, 20
    Phi = gaussian(m, n, rng)
    x, S = sparse_signal(n, k, rng)
    sigma = 0.01
    y = Phi @ x + sigma * rng.standard_normal(m)
    lam = 0.1 * np.max(np.abs(Phi.T @ y))
    iters = 400
    t0 = time.time(); x_ista, h_ista = ista(Phi, y, lam, iters); t_ista = time.time() - t0
    t0 = time.time(); x_fista, h_fista = fista(Phi, y, lam, iters); t_fista = time.time() - t0
    t0 = time.time(); x_admm, h_admm = admm_lasso(Phi, y, lam, beta=1.0, iters=iters); t_admm = time.time() - t0
    # ค่าเหมาะที่สุดอ้างอิงจาก FISTA จำนวนรอบมาก
    _, h_ref = fista(Phi, y, lam, 6000)
    fstar = h_ref[-1]
    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    it = np.arange(1, iters + 1)
    ax.semilogy(it, np.maximum(h_ista - fstar, 1e-14), color=GRY, label="ISTA")
    ax.semilogy(it, np.maximum(h_fista - fstar, 1e-14), color=ACC, label="FISTA")
    ax.semilogy(it, np.maximum(h_admm - fstar, 1e-14), color=INK, ls="--", label=r"ADMM ($\beta = 1$)")
    ax.set_xlabel("$t$"); ax.set_ylabel(r"$f(\mathbf{x}^{(t)}) - f^\star$")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "fig04-convergence")
    def it_to(h, tol):
        idx = np.where(h - fstar <= tol)[0]
        return int(idx[0]) + 1 if len(idx) else None
    print(f"convergence: n={n} m={m} k={k} lambda={lam:.4f} f*={fstar:.5f}; iterations to gap 1e-6: ISTA {it_to(h_ista,1e-6)}, FISTA {it_to(h_fista,1e-6)}, ADMM {it_to(h_admm,1e-6)}; "
          f"time for {iters} iters (s): ISTA {t_ista:.2f}, FISTA {t_fista:.2f}, ADMM {t_admm:.2f}; rel.err FISTA = {rel_err(x_fista,x):.3f}, supp recovered = {set(np.where(np.abs(x_fista)>1e-3)[0])==set(S)}")
    # ผลของ beta ต่อ ADMM
    for b in [0.1, 1.0, 10.0]:
        _, hb = admm_lasso(Phi, y, lam, beta=b, iters=iters)
        print(f"  ADMM beta={b}: iterations to gap 1e-6 = {it_to(hb,1e-6)}")

# ============================================================
# fig04-irls : อัตราความสำเร็จเทียบกับ m สำหรับ BP, IRLS p=1, IRLS p=1/2
# ============================================================
if want("irls"):
    rng = np.random.default_rng(3)
    n, k, trials = 256, 16, 50
    ms = np.arange(32, 161, 8)
    rates = {"BP (LP)": [], "IRLS $p=1$": [], "IRLS $p=0.5$": []}
    for m in ms:
        ok = {key: 0 for key in rates}
        for _ in range(trials):
            Phi = gaussian(m, n, rng); x, S = sparse_signal(n, k, rng); y = Phi @ x
            xb = basis_pursuit(Phi, y)
            if xb is not None and rel_err(xb, x) < 1e-3: ok["BP (LP)"] += 1
            if rel_err(irls(Phi, y, 1.0, k), x) < 1e-3: ok["IRLS $p=1$"] += 1
            if rel_err(irls_cy(Phi, y, 0.5), x) < 1e-3: ok["IRLS $p=0.5$"] += 1
        for key in rates: rates[key].append(ok[key] / trials)
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.plot(ms, rates["BP (LP)"], "o-", color=INK, ms=3.5, label="BP (LP)")
    ax.plot(ms, rates["IRLS $p=1$"], "s--", color=GRY, ms=3.5, label="IRLS $p=1$")
    ax.plot(ms, rates["IRLS $p=0.5$"], "^-", color=ACC, ms=4, label="IRLS $p=0.5$")
    ax.set_xlabel("$m$"); ax.set_ylabel("success rate")
    ax.set_ylim(-0.03, 1.03); ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "fig04-irls")
    def m95(r):
        idx = np.where(np.array(r) >= 0.95)[0]
        return int(ms[idx[0]]) if len(idx) else None
    print(f"irls: n={n} k={k}; m for >=95% success: BP {m95(rates['BP (LP)'])}, IRLS p=1 {m95(rates['IRLS $p=1$'])}, IRLS p=0.5 {m95(rates['IRLS $p=0.5$'])}")
    print("  rates table (m, BP, IRLS1, IRLS0.5):")
    for i, m in enumerate(ms):
        print(f"   {m:4d} {rates['BP (LP)'][i]:.2f} {rates['IRLS $p=1$'][i]:.2f} {rates['IRLS $p=0.5$'][i]:.2f}")

# ============================================================
# fig04-fnr : การทำซ้ำอิสระของข้อค้นพบในบทความ IP-FNR / TSVD-FNR / IRLS-DR
#   (ก) RMSRE เทียบกับ p กรณีไร้สัญญาณรบกวน  (ข) RMSRE เทียบกับพารามิเตอร์ก่อกวน eta
#   และตาราง RMSRE เทียบกับ SNR (พิมพ์ออกทางจอ ใช้ในตาราง tab:fnr-noisy)
# ============================================================
# ============================================================
# แบบฝึกหัด 4.6 : การถ่วงน้ำหนักซ้ำบนนอร์ม l1 (Candes--Wakin--Boyd 2008) เทียบกับ IRLS p = 1/2
#   ขนาดเดียวกับภาพ fig04-irls (n = 256, k = 16, สุ่ม 50 ครั้งต่อ m, สำเร็จเมื่อ rel. err < 1e-3)
#   แต่ใช้ seed แยก จึงพิมพ์ผลของ BP และ IRLS p = 1/2 บนปัญหาชุดเดียวกันไว้เทียบด้วย
# ============================================================
def weighted_bp(Phi, y, w):
    """min sum_i w_i |x_i| s.t. Phi x = y ผ่านกำหนดการเชิงเส้น (HiGHS); คืน None หากตัวแก้ไม่สำเร็จ"""
    m, n = Phi.shape
    res = linprog(np.concatenate([w, w]), A_eq=np.hstack([Phi, -Phi]), b_eq=y,
                  bounds=(0, None), method="highs")
    return None if res.status != 0 else res.x[:n] - res.x[n:]

if want("rwl1"):
    rng = np.random.default_rng(46)
    n, k, trials, eps_rw, n_rw = 256, 16, 50, 0.1, 5
    ms = np.arange(40, 89, 8)
    rates = {"BP": [], "RW-l1": [], "IRLS p=0.5": []}; worse = 0; better = 0
    for m in ms:
        ok = {key: 0 for key in rates}
        for _ in range(trials):
            Phi = gaussian(m, n, rng); x, S = sparse_signal(n, k, rng); y = Phi @ x
            xb = weighted_bp(Phi, y, np.ones(n)); xr = xb
            for _ in range(n_rw):
                if xr is None: break
                xr = weighted_bp(Phi, y, 1.0 / (np.abs(xr) + eps_rw))
            sb = xb is not None and rel_err(xb, x) < 1e-3
            sr = xr is not None and rel_err(xr, x) < 1e-3
            ok["BP"] += sb; ok["RW-l1"] += sr
            ok["IRLS p=0.5"] += rel_err(irls_cy(Phi, y, 0.5), x) < 1e-3
            worse += (sb and not sr); better += (sr and not sb)
        for key in rates: rates[key].append(ok[key] / trials)
    print(f"rwl1 (exercise 4.6): n={n} k={k} trials={trials} eps={eps_rw}, BP + {n_rw} reweighted solves, seed 46")
    print("   m     BP  RW-l1  IRLS p=0.5")
    for i, m in enumerate(ms):
        print(f"   {m:3d}  {rates['BP'][i]:.2f}  {rates['RW-l1'][i]:.2f}  {rates['IRLS p=0.5'][i]:.2f}")
    def m95r(r):
        idx = np.where(np.array(r) >= 0.95)[0]
        return int(ms[idx[0]]) if len(idx) else None
    print("   m for >=95%: " + ", ".join(f"{key} {m95r(v)}" for key, v in rates.items()))
    print(f"   trials where BP succeeded but reweighting failed: {worse}; reweighting succeeded but BP failed: {better}")

if want("fnr"):
    m, n, k = 128, 256, 32
    def draw(rng):
        Phi = gaussian(m, n, rng)
        x = np.zeros(n); S = rng.choice(n, k, replace=False); x[S] = rng.standard_normal(k) / np.sqrt(k)
        return Phi, x
    rmsre = lambda e: float(np.sqrt(np.mean(np.square(e))))
    # --- ความสมมูลของรูป n x n (บทความ) กับรูป m x m (เอกลักษณ์การผกผันเมทริกซ์)
    rng = np.random.default_rng(1)
    Phi, x = draw(rng); y = Phi @ x
    t0 = time.time(); xa = ip_fnr(Phi, y, 0.9, form="nn"); ta = time.time() - t0
    t0 = time.time(); xb = ip_fnr(Phi, y, 0.9, form="mm"); tb = time.time() - t0
    print(f"fnr equivalence n x n vs m x m: rel diff {np.linalg.norm(xa-xb)/np.linalg.norm(xa):.1e}; time {ta*1e3:.1f} ms vs {tb*1e3:.1f} ms")
    # --- (ก) RMSRE เทียบกับ p
    ps = np.round(np.arange(0.5, 1.0001, 0.05), 2); trials = 200
    algs = {"FOCUSS": lambda P, yy, p: irls_dr(P, yy, p, xi=0.0),
            "IRLS-DR": lambda P, yy, p: irls_dr(P, yy, p, xi=1e-6),
            "IP-FNR": lambda P, yy, p: ip_fnr(P, yy, p),
            "TSVD-FNR": lambda P, yy, p: tsvd_fnr(P, yy, p)}
    resp = {a: [] for a in algs}
    for p in ps:
        errs = {a: [] for a in algs}
        r2 = np.random.default_rng(100)
        for _ in range(trials):
            Phi, x = draw(r2); y = Phi @ x
            for a, fn in algs.items():
                errs[a].append(rel_err(fn(Phi, y, p), x))
        for a in algs:
            resp[a].append(rmsre(errs[a]))
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.7))
    axes[0].semilogy(ps, resp["IRLS-DR"], "o-", color=GRY, ms=3.5, label="IRLS-DR")
    axes[0].semilogy(ps, resp["IP-FNR"], "s--", color=ACC, ms=3.5, label="IP-FNR")
    axes[0].semilogy(ps, resp["TSVD-FNR"], "^-.", color=RED, ms=3.5, label="TSVD-FNR")
    axes[0].set_xlabel("$p$"); axes[0].set_ylabel("RMSRE"); axes[0].set_title("(a)", fontsize=9, loc="left")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    print(f"fnr RMSRE vs p (noiseless, m,n,k={m},{n},{k}, {trials} trials, lam=5e-7, eta=xi=1e-6, rmin=1e-15):")
    for i, p in enumerate(ps):
        print(f"   p={p:.2f}: " + "  ".join(f"{a} {resp[a][i]:.2e}" for a in algs))
    # --- (ข) RMSRE เทียบกับ eta
    etas = [1e-300, 1e-12, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    rese = {0.8: [], 0.9: []}
    for p in rese:
        for eta in etas:
            errs = []; r2 = np.random.default_rng(300)
            for _ in range(trials):
                Phi, x = draw(r2); y = Phi @ x
                errs.append(rel_err(ip_fnr(Phi, y, p, eta=eta), x))
            rese[p].append(rmsre(errs))
    xs = np.arange(len(etas))
    axes[1].semilogy(xs, rese[0.8], "s--", color=ACC, ms=3.5, label="$p=0.8$")
    axes[1].semilogy(xs, rese[0.9], "o-", color=INK, ms=3.5, label="$p=0.9$")
    axes[1].set_xticks(xs); axes[1].set_xticklabels([f"$10^{{{int(np.round(np.log10(e)))}}}$" for e in etas], fontsize=6.5, rotation=45)
    axes[1].set_xlabel(r"$\eta$"); axes[1].set_ylabel("RMSRE"); axes[1].set_title("(b)", fontsize=9, loc="left")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig04-fnr")
    print(f"fnr RMSRE vs eta (noiseless, {trials} trials, lam=5e-7):")
    for p in rese:
        print(f"   p={p}: " + "  ".join(f"{e:.0e}:{r:.1e}" for e, r in zip(etas, rese[p])))
    # --- ตาราง RMSRE เทียบกับ SNR (p = 0.9)
    trials_n = 100
    algs_n = {"BP": lambda P, yy: basis_pursuit(P, yy), "IRLS-DR": lambda P, yy: irls_dr(P, yy, 0.9),
              "IP-FNR": lambda P, yy: ip_fnr(P, yy, 0.9), "TSVD-FNR": lambda P, yy: tsvd_fnr(P, yy, 0.9)}
    print(f"fnr RMSRE vs SNR (p=0.9, {trials_n} trials):")
    for snr in [10, 20, 30, 40, 50]:
        errs = {a: [] for a in algs_n}; r2 = np.random.default_rng(400)
        for _ in range(trials_n):
            Phi, x = draw(r2); s0 = Phi @ x
            y = s0 + np.sqrt(np.mean(s0 ** 2) * 10 ** (-snr / 10)) * r2.standard_normal(m)
            for a, fn in algs_n.items():
                errs[a].append(rel_err(fn(Phi, y), x))
        print(f"   SNR {snr:2d} dB: " + "  ".join(f"{a} {rmsre(errs[a]):.3e}" for a in algs_n))
    # --- การประมาณ MSE แบบตรึงน้ำหนัก: ตัวผกผันเทียมให้ขอบเขตออราเคิลพอดี ส่วน delta > 0 ให้ความเอนเอียงที่ขึ้นกับ p
    rng = np.random.default_rng(5)
    Phi = gaussian(19, 64, rng); x = np.zeros(64); S = np.sort(rng.choice(64, 3, replace=False)); x[S] = rng.standard_normal(3) / np.sqrt(3)
    PS = Phi[:, S]; orc = 0.01 * np.trace(np.linalg.inv(PS.T @ PS))
    print(f"frozen-weight MSE (m,n,k=19,64,3, sigma^2=0.01): oracle {orc:.6f}")
    for p in [0.3, 0.9, 1.0]:
        v, b2 = frozen_weight_mse(Phi, x, p, 0.01); print(f"   pinv   p={p}: var {v:.6f}  bias^2 {b2:.1e}")
    for dl in [1e-6, 1e-3, 1e-1]:
        for p in [0.5, 0.9]:
            v, b2 = frozen_weight_mse(Phi, x, p, 0.01, delta=dl); print(f"   delta={dl:g} p={p}: var {v:.6f}  bias^2 {b2:.2e}")

# ============================================================
# fig04-focuss-mse : MSE ของ FOCUSS เทียบ SNR กับขอบเขตออราเคิล sigma^2 tr((Phi_S^T Phi_S)^{-1})
# ============================================================
if want("focuss-mse"):
    rng = np.random.default_rng(9)
    n, m, k, trials = 128, 48, 6, 300
    Phi = gaussian(m, n, rng)
    x, S = sparse_signal(n, k, rng, kind="sign")
    PhiS = Phi[:, S]
    oracle_const = np.trace(np.linalg.inv(PhiS.T @ PhiS))
    snrs_db = np.arange(0, 41, 5)
    mse_focuss, mse_oracle, mse_minnorm, supp_rate = [], [], [], []
    signal_power = np.sum((Phi @ x) ** 2) / m
    for snr in snrs_db:
        sigma2 = signal_power / 10 ** (snr / 10)
        errs = []; errs_mn = []; hits = 0
        for _ in range(trials):
            y = Phi @ x + np.sqrt(sigma2) * rng.standard_normal(m)
            xh = focuss(Phi, y, p=0.5, lam=sigma2 * m, iters=60)
            errs.append(np.sum((xh - x) ** 2))
            top = np.argsort(np.abs(xh))[::-1][:k]
            hits += set(top) == set(S)
            xmn = Phi.T @ np.linalg.solve(Phi @ Phi.T, y)
            errs_mn.append(np.sum((xmn - x) ** 2))
        mse_focuss.append(np.mean(errs)); mse_oracle.append(sigma2 * oracle_const)
        mse_minnorm.append(np.mean(errs_mn)); supp_rate.append(hits / trials)
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.semilogy(snrs_db, mse_focuss, "o-", color=INK, ms=3.5, label=r"FOCUSS ($p = 0.5$)")
    ax.semilogy(snrs_db, mse_oracle, "-", color=ACC, lw=1.2, label=r"$\sigma^2\,\mathrm{tr}[(\mathbf{\Phi}_S^{\mathsf{T}}\mathbf{\Phi}_S)^{-1}]$")
    ax.semilogy(snrs_db, mse_minnorm, ":", color=GRY, label="min-norm LS")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("MSE"); ax.legend(frameon=False, fontsize=8)
    save(fig, "fig04-focuss-mse")
    print(f"focuss-mse: n={n} m={m} k={k} oracle const tr((Phi_S^T Phi_S)^-1) = {oracle_const:.3f}")
    for s_, a, b, c, r in zip(snrs_db, mse_focuss, mse_oracle, mse_minnorm, supp_rate):
        print(f"   SNR {s_:2d} dB: FOCUSS {a:.3e}  oracle {b:.3e}  ratio {a/b:.2f}  min-norm {c:.3e}  support-rate {r:.2f}")
    # ข้อสังเกตท้ายหัวข้อ 4.6.4: เมื่อ xi เล็กมาก (1e-6) การวนซ้ำปรับเข้ากับสัญญาณรบกวน ผลเฉลยไม่เบาบางพอดี
    # ค่ามัธยฐานของความคลาดเคลื่อนกำลังสองจึงสูงกว่าขอบเขตออราเคิลแม้ที่ SNR สูง (ขนาดปัญหาเดียวกับบทความ JCSSE 2023)
    for (k2, m2, n2) in [(3, 19, 64), (2, 51, 128)]:
        for snr in [20, 40]:
            r2 = np.random.default_rng(77); se = []; orc = []; exact = 0; cnt = 0
            for _ in range(150):
                P2 = gaussian(m2, n2, r2); x2 = np.zeros(n2); S2 = r2.choice(n2, k2, replace=False)
                x2[S2] = r2.standard_normal(k2) / np.sqrt(k2); sig2 = 10 ** (-snr / 10) / m2
                A2 = P2[:, S2]; orc.append(sig2 * np.trace(np.linalg.inv(A2.T @ A2)))
                for _ in range(10):
                    xh = irls_dr(P2, P2 @ x2 + np.sqrt(sig2) * r2.standard_normal(m2), 0.9, xi=1e-6)
                    se.append(np.sum((xh - x2) ** 2)); cnt += 1
                    exact += set(np.flatnonzero(np.abs(xh) > 1e-3 * np.abs(xh).max())) == set(S2)
            print(f"small-xi check k,m,n={k2},{m2},{n2} SNR {snr} dB: median sq. error {np.median(se):.2e}  "
                  f"mean oracle {np.mean(orc):.2e}  ratio {np.median(se)/np.mean(orc):.1f}  exact-support rate {exact/cnt:.2f}")

# ============================================================
# fig04-runtime : เวลาคำนวณเทียบกับ n (m = n/4, k = n/32)
# ============================================================
if want("runtime"):
    rng = np.random.default_rng(13)
    ns = [128, 256, 512, 1024, 2048]
    T = {"BP (LP)": [], "FISTA": [], "ADMM": [], "IRLS $p=1$": []}
    for n in ns:
        m, k = n // 4, n // 32
        Phi = gaussian(m, n, rng); x, S = sparse_signal(n, k, rng); y = Phi @ x
        lam = 1e-4 * np.max(np.abs(Phi.T @ y))
        t0 = time.time(); basis_pursuit(Phi, y); T["BP (LP)"].append(time.time() - t0)
        t0 = time.time(); fista(Phi, y, lam, 500); T["FISTA"].append(time.time() - t0)
        t0 = time.time(); admm_lasso(Phi, y, lam, 1.0, 300); T["ADMM"].append(time.time() - t0)
        t0 = time.time(); irls(Phi, y, 1.0, k, iters=300); T["IRLS $p=1$"].append(time.time() - t0)
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for key, mk, col in [("BP (LP)", "o", INK), ("FISTA", "s", ACC), ("ADMM", "^", GRY), ("IRLS $p=1$", "d", RED)]:
        ax.loglog(ns, T[key], marker=mk, ms=3.5, color=col, label=key, base=2)
    ax.set_xlabel("$n$"); ax.set_ylabel("time (s)"); ax.legend(frameon=False, fontsize=8)
    save(fig, "fig04-runtime")
    print("runtime (s) for n =", ns)
    for key in T:
        print(f"   {key:12s}", " ".join(f"{t:8.3f}" for t in T[key]),
              f" slope(log2) = {np.polyfit(np.log2(ns), np.log2(T[key]), 1)[0]:.2f}")
