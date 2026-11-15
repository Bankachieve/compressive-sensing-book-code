# -*- coding: utf-8 -*-
"""สร้างรูปประกอบและผลการทดลองเชิงตัวเลขทั้งหมดของบทที่ 3 -> figures/fig03-*.pdf
ใช้ NumPy, SciPy (linprog สำหรับ basis pursuit) และ Matplotlib เท่านั้น
ค่าตัวเลขทุกค่าที่อ้างในคำบรรยายภาพและเนื้อหาพิมพ์ออกทางจอเมื่อรันสคริปต์นี้"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle  # กริดพื้นหลังและตำแหน่ง legend แบบเดียวกันทั้งเล่ม
from scipy.optimize import linprog
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
import os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT, exist_ok=True)
INK, ACC, GRY, FILL = "#1F2A37", "#1D4ED8", "#9AA3AF", "#DBEAFE"
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

# ------------------------------------------------------------
# ตัวแก้ปัญหาพื้นฐาน
# ------------------------------------------------------------
def basis_pursuit(Phi, y):
    """min ||x||_1 s.t. Phi x = y  ผ่านกำหนดการเชิงเส้น x = u - v, u,v >= 0"""
    m, n = Phi.shape
    c = np.ones(2 * n)
    A_eq = np.hstack([Phi, -Phi])
    res = linprog(c, A_eq=A_eq, b_eq=y, bounds=(0, None), method="highs")
    if res.status != 0:
        return None
    return res.x[:n] - res.x[n:]

def bpdn_socp(Phi, y, eps):
    """min ||x||_1 s.t. ||Phi x - y||_2 <= eps  เป็นกำหนดการกรวยอันดับสอง แก้ด้วย Clarabel ผ่าน CVXPY
    ตรวจสถานะของตัวแก้ และคืนค่าการละเมิดเงื่อนไขบังคับสัมพัทธ์ (||Phi x - y|| - eps)_+ / eps เพื่อบันทึกใน log"""
    import cvxpy as cp
    n = Phi.shape[1]; xv = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.norm1(xv)), [cp.norm2(Phi @ xv - y) <= eps])
    prob.solve(solver="CLARABEL")
    if prob.status != "optimal" or xv.value is None:
        raise RuntimeError(f"BPDN SOCP failed: status {prob.status}")
    xh = np.asarray(xv.value)
    return xh, max(0.0, np.linalg.norm(Phi @ xh - y) - eps) / eps

def omp(Phi, y, k):
    r = y.copy(); S = []
    for _ in range(k):
        S.append(int(np.argmax(np.abs(Phi.T @ r))))
        xs, *_ = np.linalg.lstsq(Phi[:, S], y, rcond=None)
        r = y - Phi[:, S] @ xs
    x = np.zeros(Phi.shape[1]); x[S] = xs
    return x

def gaussian(m, n, rng):
    return rng.standard_normal((m, n)) / np.sqrt(m)

def coherence(Phi):
    G = Phi / np.linalg.norm(Phi, axis=0)
    G = np.abs(G.T @ G); np.fill_diagonal(G, 0)
    return G.max()

# ============================================================
# fig03-geometry : เหตุใดนอร์ม l1 จึงให้ผลเฉลยเบาบาง (เรขาคณิตสองมิติ)
# ============================================================
def ball_xy(p, n=800):
    t = np.linspace(0, 2 * np.pi, n)
    c, s = np.cos(t), np.sin(t)
    return np.sign(c) * np.abs(c) ** (2 / p), np.sign(s) * np.abs(s) ** (2 / p)

# เส้นตรง {x : phi1 x1 + phi2 x2 = y} ผ่านจุด (0, 1) ด้วยความชัน -0.45
a1, a2 = 0.45, 1.0
def line(ax):
    xs = np.linspace(-1.6, 1.6, 2)
    ax.plot(xs, (1 - a1 * xs) / a2, color=GRY, lw=1.2, ls="--")
fig, axes = plt.subplots(1, 3, figsize=(5.6, 1.95))
for ax, p, lab in zip(axes, [2, 1, 0.5], ["(a)", "(b)", "(c)"]):
    if p == 2:
        r = 1 / np.sqrt(a1 ** 2 + a2 ** 2)              # รัศมีที่แตะเส้น
        xs, ys = ball_xy(2); xs, ys = r * xs, r * ys
        pt = (a1 * r ** 2, a2 * r ** 2)                   # จุดสัมผัส (ไม่เบาบาง)
    else:
        xs, ys = ball_xy(p)                               # บอลรัศมี 1 แตะที่ (0,1)
        pt = (0.0, 1.0)
    ax.fill(xs, ys, color=FILL); ax.plot(xs, ys, color=INK, lw=1.3)
    line(ax)
    ax.plot(*pt, "o", color=ACC, ms=5, zorder=5)
    ax.axhline(0, color=GRY, lw=0.5); ax.axvline(0, color=GRY, lw=0.5)
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.set_aspect("equal")
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
    ax.set_title(lab, loc="left", fontsize=9, fontweight="bold")
print(f"geometry: l2 touching point = ({a1/(a1**2+a2**2):.3f}, {a2/(a1**2+a2**2):.3f})")
save(fig, "fig03-geometry")

# ============================================================
# fig03-coherence : ความสอดคล้องร่วมของเมทริกซ์สุ่มเทียบกับขอบเขตเวลช์
# ============================================================
rng = np.random.default_rng(11)
n = 256; ms = np.array([16, 24, 32, 48, 64, 96, 128, 192])
mu_g = np.array([[coherence(gaussian(m, n, rng)) for _ in range(20)] for m in ms])
mu_b = np.array([[coherence(rng.choice([-1.0, 1.0], (m, n))) for _ in range(20)] for m in ms])
welch = np.sqrt((n - ms) / (ms * (n - 1)))
# DCT บางส่วน (แถวสุ่ม)
def dct_matrix(n):
    k = np.arange(n)[:, None]; i = np.arange(n)[None, :]
    C = np.sqrt(2 / n) * np.cos(np.pi * (2 * i + 1) * k / (2 * n)); C[0] /= np.sqrt(2)
    return C
C = dct_matrix(n)
mu_d = np.array([[coherence(C[np.sort(rng.choice(n, m, replace=False))]) for _ in range(20)] for m in ms])
print("coherence (mean over 20 trials): m, gaussian, bernoulli, partial-DCT, welch")
for i, m in enumerate(ms):
    print(f"  {m:4d}  {mu_g[i].mean():.3f}  {mu_b[i].mean():.3f}  {mu_d[i].mean():.3f}  {welch[i]:.3f}")
fig, ax = plt.subplots(figsize=(3.9, 2.5))
ax.plot(ms, mu_g.mean(1), "o-", color=INK, ms=3.5, label="Gaussian")
ax.plot(ms, mu_b.mean(1), "s--", color=INK, ms=3.5, mfc="white", label="Bernoulli")
ax.plot(ms, mu_d.mean(1), "^:", color=ACC, ms=3.5, label="partial DCT")
ax.plot(ms, welch, color=GRY, lw=1.8, label="Welch bound")
ax.plot(ms, 1 / np.sqrt(ms), color=GRY, lw=0.9, ls="-.", label=r"$1/\sqrt{m}$")
ax.set_xlabel("$m$"); ax.set_ylabel(r"$\mu(\mathbf{\Phi})$"); ax.set_xscale("log", base=2)
ax.set_xticks(ms); ax.set_xticklabels(ms); ax.set_ylim(0, 1.0)
ax.legend(frameon=False, fontsize=7.5, ncol=1)
save(fig, "fig03-coherence")

# ============================================================
# fig03-rip : การกระจายของ ||Phi x||^2/||x||^2 และค่าเอกฐานของเมทริกซ์ย่อยสุ่ม
# ============================================================
rng = np.random.default_rng(5)
m, n, k = 64, 256, 8
Phi = gaussian(m, n, rng)
ratios = []
smin, smax = [], []
for _ in range(5000):
    S = rng.choice(n, k, replace=False)
    x = np.zeros(n); x[S] = rng.standard_normal(k)
    ratios.append(np.sum((Phi @ x) ** 2) / np.sum(x ** 2))
    s = np.linalg.svd(Phi[:, S], compute_uv=False)
    smin.append(s[-1] ** 2); smax.append(s[0] ** 2)
ratios = np.array(ratios); smin = np.array(smin); smax = np.array(smax)
delta_emp = max(1 - smin.min(), smax.max() - 1)
print(f"RIP experiment m={m} n={n} k={k}: ratio range [{ratios.min():.3f}, {ratios.max():.3f}], "
      f"sigma_min^2 range [{smin.min():.3f},{smin.max():.3f}], sigma_max^2 range [{smax.min():.3f},{smax.max():.3f}], "
      f"empirical delta_k lower bound = {delta_emp:.3f}")
fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.2))
ax = axes[0]
ax.hist(ratios, bins=40, color=FILL, edgecolor=INK, lw=0.5)
ax.axvline(1, color=ACC, lw=1.2)
ax.set_xlabel(r"$\|\mathbf{\Phi}\mathbf{x}\|_2^2/\|\mathbf{x}\|_2^2$"); ax.set_ylabel("count")
ax.set_title("(a)", loc="left", fontsize=9, fontweight="bold")
ax = axes[1]
ax.hist(smin, bins=40, color=FILL, edgecolor=INK, lw=0.25, label=r"$\sigma_{\min}^2$")
ax.hist(smax, bins=40, color="white", edgecolor=ACC, lw=0.7, hatch="///", label=r"$\sigma_{\max}^2$")
ax.axvline(1, color=GRY, lw=1.0)
ax.set_xlabel(r"squared singular values of $\mathbf{\Phi}_S$"); ax.legend(frameon=False, fontsize=8)
ax.set_title("(b)", loc="left", fontsize=9, fontweight="bold")
fig.tight_layout(w_pad=1.0)
save(fig, "fig03-rip")

# ============================================================
# fig03-matrices : ตัวอย่างเมทริกซ์การวัดสี่ตระกูล
# ============================================================
rng = np.random.default_rng(2)
mm, nn = 24, 64
G = rng.standard_normal((mm, nn))
B = rng.choice([-1.0, 1.0], (mm, nn))
Cn = dct_matrix(nn); PD = Cn[np.sort(rng.choice(nn, mm, replace=False))]
# เมทริกซ์เทอพลิตซ์บางส่วน (การบิดเบือนแบบสุ่มแล้วชักตัวอย่างลง)
h = rng.standard_normal(nn)
T = np.array([np.roll(h, i) for i in range(nn)])[::nn // mm][:mm]
fig, axes = plt.subplots(2, 2, figsize=(5.6, 2.6))
for ax, M, lab in zip(axes.ravel(), [G, B, PD, T], ["(a)", "(b)", "(c)", "(d)"]):
    v = np.abs(M).max()
    ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(lab, loc="left", fontsize=9, fontweight="bold")
fig.tight_layout(h_pad=0.6, w_pad=0.6)
save(fig, "fig03-matrices")

# ============================================================
# fig03-phase : แผนภาพการเปลี่ยนวัฏภาคเชิงประจักษ์ของ basis pursuit
#               เทียบกับเส้นโค้งมิติเชิงสถิติ (Amelunxen et al. 2014)
# ============================================================
def stat_dim_l1(rho):
    """มิติเชิงสถิติ (ต่อ n) ของกรวยเชิงลง ของนอร์ม l1 ที่เวกเตอร์ rho*n-เบาบาง"""
    def J(tau):
        tail = quad(lambda u: (u - tau) ** 2 * np.exp(-u ** 2 / 2), tau, np.inf)[0]
        return rho * (1 + tau ** 2) + (1 - rho) * np.sqrt(2 / np.pi) * tail
    return minimize_scalar(J, bounds=(0, 10), method="bounded").fun

rng = np.random.default_rng(1)
n = 100; trials = 20
deltas = np.linspace(0.1, 0.9, 9)
rhos = np.linspace(0.1, 0.9, 9)
success = np.zeros((len(rhos), len(deltas)))
for j, d in enumerate(deltas):
    m = int(round(d * n))
    for i, r in enumerate(rhos):
        k = max(1, int(round(r * m)))
        ok = 0
        for _ in range(trials):
            Phi = gaussian(m, n, rng)
            x = np.zeros(n); S = rng.choice(n, k, replace=False); x[S] = rng.standard_normal(k)
            xh = basis_pursuit(Phi, Phi @ x)
            if xh is not None and np.linalg.norm(xh - x) <= 1e-4 * np.linalg.norm(x):
                ok += 1
        success[i, j] = ok / trials
print("phase transition success matrix (rows rho=k/m 0.1..0.9, cols delta=m/n 0.1..0.9):")
print(np.array2string(success, precision=2, max_line_width=120))
# เส้นโค้งทฤษฎี: m/n = psi(k/n) -> แปลงเป็น rho = k/m
rho_n = np.linspace(0.002, 0.6, 120)
psi = np.array([stat_dim_l1(r) for r in rho_n])
rho_curve = rho_n / psi
print(f"stat-dim curve check: psi(0.1)={stat_dim_l1(0.1):.3f}, psi(0.2)={stat_dim_l1(0.2):.3f}")
fig, ax = plt.subplots(figsize=(3.9, 3.0))
im = ax.imshow(success, origin="lower", cmap="Blues", vmin=0, vmax=1,
               extent=[0.05, 0.95, 0.05, 0.95], aspect="auto")
ax.plot(psi, rho_curve, color=INK, lw=1.6)
ax.set_xlim(0.05, 0.95); ax.set_ylim(0.05, 0.95)
ax.set_xlabel(r"$\nu = m/n$"); ax.set_ylabel(r"$\kappa = k/m$")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("success rate")
save(fig, "fig03-phase")

# ============================================================
# fig03-noisy : ความคลาดเคลื่อนของ BPDN เทียบกับระดับสัญญาณรบกวน
# ============================================================
rng = np.random.default_rng(21)
m, n, k = 64, 256, 8
Phi = gaussian(m, n, rng)
x = np.zeros(n); S = rng.choice(n, k, replace=False); x[S] = rng.standard_normal(k)
sigmas = np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.3])
errs_bp, errs_ls, errs_omp = [], [], []; viol_max = 0.0; enorms = []
for s in sigmas:
    e_bp, e_ls, e_omp = [], [], []
    for _ in range(10):
        e = s * rng.standard_normal(m)
        y = Phi @ x + e
        eps = np.linalg.norm(e)                      # ใช้ขนาดสัญญาณรบกวนจริงเป็น epsilon
        xb, viol = bpdn_socp(Phi, y, eps); viol_max = max(viol_max, viol); enorms.append(eps)
        xo = omp(Phi, y, k)
        xl = np.linalg.pinv(Phi) @ y
        e_bp.append(np.linalg.norm(xb - x)); e_omp.append(np.linalg.norm(xo - x)); e_ls.append(np.linalg.norm(xl - x))
    errs_bp.append(np.mean(e_bp)); errs_omp.append(np.mean(e_omp)); errs_ls.append(np.mean(e_ls))
errs_bp, errs_omp, errs_ls = map(np.array, (errs_bp, errs_omp, errs_ls))
eps_mean = sigmas * np.sqrt(m)                       # sigma sqrt(m) = sqrt(E||e||^2) ใช้เป็นแกนนอน (epsilon จริงของแต่ละครั้งคือ ||e||)
en = np.array(enorms).reshape(len(sigmas), -1).mean(axis=1)
print(f"noisy: BPDN solved as SOCP (Clarabel), all solves optimal; max relative constraint violation {viol_max:.1e}")
print("noisy: sigma, sigma*sqrt(m), mean measured ||e||, err_BPDN, err_OMP, err_LS, ratio err_BPDN/(sigma sqrt(m))")
for s, ep, em, a, b, c in zip(sigmas, eps_mean, en, errs_bp, errs_omp, errs_ls):
    print(f"  {s:.3f} {ep:.3f} {em:.3f} {a:.4f} {b:.4f} {c:.4f} {a/ep:.2f}")
slope = np.polyfit(np.log(eps_mean), np.log(errs_bp), 1)[0]
print(f"log-log slope of BPDN error vs eps = {slope:.2f}")
fig, ax = plt.subplots(figsize=(3.9, 2.6))
ax.loglog(eps_mean, errs_bp, "o-", color=INK, ms=4, label="BPDN")
ax.loglog(eps_mean, errs_omp, "s--", color=ACC, ms=4, mfc="white", label="OMP")
ax.loglog(eps_mean, errs_ls, "^:", color=GRY, ms=4, label="min-norm LS")
ax.loglog(eps_mean, eps_mean, color=GRY, lw=0.8, ls="-")
ax.set_xlabel(r"$\sigma\sqrt{m}$"); ax.set_ylabel(r"$\|\hat{\mathbf{x}}-\mathbf{x}\|_2$")
ax.legend(frameon=False, fontsize=8)
save(fig, "fig03-noisy")
print("ALL DONE")
