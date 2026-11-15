# -*- coding: utf-8 -*-
"""สร้างรูปประกอบบทที่ 1 -> figures/fig01-*.pdf (ภาพทั้งหมดผู้แต่งสร้างเอง)"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle  # กริดพื้นหลังและตำแหน่ง legend แบบเดียวกันทั้งเล่ม
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)

INK, ACC, GRY, FILL = "#1F2A37", "#1D4ED8", "#9AA3AF", "#DBEAFE"
plt.rcParams.update({
    "font.size": 9, "axes.linewidth": 0.6, "axes.edgecolor": INK,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.labelcolor": INK, "lines.linewidth": 1.5,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
rng = np.random.default_rng(2026)

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png"); plt.close(fig)
    print("saved", name)

# ---------- fig01-sampling : การชักตัวอย่างเหนือ/ต่ำกว่าอัตราไนควิสต์ ----------
B = 3.0                                   # ความถี่สูงสุด (Hz)
f1, f2, f3 = 1.0, 2.2, 3.0
x = lambda t: (0.9*np.sin(2*np.pi*f1*t) + 0.6*np.cos(2*np.pi*f2*t + 0.4)
               + 0.4*np.sin(2*np.pi*f3*t + 1.1))
tt = np.linspace(0, 2, 2000)

def sinc_recon(ts, xs, fs, t):
    return np.sum(xs[:, None]*np.sinc(fs*(t[None, :] - ts[:, None])), axis=0)

fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.1), sharey=True)
for ax, fs, lab in zip(axes, [8.0, 4.0], ["(a)", "(b)"]):
    ts = np.arange(-3, 5, 1/fs)            # ชักตัวอย่างยาวกว่าช่วงแสดงเพื่อลดผลขอบ
    xs = x(ts)
    ax.plot(tt, x(tt), color=GRY, lw=2.2, label="x(t)")
    ax.plot(tt, sinc_recon(ts, xs, fs, tt), color=ACC, lw=1.2, ls="--")
    m = (ts >= 0) & (ts <= 2)
    ax.vlines(ts[m], 0, xs[m], color=INK, lw=0.8)
    ax.plot(ts[m], xs[m], "o", color=INK, ms=3)
    ax.set_title(f"{lab}  $f_s = {fs:g}$ Hz", fontsize=9, loc="left")
    ax.set_xlabel("t (s)"); ax.axhline(0, color=GRY, lw=0.4)
    ax.spines[["top", "right"]].set_visible(False)
save(fig, "fig01-sampling")

# ---------- fig01-spectrum : สเปกตรัมของสัญญาณที่ถูกชักตัวอย่าง ----------
def tri_fn(f, c, w):
    return np.clip(1 - np.abs(f - c)/w, 0, None)

fig, axes = plt.subplots(2, 1, figsize=(5.5, 3.3))
ff = np.linspace(-11, 11, 4001)
for ax, fs, lab in zip(axes, [8.0, 4.0], ["(a)", "(b)"]):
    for k in range(-2, 3):
        yk = tri_fn(ff, k*fs, B)
        ax.fill_between(ff, 0, yk, color=(FILL if k else "#93C5FD"), lw=0, zorder=1, alpha=0.9)
        ax.plot(ff, yk, color=INK, lw=0.8, zorder=2)
    if fs < 2*B:
        for k in [-1, 1]:
            ov = np.minimum(tri_fn(ff, 0, B), tri_fn(ff, k*fs, B))
            ax.fill_between(ff, 0, ov, color="#F59E0B", alpha=0.8, lw=0, zorder=3)
    ax.set_xlim(-11, 11); ax.set_ylim(0, 1.15)
    ax.set_yticks([]); ax.set_title(f"{lab}  $f_s = {fs:g}$ Hz,  $W = {B:g}$ Hz", fontsize=9, loc="left", pad=4)
    # สัญกรณ์ตามเนื้อหา: ความถี่สูงสุดของสัญญาณคือ W (ตัวแปร B ในรหัสเก็บค่า W)
    # ป้าย ±W วางเป็นแถวที่สองเพื่อไม่ให้ทับป้าย ±f_s เมื่อ f_s ใกล้ W
    ax.set_xticks([-fs, 0, fs]); ax.set_xticklabels([r"$-f_s$", "0", r"$f_s$"])
    ax.set_xticks([-B, B], minor=True); ax.set_xticklabels([r"$-W$", r"$W$"], minor=True)
    ax.tick_params(axis="x", which="minor", pad=13, length=3.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
axes[1].set_xlabel("f")
fig.subplots_adjust(hspace=0.75)
save(fig, "fig01-spectrum")

# ---------- fig01-timeline : เส้นเวลาพัฒนาการ ----------
# (ปี, ป้าย, ด้าน, ตำแหน่ง x ของป้าย, ระดับความสูง)
events = [
    (1928, "Nyquist", -1, 1928, 1), (1949, "Shannon", 1, 1949, 1),
    (1973, "Claerbout–Muir\n$\\ell_1$ in geophysics", -1, 1973, 1),
    (1986, "Santosa–Symes", 1, 1984, 1),
    (1993, "Mallat–Zhang\nmatching pursuit", -1, 1990, 2),
    (1996, "Tibshirani\nLASSO", 1, 1996, 2),
    (1998, "Chen–Donoho–Saunders\nbasis pursuit", -1, 2000, 1),
    (2005, "Candès–Tao\nRIP", 1, 2004, 1),
    (2006, "Candès–Romberg–Tao\nDonoho: CS", -1, 2004, 3),
    (2007, "Tropp–Gilbert OMP\nLustig: sparse MRI", 1, 2008, 3),
    (2008, "single-pixel\ncamera", -1, 2014, 2),
    (2010, "model-based CS\nsub-Nyquist ADC", 1, 2018, 2),
    (2018, "deep unrolling\nISTA-Net", -1, 2030, 1),
    (2021, "algorithm unrolling\nreview", 1, 2036, 1),
]
fig, ax = plt.subplots(figsize=(5.5, 3.7))
ax.axhline(0, color=INK, lw=1.2)
LV = {1: 0.42, 2: 0.98, 3: 1.54}
for yr, lab, side, lx, lv in events:
    h = LV[lv]*side
    ax.plot([yr, lx], [0, h], color=GRY, lw=0.7)
    ax.plot(yr, 0, "o", color=ACC if yr in (2005, 2006) else INK, ms=4, zorder=3)
    ax.text(lx, h + 0.06*side, f"{yr}\n{lab}", ha="center",
            va="bottom" if side > 0 else "top", fontsize=6.0, linespacing=1.1)
ax.set_xlim(1916, 2046); ax.set_ylim(-2.25, 2.25)
ax.set_yticks([]); ax.spines[["top", "right", "left"]].set_visible(False)
ax.set_xticks([1930, 1950, 1970, 1990, 2010, 2030])
save(fig, "fig01-timeline")

# ---------- fig01-demo : กู้คืนสัญญาณเบาบางจากการวัดน้อยกว่ามิติ ----------
n, m, k = 256, 64, 8
x0 = np.zeros(n)
supp = rng.choice(n, k, replace=False)
x0[supp] = rng.normal(0, 1, k) * rng.choice([-1, 1], k) + np.sign(rng.normal(size=k))*0.5
Phi = rng.normal(0, 1/np.sqrt(m), (m, n))
y = Phi @ x0

# ผลเฉลยนอร์มน้อยที่สุด (กำลังสองน้อยที่สุด)
x_ls = Phi.T @ np.linalg.solve(Phi @ Phi.T, y)

def omp(A, y, k):
    r = y.copy(); S = []
    for _ in range(k):
        j = int(np.argmax(np.abs(A.T @ r))); S.append(j)
        xs, *_ = np.linalg.lstsq(A[:, S], y, rcond=None)
        r = y - A[:, S] @ xs
    xh = np.zeros(A.shape[1]); xh[S] = xs
    return xh

x_omp = omp(Phi, y, k)
err_ls = np.linalg.norm(x_ls - x0)/np.linalg.norm(x0)
err_omp = np.linalg.norm(x_omp - x0)/np.linalg.norm(x0)
print(f"rel. error  LS = {err_ls:.3f},  OMP = {err_omp:.2e}")

fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.9), sharey=True)
for ax, v, lab in zip(axes, [x0, x_ls, x_omp], ["(a)", "(b)", "(c)"]):
    ax.vlines(np.arange(n), 0, v, color=INK, lw=0.8)
    ax.axhline(0, color=GRY, lw=0.4)
    ax.set_title(lab, fontsize=9, loc="left"); ax.set_xlabel("i")
    ax.spines[["top", "right"]].set_visible(False)
save(fig, "fig01-demo")

# ---------- fig01-measurements : m = C k log(n/k) เทียบ n ----------
nn = np.logspace(2, 6, 200)
fig, ax = plt.subplots(figsize=(4.2, 2.8))
ax.loglog(nn, nn, color=GRY, lw=1.2, ls="--")
ax.text(2e4, 4e4, "$m = n$", color=INK, fontsize=9, rotation=33)
for kk, ls in zip([10, 100, 1000], ["-", "-", "-"]):
    mm = 2*kk*np.log(nn/kk); ok = nn > kk*2
    ax.loglog(nn[ok], mm[ok], color=INK, lw=1.4, ls=ls)
    ax.text(nn[ok][-1]*1.1, mm[ok][-1], f"$k={kk}$", fontsize=8, va="center")
ax.set_xlim(1e2, 3e6); ax.set_xlabel("$n$"); ax.set_ylabel("$m$")
ax.spines[["top", "right"]].set_visible(False)
save(fig, "fig01-measurements")
print("ALL DONE")
