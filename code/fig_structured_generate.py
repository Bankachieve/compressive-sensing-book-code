# -*- coding: utf-8 -*-
"""สร้างรูปประกอบเพิ่มเติมของบทที่ 1 และ 2 ว่าด้วยการถดถอยเบาบาง
ความเบาบางเชิงโครงสร้าง และโครงสร้างแรงก์ต่ำ -> figures/*.pdf
ใช้เฉพาะ NumPy และ Matplotlib เพื่อให้ทำซ้ำผลได้โดยไม่ต้องพึ่งไลบรารีอื่น"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle  # กริดพื้นหลังและตำแหน่ง legend แบบเดียวกันทั้งเล่ม
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

def stem(ax, v, color=INK, ms=3.2, lw=0.9):
    idx = np.arange(1, len(v) + 1)
    ax.vlines(idx, 0, v, color=color, lw=lw)
    ax.plot(idx, v, "o", color=color, ms=ms)
    ax.axhline(0, color=GRY, lw=0.5)

# ============================================================
# fig01-regression : กำลังสองน้อยที่สุด / ริดจ์ / LASSO ในปัญหาการถดถอย
# ============================================================
rng = np.random.default_rng(7)
N, P, K = 80, 40, 4                       # N ตัวอย่าง P ปัจจัย K ปัจจัยที่มีผลจริง
H = rng.standard_normal((N, P))
H /= np.linalg.norm(H, axis=0)            # ทำให้สดมภ์มีนอร์มหนึ่งหน่วย
x_true = np.zeros(P)
supp = np.array([4, 11, 23, 31])
x_true[supp] = np.array([3.0, -2.5, 2.0, -1.5])
e = 0.05 * rng.standard_normal(N)
y = H @ x_true + e

x_ls = np.linalg.lstsq(H, y, rcond=None)[0]                     # กำลังสองน้อยที่สุด
lam_ridge = 0.5
x_ridge = np.linalg.solve(H.T @ H + lam_ridge * np.eye(P), H.T @ y)  # ริดจ์

def ista(H, y, lam, iters=3000):
    """iterative soft thresholding สำหรับ min 0.5||y-Hx||^2 + lam||x||_1"""
    L = np.linalg.norm(H, 2) ** 2
    x = np.zeros(H.shape[1])
    for _ in range(iters):
        g = x - (H.T @ (H @ x - y)) / L
        x = np.sign(g) * np.maximum(np.abs(g) - lam / L, 0.0)
    return x
lam_lasso = 0.15
x_lasso = ista(H, y, lam_lasso)

def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)
print(f"regression: ||x_ls||_0={np.sum(np.abs(x_ls) > 1e-6)}, "
      f"||x_ridge||_0={np.sum(np.abs(x_ridge) > 1e-6)}, "
      f"||x_lasso||_0={np.sum(np.abs(x_lasso) > 1e-6)}")
print(f"regression rel.err: LS={rel(x_ls, x_true):.3f}, ridge={rel(x_ridge, x_true):.3f}, "
      f"lasso={rel(x_lasso, x_true):.3f}; lasso support={np.flatnonzero(np.abs(x_lasso) > 1e-6)}")

fig, axes = plt.subplots(2, 2, figsize=(5.6, 3.3), sharex=True, sharey=True)
for ax, v, lab in zip(axes.ravel(), [x_true, x_ls, x_ridge, x_lasso],
                      ["(a)", "(b)", "(c)", "(d)"]):
    stem(ax, v)
    ax.text(0.02, 0.9, lab, transform=ax.transAxes, fontsize=9, fontweight="bold")
    ax.set_xlim(0, P + 1); ax.set_ylim(-3.4, 3.6)
    ax.set_yticks([-3, 0, 3])
for ax in axes[1]:
    ax.set_xlabel("index $j$")
fig.tight_layout(h_pad=0.4, w_pad=0.6)
save(fig, "fig01-regression")

# ============================================================
# fig02-structured : แบบแผนความเบาบางสี่ลักษณะ
# ============================================================
rng = np.random.default_rng(3)
n = 48
# (a) เวกเตอร์เบาบางทั่วไป
v_sparse = np.zeros(n); idx = rng.choice(n, 6, replace=False)
v_sparse[idx] = rng.uniform(0.6, 1.5, 6) * rng.choice([-1, 1], 6)
# (b) เวกเตอร์เบาบางเชิงกลุ่ม (กลุ่มละ 6 สมาชิก 8 กลุ่ม เลือก 2 กลุ่ม)
v_block = np.zeros(n); g = 6
for b in [1, 5]:
    v_block[b*g:(b+1)*g] = rng.uniform(0.5, 1.5, g) * rng.choice([-1, 1], g)
# (c) เมทริกซ์เบาบางเชิงแถว 24 x 8
X_row = np.zeros((24, 8)); rows = [3, 9, 16, 20]
X_row[rows, :] = rng.uniform(0.4, 1.0, (len(rows), 8))
# (d) เมทริกซ์แรงก์ต่ำ 24 x 24 แรงก์ 2
t = np.linspace(0, 1, 24)
U = np.stack([np.sin(np.pi * t), np.exp(-((t - 0.3) / 0.2) ** 2)], axis=1)
V = np.stack([t, np.cos(2 * np.pi * t) ** 2], axis=1)
X_lr = U @ V.T; X_lr = X_lr / X_lr.max()          # แรงก์ 2 ทุกสมาชิกไม่เป็นศูนย์

fig = plt.figure(figsize=(5.6, 3.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.35], hspace=0.55, wspace=0.35)
ax = fig.add_subplot(gs[0, 0]); stem(ax, v_sparse, ms=2.6, lw=0.8)
ax.set_xlim(0, n + 1); ax.set_ylim(-1.8, 1.8); ax.set_yticks([-1, 0, 1])
ax.set_title("(a)", loc="left", fontsize=9, fontweight="bold")
ax = fig.add_subplot(gs[0, 1])
for b in range(n // g):
    ax.axvspan(b*g + 0.5, (b+1)*g + 0.5, color=FILL if b % 2 == 0 else "white", lw=0, zorder=0)
stem(ax, v_block, ms=2.6, lw=0.8)
ax.set_xlim(0, n + 1); ax.set_ylim(-1.8, 1.8); ax.set_yticks([-1, 0, 1])
ax.set_title("(b)", loc="left", fontsize=9, fontweight="bold")
ax = fig.add_subplot(gs[1, 0])
ax.imshow(X_row, cmap="Blues", aspect="auto", vmin=0, vmax=1.2)
ax.set_xticks([0, 7]); ax.set_xticklabels([1, 8]); ax.set_yticks([0, 23]); ax.set_yticklabels([1, 24])
ax.set_title("(c)", loc="left", fontsize=9, fontweight="bold")
ax = fig.add_subplot(gs[1, 1])
ax.imshow(X_lr, cmap="Blues", aspect="equal", vmin=0, vmax=1)
ax.set_xticks([0, 23]); ax.set_xticklabels([1, 24]); ax.set_yticks([0, 23]); ax.set_yticklabels([1, 24])
ax.set_title("(d)", loc="left", fontsize=9, fontweight="bold")
save(fig, "fig02-structured")

# ============================================================
# fig02-lowrank : การแยกค่าเอกฐานของภาพทดสอบและการประมาณแรงก์ต่ำ (Eckart--Young)
# ============================================================
def phantom(n=256):
    ells = [(1.0, .69, .92, 0, 0, 0), (-.8, .6624, .874, 0, -.0184, 0),
            (-.2, .11, .31, .22, 0, -18), (-.2, .16, .41, -.22, 0, 18),
            (.1, .21, .25, 0, .35, 0), (.1, .046, .046, 0, .1, 0),
            (.1, .046, .046, 0, -.1, 0), (.1, .046, .023, -.08, -.605, 0),
            (.1, .023, .023, 0, -.606, 0), (.1, .023, .046, .06, -.605, 0)]
    y, xg = np.mgrid[-1:1:n*1j, -1:1:n*1j]
    img = np.zeros((n, n))
    for A, a, b, x0, y0, phi in ells:
        th = np.deg2rad(phi)
        xr = (xg - x0)*np.cos(th) + (y + y0)*np.sin(th)
        yr = -(xg - x0)*np.sin(th) + (y + y0)*np.cos(th)
        img[(xr/a)**2 + (yr/b)**2 <= 1] += A
    return np.clip(img, 0, 1)

img = phantom(256)
Uu, s, Vt = np.linalg.svd(img, full_matrices=False)
def rank_k(k):
    return (Uu[:, :k] * s[:k]) @ Vt[:k, :]
fro = np.linalg.norm(img)
for k in [10, 30]:
    err_formula = np.sqrt(np.sum(s[k:]**2)) / fro
    err_direct = np.linalg.norm(img - rank_k(k)) / fro
    print(f"rank {k}: rel. Frobenius error = {err_direct:.3f} (formula {err_formula:.3f})")
print(f"energy in top 30 singular values = {100*np.sum(s[:30]**2)/np.sum(s**2):.1f}%")

fig, axes = plt.subplots(1, 3, figsize=(5.6, 2.1))
ax = axes[0]
ax.semilogy(np.arange(1, 101), s[:100] / s[0], color=INK)
ax.axvline(10, color=GRY, lw=0.7, ls="--"); ax.axvline(30, color=GRY, lw=0.7, ls="--")
ax.set_xlabel("index $i$"); ax.set_ylabel(r"$\sigma_i/\sigma_1$")
ax.set_title("(a)", loc="left", fontsize=9, fontweight="bold")
for ax, k, lab in zip(axes[1:], [10, 30], ["(b)", "(c)"]):
    ax.imshow(np.clip(rank_k(k), 0, 1), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(lab, loc="left", fontsize=9, fontweight="bold")
fig.tight_layout(w_pad=0.8)
save(fig, "fig02-lowrank")
print("ALL DONE")
