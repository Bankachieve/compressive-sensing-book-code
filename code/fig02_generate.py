# -*- coding: utf-8 -*-
"""สร้างรูปประกอบบทที่ 2 ทั้งหมด -> ../figures/ (โฟลเดอร์ figures ถัดจาก code).pdf (+ .png พรีวิว)"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle  # กริดพื้นหลังและตำแหน่ง legend แบบเดียวกันทั้งเล่ม
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT, exist_ok=True)

INK = "#1F2A37"   # หมึกหลัก (เกือบดำ อ่านได้ทั้งพิมพ์สี/ขาวดำ)
ACC = "#1D4ED8"   # น้ำเงินเข้ม ใช้เน้นจุดเดียวต่อรูป
GRY = "#9AA3AF"   # เทากลาง สำหรับเส้นอ้างอิง/กริด
FILL = "#DBEAFE"  # ฟ้าอ่อน ใช้ถมพื้นบาง ๆ

plt.rcParams.update({
    "font.size": 9, "axes.linewidth": 0.6, "axes.edgecolor": INK,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.labelcolor": INK, "lines.linewidth": 1.6,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf")
    fig.savefig(f"{OUT}/{name}.png")
    plt.close(fig)
    print("saved", name)

# ---------- fig02-unit-balls : บอลหนึ่งหน่วย lp ----------
def ball_xy(p, n=800):
    t = np.linspace(0, 2*np.pi, n)
    c, s = np.cos(t), np.sin(t)
    x = np.sign(c)*np.abs(c)**(2.0/p)
    y = np.sign(s)*np.abs(s)**(2.0/p)
    return x, y

fig, axes = plt.subplots(1, 4, figsize=(5.5, 1.75))
for ax, (p, lab) in zip(axes, [(0.5, r"$p=\frac{1}{2}$"), (1, r"$p=1$"),
                               (2, r"$p=2$"), (np.inf, r"$p=\infty$")]):
    if np.isinf(p):
        x = np.array([1, 1, -1, -1, 1]); y = np.array([1, -1, -1, 1, 1])
    else:
        x, y = ball_xy(p)
    ax.fill(x, y, color=FILL, zorder=1)
    ax.plot(x, y, color=INK, lw=1.4, zorder=2)
    ax.axhline(0, color=GRY, lw=0.5, zorder=0)
    ax.axvline(0, color=GRY, lw=0.5, zorder=0)
    ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal"); ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
    ax.set_title(lab, fontsize=10)
save(fig, "fig02-unit-balls")

# ---------- fig02-projection : การฉายเชิงตั้งฉาก ----------
fig, ax = plt.subplots(figsize=(3.8, 3.0))
t = np.linspace(-0.8, 3.4, 2)
ax.plot(t, t, color=GRY, lw=1.2, ls="--", zorder=1)
ax.annotate(r"$\mathcal{V}=\mathrm{span}\{(1,1)\}$", xy=(2.75, 2.45), fontsize=9,
            color=INK, rotation=38)
ax.annotate("", xy=(3, 1), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.8))
ax.annotate("", xy=(2, 2), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=ACC, lw=1.8))
ax.plot([2, 3], [2, 1], color=GRY, lw=1.2, ls=":")
# เครื่องหมายมุมฉาก: วาดด้วยสองเส้นสั้น
v1 = np.array([1, 1])/np.sqrt(2); v2 = np.array([1, -1])/np.sqrt(2)
p0 = np.array([2, 2]) - 0.22*v1
ax.plot([p0[0], p0[0]+0.22*v2[0]], [p0[1], p0[1]+0.22*v2[1]], color=INK, lw=0.8)
p1 = p0 + 0.22*v2
ax.plot([p1[0], p1[0]+0.22*v1[0]], [p1[1], p1[1]+0.22*v1[1]], color=INK, lw=0.8)
ax.text(3.03, 0.95, r"$\mathbf{x}=(3,1)$", fontsize=10)
ax.text(1.62, 2.12, r"$P_{\mathcal{V}}\,\mathbf{x}=(2,2)$", fontsize=10, color=ACC)
ax.text(2.6, 1.62, r"$\mathbf{x}-P_{\mathcal{V}}\,\mathbf{x}$", fontsize=9, color=INK)
ax.set_xlim(-0.6, 4.1); ax.set_ylim(-0.6, 3.2)
ax.set_aspect("equal")
ax.axhline(0, color=GRY, lw=0.5); ax.axvline(0, color=GRY, lw=0.5)
ax.set_xticks([0, 1, 2, 3]); ax.set_yticks([0, 1, 2, 3])
save(fig, "fig02-projection")

# ---------- fig02-mb-frame : เฟรม Mercedes-Benz ----------
fig, ax = plt.subplots(figsize=(3.1, 3.1))
th = np.linspace(0, 2*np.pi, 400)
ax.plot(np.cos(th), np.sin(th), color=GRY, lw=0.8, ls="--")
angs = [90, 210, 330]
labs = [r"$\boldsymbol{\psi}_1$", r"$\boldsymbol{\psi}_2$", r"$\boldsymbol{\psi}_3$"]
offs = [(0.07, 0.08), (-0.30, -0.02), (0.10, -0.02)]
for a, lb, off in zip(angs, labs, offs):
    r = np.deg2rad(a)
    ax.annotate("", xy=(np.cos(r), np.sin(r)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.8))
    ax.text(np.cos(r)+off[0], np.sin(r)+off[1], lb, fontsize=11)
ax.axhline(0, color=GRY, lw=0.5); ax.axvline(0, color=GRY, lw=0.5)
ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45)
ax.set_aspect("equal"); ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
save(fig, "fig02-mb-frame")

# ---------- fig02-dct-basis : เวกเตอร์ฐาน DCT ----------
def dct_matrix(n):
    k = np.arange(n)[:, None]; t = np.arange(n)[None, :]
    C = np.cos(np.pi*(2*t+1)*k/(2*n))
    C[0] *= np.sqrt(1.0/n); C[1:] *= np.sqrt(2.0/n)
    return C

n = 32
C = dct_matrix(n)
fig, axes = plt.subplots(2, 4, figsize=(5.5, 2.5), sharex=True, sharey=True)
for k, ax in enumerate(axes.ravel()):
    ax.axhline(0, color=GRY, lw=0.5)
    ax.plot(C[k], color=INK, lw=1.1)
    ax.plot(C[k], ".", color=ACC, ms=2.2)
    ax.set_title(rf"$j={k}$", fontsize=8, pad=2)
    ax.set_xticks([]); ax.set_yticks([])
save(fig, "fig02-dct-basis")

# ---------- fig02-dct-demo : สัญญาณเรียบ vs สัมประสิทธิ์ DCT ----------
n = 128
t = np.arange(n)
x = (1.0*np.sin(2*np.pi*2.0*t/n) + 0.6*np.cos(2*np.pi*4.5*t/n)
     + 0.35*np.sin(2*np.pi*1.0*t/n + 0.7))
s = dct_matrix(n) @ x
fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.9))
axes[0].plot(t, x, color=INK, lw=1.3)
axes[0].set_xlabel(r"$t$"); axes[0].set_title("(a)", fontsize=9, loc="left")
axes[1].vlines(np.arange(n), 0, np.abs(s), color=INK, lw=1.0)
axes[1].set_xlim(-1, 40)
axes[1].set_xlabel(r"$j$"); axes[1].set_ylabel(r"$|s_j|$")
axes[1].set_title("(b)", fontsize=9, loc="left")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
save(fig, "fig02-dct-demo")

# ---------- Shepp-Logan phantom ----------
def phantom(n=256):
    # (A, a, b, x0, y0, phi_deg) — ชุดพารามิเตอร์มาตรฐานของ Shepp–Logan
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

def haar2(img, levels=3):
    c = img.astype(float).copy()
    n = img.shape[0]
    for _ in range(levels):
        a = c[:n, :n]
        # แถว
        lo = (a[:, 0::2] + a[:, 1::2])/np.sqrt(2)
        hi = (a[:, 0::2] - a[:, 1::2])/np.sqrt(2)
        a = np.hstack([lo, hi])
        # สดมภ์
        lo = (a[0::2, :] + a[1::2, :])/np.sqrt(2)
        hi = (a[0::2, :] - a[1::2, :])/np.sqrt(2)
        c[:n, :n] = np.vstack([lo, hi])
        n //= 2
    return c

def ihaar2(c, levels=3):
    x = c.copy()
    n = x.shape[0] // (2**(levels-1))
    for _ in range(levels):
        a = x[:n, :n]
        h = n//2
        lo, hi = a[:h, :], a[h:, :]
        up = np.zeros((n, a.shape[1]))
        up[0::2, :] = (lo + hi)/np.sqrt(2)
        up[1::2, :] = (lo - hi)/np.sqrt(2)
        lo, hi = up[:, :h], up[:, h:]
        out = np.zeros_like(up)
        out[:, 0::2] = (lo + hi)/np.sqrt(2)
        out[:, 1::2] = (lo - hi)/np.sqrt(2)
        x[:n, :n] = out
        n *= 2
    return x

N = 256
ph = phantom(N)
yy, xx = np.mgrid[0:1:N*1j, 0:1:N*1j]
smooth = 0.55*xx + 0.45*np.exp(-((xx-0.3)**2 + (yy-0.25)**2)/0.08)
smooth = (smooth - smooth.min())/(smooth.max() - smooth.min())
img = 0.72*ph + 0.28*smooth
W = haar2(img, 3)

# ---------- fig02-wavelet2d ----------
fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8))
axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("(a)", fontsize=9, loc="left")
axes[1].imshow(np.log1p(60*np.abs(W)), cmap="gray")
axes[1].set_title("(b)", fontsize=9, loc="left")
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
save(fig, "fig02-wavelet2d")

# ---------- fig02-sigma2 : ยูเนียนของปริภูมิย่อย ----------
# หมายเหตุ (13 พ.ย. 2569): ภาพที่ 2.7 ในเล่มเปลี่ยนไปใช้แผนภาพ TikZ figures/tikz/fig02-sigma2.tex แล้ว; ส่วนนี้คงไว้เพื่ออ้างอิง
fig = plt.figure(figsize=(3.9, 3.3))
ax = fig.add_subplot(111, projection="3d")
r = [-1, 1]
planes = [
    ([[ -1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], "#BFD3F2"),   # ระนาบ x1x2
    ([[ -1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]], "#8FB3E8"),   # ระนาบ x1x3
    ([[ 0, -1, -1], [0, 1, -1], [0, 1, 1], [0, -1, 1]], "#5E8FD8"),   # ระนาบ x2x3
]
for verts, fc in planes:
    pc = Poly3DCollection([verts], facecolor=fc, edgecolor=INK,
                          linewidths=0.7, alpha=0.55)
    ax.add_collection3d(pc)
ax.set_xlim(r); ax.set_ylim(r); ax.set_zlim(r)
ax.set_xlabel(r"$x_1$", labelpad=-6); ax.set_ylabel(r"$x_2$", labelpad=-6)
ax.set_zlabel(r"$x_3$", labelpad=-6)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.view_init(elev=18, azim=-55)
save(fig, "fig02-sigma2")

# ---------- fig02-decay : กฎกำลังของสัมประสิทธิ์ ----------
mag = np.sort(np.abs(W).ravel())[::-1]
mag = mag[mag > 1e-12]
i = np.arange(1, len(mag)+1)
# ประมาณความชันช่วงกลาง
lo, hi = 200, 20000
q, logC = np.polyfit(np.log(i[lo:hi]), np.log(mag[lo:hi]), 1)
fig, ax = plt.subplots(figsize=(4.0, 2.9))
ax.loglog(i, mag, color=INK, lw=1.4)
ref = np.exp(logC)*i**q
ax.loglog(i[50:60000], 1.8*ref[50:60000], color=GRY, lw=1.2, ls="--")
ax.text(2e3, 1.8*np.exp(logC)*(2e3)**q*2.6, rf"$\propto i^{{{q:.1f}}}$",
        color=INK, fontsize=10)
ax.set_xlabel(r"$i$")
ax.set_ylabel(r"$|s_{(i)}|$")
ax.spines[["top", "right"]].set_visible(False)
save(fig, "fig02-decay")
print(f"decay slope q ~ {-q:.2f}")

# ---------- fig02-bestk : การประมาณ k พจน์ที่ดีที่สุด ----------
def best_k_recon(W, frac):
    k = int(frac*W.size)
    thr = np.sort(np.abs(W).ravel())[::-1][k-1]
    Wk = np.where(np.abs(W) >= thr, W, 0)
    return ihaar2(Wk, 3)

def psnr(a, b):
    mse = np.mean((a-b)**2)
    return 10*np.log10(1.0/mse)

r10 = best_k_recon(W, 0.10)
r01 = best_k_recon(W, 0.01)
p10, p01 = psnr(img, r10), psnr(img, r01)
print(f"PSNR 10% = {p10:.1f} dB,  PSNR 1% = {p01:.1f} dB")

fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.1))
for ax, (im, ti) in zip(axes, [(img, "(a)"), (r10, "(b)"), (r01, "(c)")]):
    ax.imshow(im, cmap="gray", vmin=0, vmax=1)
    ax.set_title(ti, fontsize=9, loc="left")
    ax.set_xticks([]); ax.set_yticks([])
save(fig, "fig02-bestk")

print("ALL DONE")
