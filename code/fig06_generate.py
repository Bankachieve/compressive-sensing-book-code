# -*- coding: utf-8 -*-
r"""สร้างรูปประกอบและผลการทดลองเชิงตัวเลขทั้งหมดของบทที่ 6 -> figures/fig06-*.pdf
ภาพทดสอบทุกภาพสร้างขึ้นเองด้วยแบบจำลองเชิงคณิตศาสตร์ (แฟนทอมเชปป์--โลแกน ภาพสนามสุ่มจำกัดแถบความถี่
และภาพเชิงเรขาคณิต) จึงไม่มีภาพจากแหล่งอื่นในหนังสือ
ตัวดำเนินการและขั้นตอนวิธีทุกตัวเขียนขึ้นเองด้วย NumPy/SciPy/PyWavelets ตามสมการในบท
ค่าตัวเลขทุกค่าที่อ้างในเนื้อหาและคำบรรยายภาพพิมพ์ออกทางจอเมื่อรันสคริปต์นี้
เลือกสร้างเฉพาะบางรูปได้ เช่น `python fig06_generate.py mri denoise`"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle  # กริดพื้นหลังและตำแหน่ง legend แบบเดียวกันทั้งเล่ม
import pywt
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
    "image.cmap": "gray",
})

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf")
    plt.close(fig)
    print("saved", name)

def show(ax, img, title=None, vmin=0.0, vmax=1.0):
    ax.imshow(img, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)

# ============================================================
# ภาพทดสอบเชิงคณิตศาสตร์
# ============================================================
# พารามิเตอร์ของแฟนทอมเชปป์--โลแกนดัดแปลง (ความเข้ม, แกนครึ่ง a, b, จุดศูนย์กลาง, มุมหมุนเป็นองศา)
SL = [(1.00, .69, .92, 0, 0, 0), (-.80, .6624, .8740, 0, -.0184, 0),
      (-.20, .1100, .3100, .22, 0, -18), (-.20, .1600, .4100, -.22, 0, 18),
      (.10, .2100, .2500, 0, .35, 0), (.10, .0460, .0460, 0, .10, 0),
      (.10, .0460, .0460, 0, -.10, 0), (.10, .0460, .0230, -.08, -.605, 0),
      (.10, .0230, .0230, 0, -.606, 0), (.10, .0230, .0460, .06, -.605, 0)]

def phantom(n=256):
    """แฟนทอมเชปป์--โลแกนดัดแปลง สร้างจากผลรวมของวงรีสิบวงตาม Shepp และ Logan"""
    g = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(g, -g)
    img = np.zeros((n, n))
    for A, a, b, x0, y0, ph in SL:
        t = np.deg2rad(ph)
        u = (X - x0) * np.cos(t) + (Y - y0) * np.sin(t)
        v = -(X - x0) * np.sin(t) + (Y - y0) * np.cos(t)
        img[(u / a) ** 2 + (v / b) ** 2 <= 1] += A
    return np.clip(img, 0, 1)

def textured_image(n=256, seed=2):
    """ภาพคล้ายภาพธรรมชาติ: สนามสุ่มจำกัดแถบความถี่ (สเปกตรัม 1/f) รวมกับวัตถุเชิงเรขาคณิต"""
    rng = np.random.default_rng(seed)
    q = np.fft.fftfreq(n); Q = np.sqrt(q[:, None] ** 2 + q[None, :] ** 2); Q[0, 0] = 1e-6
    fld = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((n, n))) / (Q ** 1.6)))
    fld = (fld - fld.min()) / (fld.max() - fld.min())
    g = np.linspace(0, 1, n); X, Y = np.meshgrid(g, g)
    img = 0.25 + 0.55 * fld
    img[(X - .70) ** 2 + (Y - .68) ** 2 < .018] = 0.95
    img[(np.abs(X - .28) < .14) & (np.abs(Y - .25) < .10)] = 0.08
    return np.clip(img, 0, 1)

def geometric_image(n=256, rng=None):
    """ภาพเชิงเรขาคณิตที่มีทั้งบริเวณเรียบ ขอบคม และลวดลายเป็นคาบ"""
    rng = rng or np.random.default_rng(0)
    g = np.linspace(0, 1, n)
    X, Y = np.meshgrid(g, g)
    img = 0.25 + 0.3 * X * (1 - Y)                                   # ไล่ระดับอย่างเรียบ
    img[(X - .30) ** 2 + (Y - .32) ** 2 < .028] = 0.95               # จานสว่าง
    img[(np.abs(X - .72) < .16) & (np.abs(Y - .30) < .12)] = 0.10    # สี่เหลี่ยมมืด
    band = (np.abs(X - .5) < .34) & (np.abs(Y - .74) < .13)
    img[band] = 0.5 + 0.35 * np.sign(np.sin(2 * np.pi * 14 * X[band]))  # ลวดลายเป็นคาบ
    tri = (Y - .52 > 1.6 * (X - .12)) & (Y - .52 > -1.6 * (X - .40)) & (Y < .70)
    img[tri] = 0.78
    return np.clip(img, 0, 1)

# ============================================================
# การแปลงเวฟเล็ตเชิงตั้งฉาก (db4, โหมดคาบ) ในรูปตัวดำเนินการคู่กัน
# ============================================================
WAVE, LEVEL, MODE = "db4", 4, "periodization"

def wav(img):
    c = pywt.wavedec2(img, WAVE, mode=MODE, level=LEVEL)
    arr, sl = pywt.coeffs_to_array(c)
    return arr, sl

def iwav(arr, sl):
    return pywt.waverec2(pywt.array_to_coeffs(arr, sl, output_format="wavedec2"), WAVE, mode=MODE)

def psnr(a, b, peak=1.0):
    return 10 * np.log10(peak ** 2 / np.mean((a - b) ** 2))

def nerr(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)

def soft(z, t):
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)

# ============================================================
# ตัวดำเนินการของ MRI: F เป็นการแปลงฟูเรียร์เชิงเอกภาค, R เป็นหน้ากากการชักตัวอย่าง
# ============================================================
def F(x):
    return np.fft.fft2(x, norm="ortho")

def Ft(k):
    return np.real(np.fft.ifft2(k, norm="ortho"))

def mask_uniform(n, R):
    """ชักตัวอย่างแถวแบบระยะเท่ากัน (ทุก ๆ R แถว เริ่มที่ความถี่ศูนย์) โดยไม่เพิ่มแถบกลาง"""
    M = np.zeros((n, n), bool); M[::R, :] = True
    return M

def mask_random(n, R, rng, center=0.0):
    """ชักตัวอย่างแถวแบบสุ่มเอกรูป โดยสงวนสัดส่วน center ของงบประมาณให้แถบความถี่ต่ำ"""
    q = np.fft.fftfreq(n)
    nkeep = int(round(n / R)); nc = int(round(center * nkeep))
    order = np.argsort(np.abs(q))
    M = np.zeros((n, n), bool); chosen = list(order[:nc])
    rest = [i for i in range(n) if i not in set(chosen)]
    chosen += list(rng.choice(rest, nkeep - nc, replace=False))
    M[np.array(chosen), :] = True
    return M

def mask_vardens(n, R, rng, power=3.0, center=0.15):
    """ชักตัวอย่างแถวแบบความหนาแน่นแปรผัน p(q) ∝ (1-|q|/q_max)^power"""
    q = np.abs(np.fft.fftfreq(n)); q = q / q.max()
    nkeep = int(round(n / R)); nc = int(round(center * nkeep))
    order = np.argsort(q)
    chosen = list(order[:nc]); rest = np.array([i for i in range(n) if i not in set(chosen)])
    p = (1 - q[rest]) ** power; p = p / p.sum()
    chosen += list(rng.choice(rest, nkeep - nc, replace=False, p=p))
    M = np.zeros((n, n), bool); M[np.array(chosen), :] = True
    return M

# ============================================================
# ตัวดำเนินการเกรเดียนต์เชิงคาบและตัวแก้ปัญหา ADMM ของการสร้างภาพ MRI คืน
# ============================================================
def grad(x):
    return np.stack([np.roll(x, -1, 0) - x, np.roll(x, -1, 1) - x])

def divg(g):
    """ตัวดำเนินการคู่กันของ grad (เครื่องหมายลบของไดเวอร์เจนซ์): <grad x, g> = <x, divg g>"""
    return (np.roll(g[0], 1, 0) - g[0]) + (np.roll(g[1], 1, 1) - g[1])

def lap_hat(n):
    """สเปกตรัมของ grad^T grad สำหรับผลต่างไปข้างหน้าเชิงคาบ"""
    k = np.fft.fftfreq(n) * 2 * np.pi
    d = np.abs(1 - np.exp(-1j * k)) ** 2
    return d[:, None] + d[None, :]

def block_soft(g, t):
    nrm = np.sqrt(g[0] ** 2 + g[1] ** 2)
    sc = np.maximum(nrm - t, 0.0) / np.maximum(nrm, 1e-12)
    return g * sc

def flip_freq(K):
    """K(-w) โดยดัชนีความถี่คิดแบบมอดุโล n ในทั้งสองแกน"""
    return np.roll(np.flip(K, (0, 1)), 1, (0, 1))

def herm(K):
    """ส่วนสมมาตรเฮอร์มิเชียน (K(w) + conj K(-w))/2 ซึ่งเป็นสเปกตรัมของส่วนจริงของ F^H K"""
    return 0.5 * (K + np.conj(flip_freq(K)))

def admm_mri(y, M, lam_w=0.0, lam_t=0.0, rho=0.5, iters=120, x_true=None, tol=None,
             max_iters=10000, adapt=True, eps_abs=1e-7, check=10):
    """แก้  min_{x จริง}  1/2 ||M.F x - y||^2 + lam_w ||W x||_1 + lam_t ||grad x||_{2,1}  ด้วย ADMM
       ขั้นปรับปรุง x เป็นการหารเชิงสมาชิกในปริภูมิเคตามบทตั้งในบท โดยใช้หน้ากากสมมาตร
       chi_bar = (chi(w) + chi(-w))/2 และข้อมูลสมมาตร y_bar = herm(R^T y) จึงได้ผลเฉลยค่าจริงที่แม่นตรง
       tol = None : วนครบ iters รอบด้วย rho คงที่ (ใช้สาธิตผลของการหยุดก่อนลู่เข้า)
       tol = eps_rel : วนจนลู่เข้าตามเกณฑ์ส่วนเหลือปฐมภูมิ/คู่กันของ Boyd และคณะ (2011, หัวข้อ 3.3.1)
                       พร้อมการปรับ rho แบบสมดุลส่วนเหลือ (หัวข้อ 3.4.1) ในช่วง 1,000 รอบแรก แล้วตรึง rho
       คืนค่า (ภาพที่ตัดช่วง [0,1], ประวัติ PSNR) และเก็บจำนวนรอบที่ใช้ไว้ใน admm_mri.last_iters"""
    n = M.shape[0]
    L = lap_hat(n)
    Mf = M.astype(float)
    chib = 0.5 * (Mf + flip_freq(Mf))                                 # หน้ากากสมมาตร (ค่า 0, 1/2, 1)
    ybar = herm(np.where(M, y, 0).astype(complex))                   # ข้อมูลเคสเปซสมมาตร
    x = Ft(np.where(M, y, 0))
    arr, sl = wav(x)
    zw = np.zeros_like(arr); uw = np.zeros_like(arr)
    zt = np.zeros((2, n, n)); ut = np.zeros((2, n, n))
    uw_on, ut_on = lam_w > 0, lam_t > 0
    hist = []
    T = iters if tol is None else max_iters
    p_dim = (arr.size if uw_on else 0) + (2 * n * n if ut_on else 0)
    def make_den(r):
        d = chib + r * (uw_on + ut_on * L)
        return np.where(d < 1e-12, 1e-12, d)
    den = make_den(rho)
    t_ = 0; converged = False; admm_mri.last_res = None
    for t_ in range(T):
        rhs = ybar.copy()
        if uw_on:
            rhs = rhs + rho * F(iwav(zw - uw, sl))
        if ut_on:
            rhs = rhs + rho * F(divg(zt - ut))
        x = Ft(rhs / den)
        if uw_on:
            arr, sl = wav(x); zw_old = zw
            zw = soft(arr + uw, lam_w / rho); uw = uw + arr - zw
        if ut_on:
            gx = grad(x); zt_old = zt
            zt = block_soft(gx + ut, lam_t / rho); ut = ut + gx - zt
        if x_true is not None:
            hist.append(psnr(np.clip(x, 0, 1), x_true))
        if tol is not None and (t_ + 1) % check == 0 and (uw_on or ut_on):
            # ส่วนเหลือคู่กัน s = rho A^T (z - z_old) โดย A^T = [Psi, grad^T] จึงเป็นนอร์มของ "ผลรวม" ของสองพจน์
            # (แก้ตามรายงานตรวจ C01: เดิมใช้ sqrt(||dz_w||^2 + ||grad^T dz_t||^2) ซึ่งขาดพจน์ไขว้)
            rp2 = ax2 = z2 = 0.0; aty = np.zeros((n, n)); atdz = np.zeros((n, n))
            if uw_on:
                rp2 += np.sum((arr - zw) ** 2); atdz = atdz + iwav(zw - zw_old, sl)
                ax2 += np.sum(arr ** 2); z2 += np.sum(zw ** 2); aty = aty + iwav(uw, sl)
            if ut_on:
                rp2 += np.sum((gx - zt) ** 2); atdz = atdz + divg(zt - zt_old)
                ax2 += np.sum(gx ** 2); z2 += np.sum(zt ** 2); aty = aty + divg(ut)
            rp, sd = np.sqrt(rp2), rho * np.linalg.norm(atdz)
            eps_pri = np.sqrt(p_dim) * eps_abs + tol * np.sqrt(max(ax2, z2))
            eps_dual = np.sqrt(n * n) * eps_abs + tol * rho * np.linalg.norm(aty)
            admm_mri.last_res = (rp, sd, eps_pri, eps_dual)
            if rp <= eps_pri and sd <= eps_dual:
                converged = True
                break
            if adapt and t_ < 1000:                                   # การปรับ rho แบบสมดุลส่วนเหลือ
                if rp > 10 * sd:
                    rho *= 2; uw = uw / 2; ut = ut / 2; den = make_den(rho)
                elif sd > 10 * rp:
                    rho /= 2; uw = uw * 2; ut = ut * 2; den = make_den(rho)
    admm_mri.last_iters = t_ + 1
    admm_mri.last_converged = converged if tol is not None else None   # None = จำนวนรอบคงที่ (ไม่ได้ทดสอบการลู่เข้า)
    return np.clip(x, 0, 1), hist

# ============================================================
# fig06-coherence : ความสอดคล้องร่วมระหว่างฐานฟูเรียร์กับฐานเวฟเล็ต
#   (ก) |<f_w, psi_j>| เรียงตามสเกล (ข) ความสอดคล้องร่วมเฉพาะที่ต่อความถี่ และความหนาแน่นที่แนะนำ
# ============================================================
def wave_basis_1d(n, wave=WAVE, level=LEVEL):
    """เมทริกซ์ Psi ที่สดมภ์เป็นอะตอมเวฟเล็ต พร้อมดัชนีสเกลของแต่ละสดมภ์"""
    c0 = pywt.wavedec(np.zeros(n), wave, mode=MODE, level=level)
    arr0, sl = pywt.coeffs_to_array(c0)
    Psi = np.zeros((n, n)); lab = np.zeros(n, int)
    bounds = [len(c0[0])]
    for d in c0[1:]:
        bounds.append(bounds[-1] + len(d))
    for j in range(n):
        e = np.zeros(n); e[j] = 1.0
        Psi[:, j] = pywt.waverec(pywt.array_to_coeffs(e, sl, output_format="wavedec"), wave, mode=MODE)
        lab[j] = int(np.searchsorted(bounds, j, side="right"))
    return Psi, lab

if want("coherence"):
    n = 256
    Psi, lab = wave_basis_1d(n)
    Fm = np.fft.fft(np.eye(n), axis=0, norm="ortho")                 # แถวที่ w คือ f_w^H
    G = np.abs(np.conj(Fm).T @ Psi)                                  # |<f_w, psi_j>|
    q = np.fft.fftfreq(n); idx = np.argsort(q)                       # เรียงความถี่จากลบไปบวก
    Gs = G[idx, :]
    mu_loc = np.sqrt(n) * Gs.max(axis=1)                             # ความสอดคล้องร่วมเฉพาะที่
    print(f"coherence: global mu(F,Psi) = {np.sqrt(n)*G.max():.3f} (n = {n}, {WAVE}, {LEVEL} levels);"
          f" max over |q| < 0.05 = {mu_loc[np.abs(q[idx])<0.05].max():.3f};"
          f" max over |q| > 0.25 = {mu_loc[np.abs(q[idx])>0.25].max():.3f}")
    nlev = lab.max() + 1
    permu = np.zeros((len(mu_loc), nlev))
    for s_ in range(nlev):
        permu[:, s_] = np.sqrt(n) * Gs[:, lab == s_].max(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.5))
    axes[0].imshow(Gs ** 0.5, aspect="auto", cmap="magma",
                   extent=[0, n, q[idx].min(), q[idx].max()])
    axes[0].set_xlabel("wavelet atom index (coarse to fine)"); axes[0].set_ylabel(r"normalized frequency $\omega/n$")
    axes[0].set_title("(a)", fontsize=9, loc="left")
    for s_, st in enumerate([("-", INK), ("--", ACC), (":", GRN), ("-.", ORG), ("-", RED)][:nlev]):
        axes[1].plot(q[idx], permu[:, s_], st[0], color=st[1], lw=1.0,
                     label=("scaling" if s_ == 0 else f"level {nlev-s_}"))
    axes[1].set_xlabel(r"normalized frequency $\omega/n$"); axes[1].set_ylabel(r"local coherence")
    axes[1].set_title("(b)", fontsize=9, loc="left"); axes[1].legend(frameon=False, fontsize=6.5, ncol=2)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig06-coherence")
    for s_ in range(nlev):
        w_ = np.abs(q[idx][np.argmax(permu[:, s_])])
        print(f"   {'scaling' if s_==0 else f'level {nlev-s_}':9s}: peak local coherence {permu[:, s_].max():.2f} at |q| = {w_:.3f},"
              f"  mean over |q| > 0.25 = {permu[np.abs(q[idx])>0.25, s_].mean():.2f}")

# ============================================================
# fig06-sampling : หน้ากากการชักตัวอย่างสามแบบ ฟังก์ชันกระจายจุด และ TPSF
# ============================================================
def psf_profile(M):
    """ฟังก์ชันกระจายจุดของหน้ากาก (ชักตัวอย่างตามแถว) ปรกติให้ค่ากลางเป็นหนึ่ง"""
    rows = M[:, 0].astype(float)
    p = np.real(np.fft.ifft(rows)); p = p / p[0]
    return np.fft.fftshift(p)

def tpsf_sidelobe(M, level, n=256, rng=None):
    """อัตราส่วนกลีบข้างสูงสุดของ TPSF; level = 0 คือฟังก์ชันสเกลระดับหยาบสุด
       level = l > 0 คืออะตอมรายละเอียดของระดับการแยกที่ LEVEL - l + 1 (level = LEVEL คือระดับละเอียดที่สุด)"""
    c = pywt.wavedec2(np.zeros((n, n)), WAVE, mode=MODE, level=LEVEL)
    arr, sl = pywt.coeffs_to_array(c)
    nc = len(pywt.coeffs_to_array(pywt.wavedec2(np.zeros((n, n)), WAVE, mode=MODE, level=LEVEL))[0])
    a = np.zeros_like(arr)
    if level == 0:
        a[n // 32, n // 32] = 1.0                                     # อะตอมสเกลหยาบ
    else:
        b = n // (2 ** (LEVEL - level + 1))
        a[b + b // 2, b // 2] = 1.0
    atom = iwav(a, sl); atom = atom / np.linalg.norm(atom)
    t, _ = wav(Ft(np.where(M, F(atom), 0)))
    peak = np.abs(t).max(); t2 = np.abs(t.copy()); i = np.unravel_index(np.argmax(t2), t2.shape); t2[i] = 0
    return t2.max() / peak, t2.std() / peak

if want("sampling"):
    n, R = 256, 4
    rng = np.random.default_rng(3)
    masks = [("equispaced", mask_uniform(n, R)), ("uniform random", mask_random(n, R, rng)),
             ("variable density", mask_vardens(n, R, rng))]
    fig, axes = plt.subplots(2, 3, figsize=(6.2, 3.4))
    print(f"sampling: n = {n}, acceleration R = {R} (phase-encode rows kept = {int(n/R)})")
    for j, (name, M) in enumerate(masks):
        axes[0, j].imshow(np.fft.fftshift(M), aspect="auto", cmap="gray_r")
        axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
        axes[0, j].set_title(f"({chr(97+j)}) {name}", fontsize=8, loc="left")
        p = psf_profile(M); c = len(p) // 2
        axes[1, j].plot(np.arange(len(p)) - c, np.abs(p), color=INK, lw=0.7)
        axes[1, j].set_ylim(0, 1.05); axes[1, j].set_xlabel("pixel")
        if j == 0:
            axes[1, j].set_ylabel("|PSF|")
        side = np.abs(np.delete(p, c)).max()
        s0, _ = tpsf_sidelobe(M, 0, n); s3, _ = tpsf_sidelobe(M, 3, n); s4, _ = tpsf_sidelobe(M, 4, n)
        print(f"   {name:17s}: peak PSF sidelobe {side:.3f};  TPSF sidelobe  scaling atom (level 4) {s0:.3f}"
              f"  detail atom, decomposition level 2 {s3:.3f}  finest detail atom (level 1) {s4:.3f}")
    fig.tight_layout(h_pad=0.6, w_pad=0.8)
    save(fig, "fig06-sampling")

# ============================================================
# fig06-mri : การสร้างภาพ MRI คืนจากข้อมูลเคสเปซที่ชักตัวอย่างต่ำ
#   lam ทุกค่าเลือกจากกริด {0, 1e-3, 3e-3, 1e-2, 3e-2} ที่ให้ PSNR สูงสุดของแต่ละภาพ
# ============================================================
SIG_K = 0.004                                                        # ส่วนเบี่ยงเบนมาตรฐานของสัญญาณรบกวนในเคสเปซ

def acquire(x, M, seed):
    r = np.random.default_rng(seed); k = F(x)
    e = (r.standard_normal(k.shape) + 1j * r.standard_normal(k.shape)) * SIG_K / np.sqrt(2)
    return np.where(M, k + e, 0)

GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]                                       # กริดของตัวคูณ (ค่าศูนย์หมายถึงไม่ใช้โทษนั้น)

TOL = 1e-6                                                            # เกณฑ์ลู่เข้า eps_rel ของ ADMM (Boyd และคณะ)
CONV_LOG = []                                                         # สถานะลู่เข้า (ผ่านเกณฑ์ส่วนเหลือ) ของทุกการแก้ปัญหา
ITER_LOG = []                                                         # จำนวนรอบที่ใช้จนลู่เข้าของทุกการแก้ปัญหา

def _step(v, down=True):
    """ค่าถัดไปบนกริดแบบ 1-3-10: 1e-4 -> 3e-5 -> 1e-5 (ลง) และ 3e-2 -> 1e-1 (ขึ้น)"""
    lead = round(v / 10 ** np.floor(np.log10(v)))
    if down:
        return v * 0.3 if lead == 1 else v / 3
    return v * 10 / 3 if lead == 3 else v * 3

def tune(y, M, x_true, kind, iters=None, tol=TOL, max_ext=4):
    """เลือกตัวคูณจากกริดที่ให้ PSNR สูงสุด (oracle tuning) คืนค่า (PSNR, ภาพ, lam_w, lam_t)
       ค่าเริ่มต้นแก้ทุกปัญหาจนลู่เข้า (tol) หากกำหนด iters จะใช้จำนวนรอบคงที่แทน
       หากค่าที่ดีที่สุดอยู่ที่ขอบของกริด จะขยายการค้นหาออกไปทีละขั้นตามแนวของตัวคูณนั้นจนได้ค่าภายใน"""
    cache = {}
    def run(lw, lt):
        key = (round(lw, 12), round(lt, 12))
        if key not in cache:
            if iters is None:
                r, _ = admm_mri(y, M, lam_w=lw, lam_t=lt, tol=tol)
                ITER_LOG.append(admm_mri.last_iters); CONV_LOG.append(bool(admm_mri.last_converged))
            else:
                r, _ = admm_mri(y, M, lam_w=lw, lam_t=lt, iters=iters)
            cache[key] = (psnr(r, x_true), r, lw, lt)
        return cache[key]
    uw, ut = kind in ("wavelet", "TV + wavelet"), kind in ("TV", "TV + wavelet")
    gw = GRID if uw else [0.0]
    gt = GRID if ut else [0.0]
    best = max((run(lw, lt) for lw in gw for lt in gt), key=lambda b: b[0])
    for axis, on in ((2, uw), (3, ut)):                               # ขยายการค้นหาเมื่อค่าที่ดีที่สุดอยู่ที่ขอบ
        if not on:
            continue
        for down in (True, False):
            for _ in range(max_ext):
                tested = sorted({k[axis - 2] for k in cache})
                edge = tested[0] if down else tested[-1]
                if abs(best[axis] - edge) > 1e-15:
                    break
                nxt = _step(best[axis], down)
                cand = run(nxt, best[3]) if axis == 2 else run(best[2], nxt)
                if cand[0] > best[0]:
                    best = cand
                else:
                    break
    for lam, axis in ((best[2], 2), (best[3], 3)):
        if lam > 0:
            tested = sorted({k[axis - 2] for k in cache})
            if lam in (tested[0], tested[-1]):
                print(f"   [note] tuned multiplier {lam:.0e} still lies on the edge of the searched values")
    return best

if want("mri"):
    n, R = 256, 4
    x0 = phantom(n); xt = textured_image(n)
    M = mask_vardens(n, R, np.random.default_rng(3))
    Mu = mask_random(n, R, np.random.default_rng(3))
    Me = mask_uniform(n, R)
    y = acquire(x0, M, 5)
    rec = {"zero-filled": (psnr(np.clip(Ft(y), 0, 1), x0), np.clip(Ft(y), 0, 1), 0, 0)}
    for kind in ["wavelet", "TV", "TV + wavelet"]:
        rec[kind] = tune(y, M, x0, kind)
    _, hist_tv = admm_mri(y, M, lam_t=rec["TV"][3], iters=200, x_true=x0)       # rho = 0.5 คงที่ 200 รอบ (การหยุดก่อนลู่เข้า)
    r_200 = {k_: psnr(admm_mri(y, M, lam_w=rec[k_][2], lam_t=rec[k_][3], iters=200)[0], x0)
             for k_ in ["wavelet", "TV", "TV + wavelet"]}
    fig, axes = plt.subplots(2, 3, figsize=(6.2, 4.35))
    show(axes[0, 0], x0, "(a) phantom")
    show(axes[0, 1], np.clip(Ft(acquire(x0, Me, 5)), 0, 1), "(b) equispaced, zero-filled")
    show(axes[0, 2], rec["zero-filled"][1], "(c) variable density, zero-filled")
    show(axes[1, 0], rec["wavelet"][1], "(d) wavelet")
    show(axes[1, 1], rec["TV"][1], "(e) TV")
    show(axes[1, 2], rec["TV + wavelet"][1], "(f) TV + wavelet")
    fig.tight_layout(h_pad=1.1, w_pad=0.3)
    save(fig, "fig06-mri")
    print(f"mri: phantom {n}x{n}, n/m = {R}, variable density, k-space noise sigma = {SIG_K}; ADMM run to convergence"
          f" (Boyd primal/dual residual criterion, eps_rel = {TOL:.0e}, eps_abs = 1e-7, residual balancing of rho for the first 1000 iterations,"
          f" at most 10000 iterations); multipliers tuned on the grid {GRID}")
    for k_, v in rec.items():
        extra = f"   (fixed rho = 0.5, 200 iterations, same multipliers: {r_200[k_]:5.2f} dB)" if k_ in r_200 else ""
        print(f"   {k_:13s} PSNR {v[0]:5.2f} dB   normalized error {nerr(v[1], x0):.4f}   lam_w {v[2]:.0e}  lam_t {v[3]:.0e}{extra}")
    print(f"   TV with fixed rho = 0.5: PSNR after 10 / 50 / 200 iterations: {hist_tv[9]:.2f} / {hist_tv[49]:.2f} / {hist_tv[-1]:.2f} dB")
    print(f"   iterations to convergence over the tuning grid so far: median {int(np.median(ITER_LOG))}, max {max(ITER_LOG)}; converged {sum(CONV_LOG)}/{len(CONV_LOG)} solves, iteration cap reached {len(CONV_LOG) - sum(CONV_LOG)}")
    print("mri: same budget, three sampling patterns (TV, lambda_t tuned on the grid for each pattern)")
    for nm, MM in [("equispaced", Me), ("uniform random", Mu), ("variable density", M)]:
        yy = acquire(x0, MM, 5)
        zf = np.clip(Ft(yy), 0, 1); b = tune(yy, MM, x0, "TV")
        dc = bool(MM[0, 0])
        print(f"   {nm:17s} zero-filled {psnr(zf, x0):5.2f} dB   TV {b[0]:5.2f} dB (lam_t {b[3]:.0e})   DC row sampled: {dc}")
    print("mri: textured image, n/m = 4, variable density")
    yt = acquire(xt, M, 7)
    print(f"   zero-filled   PSNR {psnr(np.clip(Ft(yt), 0, 1), xt):5.2f} dB")
    for kind in ["wavelet", "TV", "TV + wavelet"]:
        b = tune(yt, M, xt, kind)
        print(f"   {kind:13s} PSNR {b[0]:5.2f} dB   lam_w {b[2]:.0e}  lam_t {b[3]:.0e}")
    print(f"   iterations to convergence over all tuning runs: median {int(np.median(ITER_LOG))}, max {max(ITER_LOG)}, runs {len(ITER_LOG)}; converged {sum(CONV_LOG)}/{len(CONV_LOG)}, iteration cap reached {len(CONV_LOG) - sum(CONV_LOG)}")

if want("mriacc"):                                                    # แยกเป็นส่วนอิสระเพื่อรันขนานกับส่วน mri
    n = 256
    x0 = phantom(n); xt = textured_image(n)
    print("mri: PSNR (dB) versus acceleration n/m (TV, lambda_t tuned on the grid at every point, ADMM run to convergence)")
    accs = [2, 3, 4, 6, 8, 12]
    curves = {}
    for nm, img in [("phantom", x0), ("textured", xt)]:
        zfs, css = [], []
        for Rk in accs:
            Mk = mask_vardens(n, Rk, np.random.default_rng(3)); yk = acquire(img, Mk, 11)
            zfs.append(psnr(np.clip(Ft(yk), 0, 1), img))
            css.append(tune(yk, Mk, img, "TV")[0])
        curves[nm] = (zfs, css)
        print(f"   {nm:9s} n/m =      " + " ".join(f"{v:5d}" for v in accs))
        print(f"   {nm:9s} zero-filled" + " ".join(f"{v:5.1f}" for v in zfs))
        print(f"   {nm:9s} TV         " + " ".join(f"{v:5.1f}" for v in css))
    print(f"   iterations to convergence over all tuning runs: median {int(np.median(ITER_LOG))}, max {max(ITER_LOG)}, runs {len(ITER_LOG)}; converged {sum(CONV_LOG)}/{len(CONV_LOG)}, iteration cap reached {len(CONV_LOG) - sum(CONV_LOG)}")
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    for nm, st in [("phantom", ("o-", ACC)), ("textured", ("s-", RED))]:
        zfs, css = curves[nm]
        ax.plot(accs, css, st[0], color=st[1], ms=3, label=f"{nm}, TV")
        ax.plot(accs, zfs, "--", color=st[1], ms=3, alpha=0.5, label=f"{nm}, zero-filled")
    ax.set_xlabel("acceleration $n/m$"); ax.set_ylabel("PSNR (dB)")
    ax.legend(frameon=False, fontsize=6.3, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    save(fig, "fig06-acceleration")

# ============================================================
# fig06-denoise : การกำจัดสัญญาณรบกวนด้วยขีดเริ่มเปลี่ยนเวฟเล็ต ความแปรผันรวม และพจนานุกรมที่เรียนรู้
# ============================================================
def wave_denoise(z, th, spins=1):
    """ขีดเริ่มเปลี่ยนแบบอ่อนในโดเมนเวฟเล็ต; spins > 1 คือการเฉลี่ยผลของการเลื่อนวัฏจักร"""
    n = z.shape[0]; acc = np.zeros_like(z); cnt = 0
    sh = np.arange(spins)                                             # การเลื่อนทีละหนึ่งจุดภาพ
    for a in sh:
        for b in sh:
            zz = np.roll(np.roll(z, a, 0), b, 1)
            arr, sl = wav(zz)
            nc = n // (2 ** LEVEL)
            d = soft(arr, th); d[:nc, :nc] = arr[:nc, :nc]            # ไม่แตะสัมประสิทธิ์สเกลหยาบ
            acc += np.roll(np.roll(iwav(d, sl), -a, 0), -b, 1); cnt += 1
    return acc / cnt

def tv_denoise(z, lam, iters=200, rho=1.0):
    """ROF: min 1/2||x - z||^2 + lam ||grad x||_{2,1} ด้วย ADMM (ตัวดำเนินการวัดเป็นเอกลักษณ์)"""
    n = z.shape[0]; L = lap_hat(n)
    x = z.copy(); zt = np.zeros((2, n, n)); ut = np.zeros((2, n, n))
    den = 1.0 + rho * L
    for _ in range(iters):
        x = Ft((F(z) + rho * F(divg(zt - ut))) / den)
        g = grad(x); zt = block_soft(g + ut, lam / rho); ut = ut + g - zt
    return x

def patches(img, w=8, step=2):
    n = img.shape[0]; idx = range(0, n - w + 1, step)
    P = np.array([img[i:i + w, j:j + w].ravel() for i in idx for j in idx]).T
    return P, list(idx)

def unpatch(P, idx, n, w=8):
    acc = np.zeros((n, n)); cnt = np.zeros((n, n)); t = 0
    for i in idx:
        for j in idx:
            acc[i:i + w, j:j + w] += P[:, t].reshape(w, w); cnt[i:i + w, j:j + w] += 1; t += 1
    return acc / np.maximum(cnt, 1)

def dct_dict(w=8, k=12):
    """พจนานุกรม DCT เกินบริบูรณ์ขนาด w^2 x k^2"""
    g = np.arange(w)[:, None] * (np.arange(k)[None, :] + 0.0)
    B = np.cos(np.pi / k * (np.arange(w)[:, None] + 0.5) * np.arange(k)[None, :])
    B = B - B.mean(axis=0, keepdims=True) * (np.arange(k) > 0)
    B = B / np.linalg.norm(B, axis=0, keepdims=True)
    D = np.kron(B, B)
    return D / np.linalg.norm(D, axis=0, keepdims=True)

def omp_batch_err(D, Y, eps, kmax=6):
    """OMP แบบกำหนดเกณฑ์ความคลาดเคลื่อน ใช้กับทุกแพตช์พร้อมกัน"""
    n = D.shape[1]; X = np.zeros((n, Y.shape[1]))
    for t in range(Y.shape[1]):
        y_ = Y[:, t]; r = y_.copy(); S = []
        for _ in range(kmax):
            if np.linalg.norm(r) <= eps:
                break
            S.append(int(np.argmax(np.abs(D.T @ r))))
            c = np.linalg.lstsq(D[:, S], y_, rcond=None)[0]; r = y_ - D[:, S] @ c
        if S:
            X[S, t] = c
    return X

def ksvd(Y, D, iters=8, eps=0.0, kmax=6, rng=None):
    rng = rng or np.random.default_rng(0)
    D = D / np.linalg.norm(D, axis=0, keepdims=True)
    for _ in range(iters):
        X = omp_batch_err(D, Y, eps, kmax)
        for j in range(D.shape[1]):
            w = np.nonzero(X[j, :])[0]
            if len(w) == 0:
                D[:, j] = Y[:, rng.integers(Y.shape[1])]; D[:, j] /= np.linalg.norm(D[:, j]) + 1e-12
                continue
            E = Y[:, w] - D @ X[:, w] + np.outer(D[:, j], X[j, w])
            U, S_, Vt = np.linalg.svd(E, full_matrices=False)
            D[:, j] = U[:, 0]; X[j, w] = S_[0] * Vt[0, :]
    return D

if want("denoise"):
    n = 128
    x0 = textured_image(256)[::2, ::2]
    rng = np.random.default_rng(4)
    sigma = 0.10
    z = x0 + sigma * rng.standard_normal(x0.shape)
    th_u = sigma * np.sqrt(2 * np.log(n * n))
    ths = np.linspace(0.2, 4.0, 25) * sigma
    mse = [np.mean((np.clip(wave_denoise(z, t), 0, 1) - x0) ** 2) for t in ths]
    t_opt = ths[int(np.argmin(mse))]
    print(f"denoise: textured image {n}x{n}, sigma = {sigma}, noisy PSNR {psnr(np.clip(z,0,1), x0):.2f} dB")
    print(f"   universal threshold sigma*sqrt(2 log n^2) = {th_u:.4f} ({th_u/sigma:.2f} sigma);"
          f" empirically best threshold {t_opt:.4f} ({t_opt/sigma:.2f} sigma)")
    out = {"noisy": np.clip(z, 0, 1)}
    out["wavelet, universal"] = np.clip(wave_denoise(z, th_u), 0, 1)
    out["wavelet, best"] = np.clip(wave_denoise(z, t_opt), 0, 1)
    out["cycle spinning"] = np.clip(wave_denoise(z, t_opt, spins=4), 0, 1)
    out["cycle spinning, universal"] = np.clip(wave_denoise(z, th_u, spins=4), 0, 1)
    lams = [0.03, 0.05, 0.08, 0.12, 0.18]
    tvs = [np.clip(tv_denoise(z, l), 0, 1) for l in lams]
    jbest = int(np.argmax([psnr(v, x0) for v in tvs]))
    out["TV"] = tvs[jbest]
    P, idx = patches(z, 8, 2)
    mu = P.mean(axis=0, keepdims=True); Pc = P - mu
    D0 = dct_dict(8, 12)
    t0 = time.time()
    sub = np.random.default_rng(1).choice(Pc.shape[1], 3000, replace=False)
    best = None
    for C in [0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5]:                             # ปรับเกณฑ์ความคลาดเคลื่อนของ OMP บนกริด
        e_ = C * sigma * 8
        Xc_ = omp_batch_err(D0, Pc, e_, 6)
        v = psnr(np.clip(unpatch(D0 @ Xc_ + mu, idx, n), 0, 1), x0)
        print(f"   OMP error bound C = {C:.2f} ({e_:.3f}): fixed DCT dictionary PSNR {v:.2f} dB,"
              f" mean sparsity {np.mean(np.sum(Xc_ != 0, 0)):.2f}")
        if best is None or v > best[0]:
            best = (v, e_, Xc_)
    eps = best[1]; Xc = best[2]
    D = ksvd(Pc[:, sub], D0.copy(), iters=8, eps=eps, kmax=6)
    out["DCT dictionary"] = np.clip(unpatch(D0 @ Xc + mu, idx, n), 0, 1)
    Xl = omp_batch_err(D, Pc, eps, 6);  out["learned dictionary"] = np.clip(unpatch(D @ Xl + mu, idx, n), 0, 1)
    print(f"   patch dictionary: {D0.shape[0]}x{D0.shape[1]} atoms, {Pc.shape[1]} patches (8x8, step 2),"
          f" K-SVD on 3000 patches, 8 iterations, {time.time()-t0:.0f} s, OMP error bound {eps:.3f};"
          f" mean sparsity fixed {np.mean(np.sum(Xc!=0,0)):.2f}, learned {np.mean(np.sum(Xl!=0,0)):.2f}")
    print(f"   TV lambda grid {lams}, best = {lams[jbest]}")
    for k_, v in out.items():
        print(f"   {k_:21s} PSNR {psnr(v, x0):5.2f} dB")
    fig = plt.figure(figsize=(6.2, 4.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    order = ["noisy", "wavelet, universal", "cycle spinning", "TV", "DCT dictionary", "learned dictionary"]
    for j, k_ in enumerate(order):
        ax = fig.add_subplot(gs[j // 3, j % 3])
        show(ax, out[k_], f"({chr(97+j)}) {k_}, {psnr(out[k_], x0):.1f} dB")
    fig.tight_layout(h_pad=1.1, w_pad=0.3)
    save(fig, "fig06-denoise")
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    ax.plot(np.array(ths) / sigma, np.sqrt(mse), color=INK)
    ax.axvline(th_u / sigma, color=ACC, ls="--", lw=1.0); ax.axvline(t_opt / sigma, color=RED, ls=":", lw=1.0)
    ax.annotate("universal", (th_u / sigma, max(np.sqrt(mse)) * 0.93), color=ACC, fontsize=7, ha="right",
                textcoords="offset points", xytext=(-4, 0))
    ax.annotate("best", (t_opt / sigma, max(np.sqrt(mse)) * 0.93), color=RED, fontsize=7, ha="left",
                textcoords="offset points", xytext=(4, 0))
    ax.set_xlabel(r"threshold $/\sigma$"); ax.set_ylabel("RMSE")
    fig.tight_layout()
    save(fig, "fig06-threshold")

# ============================================================
# fig06-irls : กรณีศึกษาของผู้แต่ง --- พรมแดนความแม่นยำกับเวลาในการกู้คืนภาพ
#   ขนาดภาพเพิ่มขึ้นเป็นลำดับ วัดทั้งความคลาดเคลื่อนและเวลา เพื่อประมาณเลขชี้กำลังของต้นทุน
# ============================================================
def wav_l(img, level):
    c = pywt.wavedec2(img, WAVE, mode=MODE, level=level)
    return pywt.coeffs_to_array(c)

def iwav_l(arr, sl):
    return pywt.waverec2(pywt.array_to_coeffs(arr, sl, output_format="wavedec2"), WAVE, mode=MODE)

def op_norm(A, At, n, iters=30, seed=0):
    """ค่าเอกฐานมากที่สุดของตัวดำเนินการด้วยวิธียกกำลัง"""
    v = np.random.default_rng(seed).standard_normal(n); v /= np.linalg.norm(v)
    nw = 0.0
    for _ in range(iters):
        w = At(A(v)); nw = np.linalg.norm(w)
        if nw == 0:
            return 0.0
        v = w / nw
    return np.sqrt(nw)

def fista_l1(A, At, y, n, lam, iters, Lc=1.0):
    """FISTA สำหรับ min 1/2||A c - y||^2 + lam||c||_1 ในโดเมนสัมประสิทธิ์ (Lc = sigma_1(A)^2)"""
    c = np.zeros(n); v = c.copy(); t = 1.0
    for _ in range(iters):
        g = At(A(v) - y)
        c_new = soft(v - g / Lc, lam / Lc)
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        v = c_new + (t - 1) / t_new * (c_new - c)
        c, t = c_new, t_new
    return c

def fista_gap(A, At, y, n, lam, Lc, tol=1e-6, max_iters=100000, groups=None, check=10):
    """FISTA ที่แก้ min 1/2||A c - y||^2 + lam*Omega(c) จนลู่เข้า
       Omega = ||c||_1 (groups=None) หรือ ||C||_{2,1} เมื่อ c.reshape(groups) มีกลุ่มเป็นแถว
       - เริ่มโมเมนตัมใหม่ตามเงื่อนไขเกรเดียนต์ของ O'Donoghue และ Candes (2015) เพื่อเร่งการลู่เข้า
       - หยุดเมื่อช่องว่างคู่กัน P(c) - D(u) <= tol * P(c) โดย u = r * min(1, lam / ||A^T r||_*), r = y - A c
         ซึ่งเป็นจุดที่เป็นไปได้ของปัญหาคู่กัน max <y,u> - 1/2||u||^2  s.t. ||A^T u||_* <= lam
       จำนวนรอบที่ใช้เก็บไว้ใน fista_gap.last_iters และช่องว่างสัมพัทธ์ใน fista_gap.last_gap"""
    if groups is None:
        prox = lambda z, t: soft(z, t)
        pen = lambda c: np.sum(np.abs(c))
        dnorm = lambda z: np.max(np.abs(z))
    else:
        def prox(z, t):
            Z = z.reshape(groups); nr = np.linalg.norm(Z, axis=1, keepdims=True)
            return (Z * np.maximum(nr - t, 0) / np.maximum(nr, 1e-300)).ravel()
        pen = lambda c: np.sum(np.linalg.norm(c.reshape(groups), axis=1))
        dnorm = lambda z: np.max(np.linalg.norm(z.reshape(groups), axis=1))
    c = np.zeros(n); v = c.copy(); t = 1.0; gap = np.inf
    for it in range(1, max_iters + 1):
        c_new = prox(v - At(A(v) - y) / Lc, lam / Lc)
        if np.dot(v - c_new, c_new - c) > 0:                          # เงื่อนไขการเริ่มโมเมนตัมใหม่
            t = 1.0
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        v = c_new + (t - 1) / t_new * (c_new - c)
        c, t = c_new, t_new
        if it % check == 0:
            r = y - A(c)
            P = 0.5 * np.dot(r, r) + lam * pen(c)
            u = r * min(1.0, lam / max(dnorm(At(r)), 1e-300))
            gap = (P - (np.dot(y, u) - 0.5 * np.dot(u, u))) / P
            if gap <= tol:
                break
    fista_gap.last_iters, fista_gap.last_gap = it, gap
    return c

def _penalty(groups):
    """โทษ ล1 (groups=None) หรือ ล2,1 ที่กลุ่มคือแถวของ c.reshape(groups): คืน (prox, ค่าโทษ, นอร์มคู่กัน)"""
    if groups is None:
        return (lambda z, t: soft(z, t)), (lambda c: np.sum(np.abs(c))), (lambda z: np.max(np.abs(z)))
    def prox(z, t):
        Z = z.reshape(groups); nr = np.linalg.norm(Z, axis=1, keepdims=True)
        return (Z * np.maximum(nr - t, 0) / np.maximum(nr, 1e-300)).ravel()
    return (prox, (lambda c: np.sum(np.linalg.norm(c.reshape(groups), axis=1))),
            (lambda z: np.max(np.linalg.norm(z.reshape(groups), axis=1))))

def l1_gap(A, At, y, c, lam, groups=None):
    """ช่องว่างคู่กันสัมพัทธ์ของ LASSO (หรือ LASSO เชิงกลุ่ม) ที่จุด c โดยจุดคู่กันได้จากการปรับมาตราส่วนเศษเหลือ"""
    _, pen, dnorm = _penalty(groups)
    r = y - A(c)
    P = 0.5 * np.dot(r, r) + lam * pen(c)
    u = r * min(1.0, lam / max(dnorm(At(r)), 1e-300))
    return (P - (np.dot(y, u) - 0.5 * np.dot(u, u))) / P

def admm_l1(A, At, y, n, lam, solve, rho=None, tol=1e-6, max_iters=300000, balance=5000, check=10, groups=None, z0=None):
    """ADMM สำหรับ min 1/2||A c - y||^2 + lam||z||_1  s.t. c = z
       solve(v, rho) คืนค่า (rho I + A A^T)^{-1} v จึงได้ขั้น c จากเอกลักษณ์ของวูดเบอรี
       (A^T A + rho I)^{-1} q = (q - A^T solve(A q, rho)) / rho  ซึ่งถูกเมื่อ A A^T มีโครงสร้างที่ผกผันง่าย
       ปรับ rho ด้วยการถ่วงดุลเศษเหลือในช่วง balance รอบแรก และหยุดเมื่อช่องว่างคู่กันสัมพัทธ์ที่ z ไม่เกิน tol
       groups ไม่เป็น None หมายถึงโทษ ล2,1 แทน ล1"""
    prox = _penalty(groups)[0]
    rho = 10.0 * lam if rho is None else rho
    Aty = At(y); z = np.zeros(n) if z0 is None else z0.copy(); u = np.zeros(n); gap = np.inf   # z0: จุดเริ่มต้น (warm start)
    for it in range(1, max_iters + 1):
        q = Aty + rho * (z - u)
        c = (q - At(solve(A(q), rho))) / rho
        z_old = z
        z = prox(c + u, lam / rho)
        u = u + c - z
        if it % check == 0:
            if it <= balance:
                rp = np.linalg.norm(c - z); sd = rho * np.linalg.norm(z - z_old)
                if rp > 10 * sd:
                    rho *= 2.0; u /= 2.0
                elif sd > 10 * rp:
                    rho /= 2.0; u *= 2.0
            gap = l1_gap(A, At, y, z, lam, groups)
            if gap <= tol:
                break
    admm_l1.last_iters, admm_l1.last_gap = it, gap
    return z

def admm_bp(A, At, y, n, gram_solve, rho=1.0, tol=1e-6, max_iters=100000, balance=5000, check=10):
    """basis pursuit  min ||c||_1  s.t.  A c = y  ด้วย ADMM (c อยู่บนเซตที่เป็นไปได้, z รับโทษ ล1, c = z)
       gram_solve(v) คืน (A A^T)^{-1} v  การฉายลงเซต {A c = y} จึงเป็น v - A^T gram_solve(A v - y)
       หยุดเมื่อช่องว่างคู่กันสัมพัทธ์ (||c||_1 - y^T nu) / ||c||_1 ไม่เกิน tol
       โดย nu = nu0 / max(1, ||A^T nu0||_inf) และ nu0 = gram_solve(A w) จากตัวคูณ w = rho u ซึ่งเป็นอนุพันธ์ย่อยของ ||z||_1"""
    proj = lambda v: v - At(gram_solve(A(v) - y))
    z = np.zeros(n); u = np.zeros(n); gap = np.inf
    for it in range(1, max_iters + 1):
        c = proj(z - u)
        z_old = z
        z = soft(c + u, 1.0 / rho)
        u = u + c - z
        if it % check == 0:
            if it <= balance:
                rp = np.linalg.norm(c - z); sd = rho * np.linalg.norm(z - z_old)
                if rp > 10 * sd:
                    rho *= 2.0; u /= 2.0
                elif sd > 10 * rp:
                    rho /= 2.0; u *= 2.0
            nu = gram_solve(A(rho * u))
            nu = nu / max(1.0, np.max(np.abs(At(nu))))
            P = np.sum(np.abs(c))
            gap = (P - np.dot(y, nu)) / P
            if gap <= tol:
                break
    admm_bp.last_iters, admm_bp.last_gap = it, gap
    return c

def bp_lp(Amat, y):
    """basis pursuit ด้วยกำหนดการเชิงเส้น min 1^T (p + q) s.t. A (p - q) = y, p, q >= 0 (HiGHS, วิธีจุดภายใน)"""
    from scipy.optimize import linprog
    n = Amat.shape[1]
    res = linprog(np.ones(2 * n), A_eq=np.hstack([Amat, -Amat]), b_eq=y, bounds=(0, None), method="highs-ipm",
                  options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10})
    assert res.status == 0, res.message
    return res.x[:n] - res.x[n:]

def fwht(x):
    """การแปลงวอลช์--อาดามาร์เร็ว (ไม่ปรับมาตราส่วน ลำดับแบบซิลเวสเตอร์) ของเวกเตอร์ความยาวกำลังของสอง"""
    x = np.array(x, dtype=float); n = x.size; h = 1
    while h < n:
        x = x.reshape(-1, 2, h)
        x = np.stack((x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]), axis=1).reshape(n)
        h *= 2
    return x

def iht_k(A, At, y, n, k, iters, alpha=1.0):
    c = np.zeros(n)
    for _ in range(iters):
        v = c + alpha * At(y - A(c))
        S = np.argsort(np.abs(v))[::-1][:k]
        c = np.zeros(n); c[S] = v[S]
    return c

def irls_dr_mat(Phi, y, p=0.9, xi=1e-6, nmax=100, snap=()):
    """IRLS ด้วยการทำให้เป็นปรกติเชิงทแยง (บทความของผู้แต่ง): x <- D Phi^T (Phi D Phi^T + xi I)^{-1} y
       snap คือรอบที่ต้องการเก็บผลลัพธ์ระหว่างทาง (ใช้ตรวจการลู่เข้า)"""
    m, n = Phi.shape; x = np.ones(n); snaps = {}
    with np.errstate(all="ignore"):
        for it in range(1, nmax + 1):
            d = np.abs(x) ** (2 - p)
            G = (Phi * d) @ Phi.T + xi * np.eye(m)
            try:
                xn = d * (Phi.T @ np.linalg.solve(G, y))
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(xn)):
                break
            x = xn
            if it in snap:
                snaps[it] = x.copy()
    return (x, snaps) if snap else x

if want("irls"):
    sizes = [32, 40, 48, 56, 64]
    frac_m, lam, it_f, it_i, it_r, p_r = 0.45, 1e-3, 600, 600, 100, 0.9   # IRLS-DR: p และจำนวนรอบเท่ากับบทความต้นฉบับ
    res = {k: {"err": [], "t": []} for k in ["FISTA", "IHT", "IRLS-DR"]}
    ns = []
    print(f"irls: textured image at five resolutions, m = {frac_m}n Gaussian measurements, sparsity basis {WAVE};"
          f" FISTA {it_f} iterations (lambda = {lam}), IHT {it_i} iterations, IRLS-DR {it_r} iterations (p = {p_r}, xi = 1e-6)")
    for n1 in sizes:
        img = textured_image(n1, 2)
        lev = 3                                                       # ระดับการแยกเท่ากันทุกขนาดเพื่อให้เปรียบเทียบกันได้
        arr, sl = wav_l(img, lev)
        srt = np.sort(arr.ravel() ** 2)[::-1]
        k99 = int(np.sum(np.cumsum(srt) < 0.99 * srt.sum()))
        n = n1 * n1; m = int(frac_m * n); ns.append(n)
        Phi = np.random.default_rng(6).standard_normal((m, n)) / np.sqrt(m)
        A = lambda c: Phi @ iwav_l(c.reshape(n1, n1), sl).ravel()
        At = lambda r: wav_l((Phi.T @ r).reshape(n1, n1), lev)[0].ravel()
        y = Phi @ img.ravel(); Lc = op_norm(A, At, n) ** 2
        rec = {}
        t0 = time.time(); c = fista_l1(A, At, y, n, lam, it_f, Lc); rec["FISTA"] = (c, time.time() - t0)
        t0 = time.time(); c = iht_k(A, At, y, n, k99, it_i, 1 / Lc); rec["IHT"] = (c, time.time() - t0)
        t0 = time.time()
        PhiC = np.zeros((m, n)); e = np.zeros(n)
        for j in range(n):
            e[j] = 1.0; PhiC[:, j] = A(e); e[j] = 0.0
        t_build = time.time() - t0
        c = irls_dr_mat(PhiC, y, p_r, 1e-6, it_r); rec["IRLS-DR"] = (c, time.time() - t0)
        t_iter = (time.time() - t0 - t_build) / it_r
        line = f"   {n1:2d}x{n1} (n = {n:5d}, m = {m:4d}, k99 = {k99:4d}; IRLS-DR build {t_build:.2f} s, {t_iter:.3f} s/iteration): "
        for k_ in res:
            cc, dt = rec[k_]
            xh = np.clip(iwav_l(np.asarray(cc).reshape(n1, n1), sl), 0, 1)
            res[k_]["err"].append(nerr(xh, img)); res[k_]["t"].append(dt)
            line += f"{k_} {res[k_]['err'][-1]:.4f} / {dt:7.2f} s   "
            if n1 == sizes[-1]:
                rec[k_] = (xh, dt)
        print(line)
    print("irls: tuning at the largest size (64x64), same measurements")
    n1 = sizes[-1]; img = textured_image(n1, 2); arr, sl = wav_l(img, 3); n = n1 * n1; m = int(frac_m * n)
    Phi = np.random.default_rng(6).standard_normal((m, n)) / np.sqrt(m)
    A = lambda c: Phi @ iwav_l(c.reshape(n1, n1), sl).ravel()
    At = lambda r: wav_l((Phi.T @ r).reshape(n1, n1), 3)[0].ravel()
    y = Phi @ img.ravel(); Lc = op_norm(A, At, n) ** 2
    PhiC = np.zeros((m, n)); e = np.zeros(n)
    for j in range(n):
        e[j] = 1.0; PhiC[:, j] = A(e); e[j] = 0.0
    for p_ in [0.5, 0.8, 0.9, 1.0]:
        _, sn = irls_dr_mat(PhiC, y, p_, 1e-6, 200, snap=(30, 100, 200))
        row = [nerr(np.clip(iwav_l(sn[k].reshape(n1, n1), sl), 0, 1), img) if k in sn else float("nan") for k in (30, 100, 200)]
        print(f"   IRLS-DR p = {p_:.1f}: normalized error {row[0]:.4f} (30 it), {row[1]:.4f} (100 it), {row[2]:.4f} (200 it)")
    for lam_ in [3e-4, 1e-3, 3e-3]:
        row = []
        for it in [600, 2000]:
            c = fista_l1(A, At, y, n, lam_, it, Lc)
            row.append(nerr(np.clip(iwav_l(c.reshape(n1, n1), sl), 0, 1), img))
        print(f"   FISTA lambda = {lam_:.0e}: normalized error {row[0]:.4f} (600 it), {row[1]:.4f} (2000 it)")
    ns = np.array(ns, float)
    sl_f = np.polyfit(np.log(ns), np.log(res["FISTA"]["t"]), 1)[0]
    sl_r = np.polyfit(np.log(ns), np.log(res["IRLS-DR"]["t"]), 1)[0]
    print(f"   fitted cost exponent (log-log slope of time vs n): FISTA {sl_f:.2f}, IRLS-DR {sl_r:.2f}"
          f"  (theory: FISTA O(mn) -> 2, IRLS-DR O(m^2 n) -> 3)")
    n_pap = 153 * 143; m_pap = 10940
    scale = (m_pap ** 2 * n_pap) / ((frac_m * ns[-1]) ** 2 * ns[-1])
    print(f"   extrapolating the measured IRLS-DR time by the O(m^2 n) law to the image of the author's paper"
          f" (153x143, m = {m_pap}, {it_r} iterations): {res['IRLS-DR']['t'][-1] * scale / 3600:.1f} h"
          f" (the paper reports 22,890 s = {22890/3600:.1f} h)")
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.6))
    st = {"FISTA": ("o-", ACC), "IHT": ("s--", GRN), "IRLS-DR": ("d-", RED)}
    for k_ in res:
        axes[0].semilogx(ns, res[k_]["err"], st[k_][0], color=st[k_][1], ms=3.5, label=k_)
        axes[1].loglog(ns, res[k_]["t"], st[k_][0], color=st[k_][1], ms=3.5, label=k_)
    axes[0].set_xlabel("$n$ (pixels)"); axes[0].set_ylabel("normalized error")
    axes[0].set_title("(a)", fontsize=9, loc="left"); axes[0].legend(frameon=False, fontsize=7)
    axes[1].set_xlabel("$n$ (pixels)"); axes[1].set_ylabel("time (s)")
    axes[1].set_title("(b)", fontsize=9, loc="left")
    axes[1].annotate(f"slope {sl_r:.1f}", (ns[-2], res["IRLS-DR"]["t"][-2]), fontsize=7, color=RED,
                     textcoords="offset points", xytext=(-40, 9))
    axes[1].annotate(f"slope {sl_f:.1f}", (ns[-2], res["FISTA"]["t"][-2]), fontsize=7, color=ACC,
                     textcoords="offset points", xytext=(-12, -17))
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig06-irls")

# ============================================================
# fig06-optical : กล้องจุดภาพเดียวและการถ่ายภาพไฮเปอร์สเปกตรัมแบบช็อตเดียว
# ============================================================
def hadamard_rows(n, rows):
    """แถวที่เลือกของเมทริกซ์อาดามาร์ขนาด n (n เป็นกำลังของสอง) โดยไม่สร้างเมทริกซ์เต็ม"""
    idx = np.arange(n)
    H = np.ones((len(rows), n))
    for b in range(int(np.log2(n))):
        H *= np.where(((idx >> b) & 1) & ((np.asarray(rows)[:, None] >> b) & 1), -1.0, 1.0)
    return H

OPT_TOL = 1e-6          # ช่องว่างคู่กันสัมพัทธ์ที่ใช้หยุดตัวแก้ปัญหาในส่วนการถ่ายภาพเชิงแสง
OPT_ITERS = []
OPT_FALLBACK = []      # ช่องว่างคู่กันของผลเฉลยที่ได้จากวิธีจุดภายใน

def solve_lasso(A, At, y, n, lam, solve=None, Lc=None, groups=None, z0=None):
    """แก้ LASSO (หรือ LASSO เชิงกลุ่ม) จนช่องว่างคู่กันสัมพัทธ์ไม่เกิน OPT_TOL
       solve ไม่เป็น None: ADMM ที่ใช้ (rho I + A A^T)^{-1} รูปปิด ; มิฉะนั้น: FISTA ที่เริ่มโมเมนตัมใหม่ด้วยค่าคงตัวลิปชิตซ์ Lc"""
    if solve is not None:
        z = admm_l1(A, At, y, n, lam, solve, tol=OPT_TOL, groups=groups, max_iters=150000, z0=z0); it, g = admm_l1.last_iters, admm_l1.last_gap
    else:
        z = fista_gap(A, At, y, n, lam, Lc, tol=OPT_TOL, groups=groups, max_iters=300000); it, g = fista_gap.last_iters, fista_gap.last_gap
    OPT_ITERS.append(it)
    if g > OPT_TOL and groups is None and len(y) <= 1024:
        # ปัญหาที่ลู่เข้าช้ามาก (แถวอาดามาร์จำนวนน้อย): แก้ซ้ำด้วยวิธีจุดภายใน (Clarabel) บนเมทริกซ์ A ที่สร้างอย่างชัดแจ้ง
        import cvxpy as cp
        Amat = np.column_stack([A(e) for e in np.eye(n)])
        cv = cp.Variable(n)
        cp.Problem(cp.Minimize(0.5 * cp.sum_squares(Amat @ cv - y) + lam * cp.norm1(cv))).solve(
            solver="CLARABEL", tol_gap_rel=1e-10, tol_gap_abs=1e-12, tol_feas=1e-10)
        z_ipm = np.asarray(cv.value).ravel(); g_ipm = l1_gap(A, At, y, z_ipm, lam)
        print(f"   [note] first-order solver stopped at relative gap {g:.1e} (lambda {lam:.0e}, m = {len(y)}); "
              f"interior-point solution has relative gap {g_ipm:.1e}", flush=True)
        OPT_FALLBACK.append(g_ipm)
        z, g = z_ipm, g_ipm
    if g > OPT_TOL:
        print(f"   [warning] solution not certified: relative gap {g:.1e} (lambda {lam:.0e})", flush=True)
    solve_lasso.last_gap = g
    return z

def lam_cont(run, grid=(1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4), tag=""):
    """เลือกตัวคูณที่ให้ PSNR สูงสุด (oracle) บนกริด 1-3-10 ทั้งช่วง โดยแก้จากค่ามากไปค่าน้อย (continuation)
       และใช้ผลเฉลยของค่าก่อนหน้าเป็นจุดเริ่มต้น  run(lam, z0) คืน (PSNR, ผลเฉลย, ช่องว่างคู่กันสัมพัทธ์)
       ผลเฉลยที่ยังไม่ผ่านเกณฑ์ OPT_TOL ไม่นำมาเลือก  ถ้าค่าที่ดีที่สุดอยู่ที่ขอบจะขยายกริดออกไปทางนั้น  คืนค่า (PSNR, ผลเฉลย, lam, ผลทั้งหมด)"""
    res = {}; z = None
    for l in sorted(grid, reverse=True):
        p_, z, g_ = run(l, z); res[l] = (p_, z, g_)
    for _ in range(8):                                                # ขยายกริดเมื่อค่าที่ดีที่สุดอยู่ที่ขอบ (ขึ้นหรือลง)
        cert = {l: v for l, v in res.items() if v[2] <= OPT_TOL}
        b = max(cert, key=lambda l: cert[l][0])
        if b == max(res):
            nl = _step(b, down=False)
        elif b == min(res):
            nl = _step(b, down=True)
        else:
            break
        p_, z_, g_ = run(nl, res[b][1]); res[nl] = (p_, z_, g_)
        if g_ > OPT_TOL:                                              # ค่าที่ขยายแล้วแก้ไม่ลู่เข้า: หยุดขยาย
            break
    cert = {l: v for l, v in res.items() if v[2] <= OPT_TOL}
    b = max(cert, key=lambda l: cert[l][0])
    unc = sorted(l for l, v in res.items() if v[2] > OPT_TOL)
    if b == min(cert) and len(cert) < len(res):
        print(f"   [note] {tag}: best certified lambda {b:.0e} is the smallest certified value; uncertified: "
              + " ".join(f"{l:.0e}" for l in unc))
    return res[b][0], res[b][1], b, res

def lam_climb(run, start=(1e-3, 3e-3), max_steps=8, tag=""):
    """เลือกตัวคูณที่ให้ PSNR สูงสุด (oracle) โดยไต่ระดับบนกริด 1-3-10 จากสองค่าเริ่มต้น ไปในทิศที่ PSNR ดีขึ้นจนกว่าจะลดลง
       run(lam, z0) คืน (PSNR, ผลเฉลย) โดย z0 คือผลเฉลยที่ตัวคูณข้างเคียงใช้เป็นจุดเริ่มต้น ; คืนค่า (PSNR, ผลเฉลย, lam)"""
    lo, hi = start
    res = {hi: run(hi, None)}
    res[lo] = run(lo, res[hi][1])
    down = res[lo][0] > res[hi][0]
    cur = lo if down else hi
    for _ in range(max_steps):
        nl = _step(cur, down)
        res[nl] = run(nl, res[cur][1])
        if res[nl][0] <= res[cur][0]:
            break
        cur = nl
    else:
        print(f"   [note] {tag}: PSNR still improving after {max_steps} grid steps")
    b = max(res, key=lambda l: res[l][0])
    return res[b][0], res[b][1], b

def lam_search(run, grid, max_ext=4, tag=""):
    """เลือกตัวคูณที่ให้ PSNR สูงสุดบนกริด 1-3-10 (oracle) และขยายกริดเมื่อค่าที่ดีที่สุดอยู่ที่ขอบ
       run(lam) คืน (PSNR, ผลเฉลย) ; คืนค่า (PSNR, ผลเฉลย, lam)"""
    res = {l: run(l) for l in grid}
    for _ in range(max_ext):
        ls = sorted(res); b = max(ls, key=lambda l: res[l][0])
        if b == ls[0]:
            nl = _step(b, True)
        elif b == ls[-1]:
            nl = _step(b, False)
        else:
            break
        res[nl] = run(nl)
    ls = sorted(res); b = max(ls, key=lambda l: res[l][0])
    if b in (ls[0], ls[-1]):
        print(f"   [note] {tag}: best lambda {b:.0e} still lies on the edge of the extended grid")
    return res[b][0], res[b][1], b

# ส่วนที่เลือกรันได้: optical (ทุกส่วนแล้ววาดภาพ) หรือแยกส่วน optical-sp, optical-noise, optical-cassi1/2/4/8, optical-plot
# ผลของแต่ละส่วนเก็บใน logs/fig06_optical_results.json (และภาพที่สร้างคืนของ CASSI ใน logs/fig06_optical_cassi_recs.npz)
# เพื่อให้แต่ละส่วนรันแยกกันได้ ตัวสุ่มเล่นซ้ำลำดับเดิมทุกครั้ง ผลจึงเท่ากับการรันรวม
_LOGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
OPT_RES = os.path.join(_LOGDIR, "fig06_optical_results.json")
OPT_RECS = os.path.join(_LOGDIR, "fig06_optical_cassi_recs.npz")

def _opt_load():
    import json
    return json.load(open(OPT_RES)) if os.path.exists(OPT_RES) else {}

def _opt_save(upd):
    """บันทึกผลแบบกันการเขียนชนกันเมื่อรันหลายส่วนพร้อมกัน ใช้ fcntl บน Linux/macOS และ msvcrt บน Windows
       (แก้ตามรายงานตรวจ C02: เดิมนำเข้า fcntl ซึ่งไม่มีบน Windows)"""
    import json
    with open(OPT_RES + ".lock", "w") as lk:
        try:
            import fcntl
            fcntl.flock(lk, fcntl.LOCK_EX)
        except ImportError:                                           # Windows
            import msvcrt
            msvcrt.locking(lk.fileno(), msvcrt.LK_LOCK, 1)
        d = _opt_load(); d.update(upd)
        with open(OPT_RES + ".tmp", "w") as f:
            json.dump(d, f, indent=1)
        os.replace(OPT_RES + ".tmp", OPT_RES)

OPT_SP = want("optical") or "optical-sp" in RUN
OPT_NOISE = want("optical") or "optical-noise" in RUN
OPT_CASSI = [K for K in (1, 2, 4, 8) if want("optical") or f"optical-cassi{K}" in RUN]
OPT_PLOT = want("optical") or "optical-plot" in RUN

if OPT_SP or OPT_NOISE or OPT_CASSI or OPT_PLOT:
    n1 = 64; n = n1 * n1; lev = 3
    img = textured_image(n1, 2)
    arr, sl = wav_l(img, lev)
    rng = np.random.default_rng(8)
    synth = lambda c: iwav_l(c.reshape(n1, n1), sl).ravel()               # สัมประสิทธิ์เวฟเล็ต -> ภาพ (ตั้งฉาก)
    anal = lambda v: wav_l(v.reshape(n1, n1), lev)[0].ravel()             # ภาพ -> สัมประสิทธิ์เวฟเล็ต
    fracs = [0.1, 0.2, 0.3, 0.45, 0.6]
    trials = 3
    sp_designs = ["Bernoulli", "Hadamard", "Hadamard, no DC row"]
    def sp_draw(nm, m):
        """สุ่มเมทริกซ์การวัดของกล้องจุดภาพเดียว (ลำดับการใช้ตัวสุ่มต้องเหมือนเดิมทุกครั้ง)"""
        if nm == "Bernoulli":
            return rng.choice([-1.0, 1.0], (m, n)) / np.sqrt(m)
        if nm == "Hadamard, no DC row":
            return rng.choice(np.arange(1, n), m, replace=False)
        return np.concatenate(([0], rng.choice(np.arange(1, n), m - 1, replace=False)))  # แถวที่ศูนย์ (ค่าเฉลี่ย) ต้องถูกเลือกเสมอ
    if OPT_SP:
        print(f"optical: single-pixel camera, image {n1}x{n1} (n = {n}), sparsity basis {WAVE} (level {lev})")
        print(f"     noiseless: basis pursuit in the wavelet domain, solved exactly (m <= 1024: LP by HiGHS interior point;"
              f" m > 1024: ADMM to relative duality gap {OPT_TOL:g})")
        curves = {nm: [] for nm in sp_designs}
        OPT_ITERS.clear(); bp_res = []
        Wsyn = np.column_stack([synth(e) for e in np.eye(n)])            # เมทริกซ์สังเคราะห์ (ใช้สร้าง A อย่างชัดแจ้งเมื่อ m เล็ก)
        for fr in fracs:
            m = int(fr * n)
            acc = {nm: [] for nm in curves}
            for tr in range(trials):
                for nm in curves:
                    D = sp_draw(nm, m)
                    if nm == "Bernoulli":
                        Pf = lambda v, P=D: P @ v
                        Pt = lambda w, P=D: P.T @ w
                        rows_dense = lambda P=D: P
                        ev, Q = (None, None) if m <= 1024 else np.linalg.eigh(D @ D.T)   # A A^T = Phi Phi^T (เวฟเล็ตตั้งฉาก)
                        gram = lambda v, Q=Q, ev=ev: Q @ ((Q.T @ v) / ev)
                    else:
                        Pf = lambda v, R=D: fwht(v)[R] / np.sqrt(m)       # แถวของเมทริกซ์อาดามาร์ผ่านการแปลงเร็ว
                        def Pt(w, R=D):
                            z = np.zeros(n); z[R] = w; return fwht(z) / np.sqrt(m)
                        rows_dense = lambda R=D: hadamard_rows(n, R) / np.sqrt(m)
                        gram = lambda v: v / (n / m)                       # A A^T = (n/m) I สำหรับแถวของเมทริกซ์อาดามาร์
                    A = lambda c, Pf=Pf: Pf(synth(c)); At = lambda w, Pt=Pt: anal(Pt(w))
                    y = Pf(img.ravel())
                    if m <= 1024:
                        z = bp_lp(rows_dense() @ Wsyn, y)
                    else:
                        z = admm_bp(A, At, y, n, gram, tol=OPT_TOL); OPT_ITERS.append(admm_bp.last_iters)
                        if admm_bp.last_gap > OPT_TOL:
                            print(f"   [warning] m/n = {fr:.2f} {nm} trial {tr}: ADMM stopped with relative gap {admm_bp.last_gap:.1e}"
                                  " (checked by LP in code/fig06_optical_lpcheck.py)", flush=True)
                    bp_res.append(np.linalg.norm(A(z) - y) / np.linalg.norm(y))
                    acc[nm].append(psnr(np.clip(synth(z).reshape(n1, n1), 0, 1), img))
            for nm in curves:
                curves[nm].append(float(np.mean(acc[nm])))
                print(f"   m/n = {fr:.2f}  {nm:20s} PSNR per trial " + " ".join(f"{v:6.2f}" for v in acc[nm]), flush=True)
        for nm in curves:
            print(f"   {nm:20s} mean PSNR (dB) at m/n = " + " ".join(f"{f_:.2f}:{v:6.2f}" for f_, v in zip(fracs, curves[nm])))
        print(f"   ADMM iterations (m > 1024): median {int(np.median(OPT_ITERS))}, max {max(OPT_ITERS)} ({len(OPT_ITERS)} solves); "
              f"largest relative residual ||A c - y|| / ||y|| = {max(bp_res):.1e}", flush=True)
        _opt_save({"fracs": fracs, "curves": curves})
    elif OPT_NOISE:
        for fr in fracs:                                                  # เล่นตัวสุ่มซ้ำโดยไม่แก้ปัญหา
            for tr in range(trials):
                for nm in sp_designs:
                    sp_draw(nm, int(fr * n))
    if OPT_NOISE:
        # --- การพับสัญญาณรบกวน: การวัดแบบมัลติเพล็กซ์เทียบกับการวัดจุดภาพโดยตรง ที่จำนวนการวัดและสัญญาณรบกวนต่อการวัดเท่ากัน
        m = int(0.35 * n)
        Bmat = rng.choice([-1.0, 1.0], (m, n))                            # แต่ละการวัดคือผลรวมเชิงเครื่องหมายของ n จุดภาพ
        Phi = Bmat / np.sqrt(n)                                           # ปรับมาตราส่วนทั้งสองข้างของสมการด้วย 1/sqrt(n)
        S_pix = rng.choice(n, m, replace=False)
        sigmas = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
        out_cs, out_dir = [], []
        OPT_ITERS.clear()
        print(f"optical: noise folding, m = {m} ({m/n:.2f}n) measurements, same detector noise per measurement (solved to gap {OPT_TOL:g})")
        A = lambda c: Phi @ synth(c); At = lambda r: anal(Phi.T @ r)
        Ad = lambda c: synth(c)[S_pix]
        def Atd(r):
            z = np.zeros(n); z[S_pix] = r; return anal(z)
        Lc = op_norm(A, At, n) ** 2; Ld = 1.0                             # A_d A_d^T = I สำหรับการเลือกจุดภาพ
        for sg in sigmas:
            y = Phi @ img.ravel() + (sg / np.sqrt(n)) * rng.standard_normal(m)
            lamc = max(2.0 * (sg / np.sqrt(n)) * np.sqrt(2 * np.log(n)), 5e-4)   # ขั้นต่ำเท่ากับค่าที่ใช้ในกรณีไร้สัญญาณรบกวน
            c = solve_lasso(A, At, y, n, lamc, None, Lc)
            out_cs.append(psnr(np.clip(synth(c).reshape(n1, n1), 0, 1), img))
            yd = img.ravel()[S_pix] + sg * rng.standard_normal(m)          # วัดจุดภาพโดยตรง m จุด
            cd = solve_lasso(Ad, Atd, yd, n, max(2.0 * sg * np.sqrt(2 * np.log(n)), 5e-4), None, Ld)
            out_dir.append(psnr(np.clip(synth(cd).reshape(n1, n1), 0, 1), img))
        for sg, a_, b_ in zip(sigmas, out_cs, out_dir):
            print(f"   sigma {sg:7.1e}: multiplexed {a_:5.2f} dB   direct pixel sampling {b_:5.2f} dB   gain {a_-b_:+5.2f} dB")
        print(f"   solver iterations: median {int(np.median(OPT_ITERS))}, max {max(OPT_ITERS)} ({len(OPT_ITERS)} solves)", flush=True)
        _opt_save({"sigmas": sigmas, "out_cs": out_cs, "out_dir": out_dir})
    # --- การถ่ายภาพไฮเปอร์สเปกตรัมแบบ CASSI (ยิงหลายช็อต รหัสต่างกันในแต่ละช็อต)
    ns_, nb = 32, 16
    gx = np.linspace(0, 1, ns_); X, Y = np.meshgrid(gx, gx)
    maps = [((X - .3) ** 2 + (Y - .3) ** 2 < .03).astype(float),
            ((np.abs(X - .7) < .22) & (np.abs(Y - .65) < .18)).astype(float),
            np.clip(1.2 - 1.6 * ((X - .5) ** 2 + (Y - .5) ** 2) ** 0.5, 0, 1)]
    lamv = np.linspace(0, 1, nb)
    sigs = [np.exp(-((lamv - .25) / .13) ** 2), 0.3 + 0.6 * lamv, np.exp(-((lamv - .72) / .20) ** 2)]
    cube = sum(m_[:, :, None] * s_[None, None, :] for m_, s_ in zip(maps, sigs))
    cube = cube / cube.max()
    from scipy.fft import dctn, idctn
    _, sl2 = wav_l(np.zeros((ns_, ns_)), 2)
    def syn(c):
        sp = idctn(c.reshape(ns_ * ns_, nb).reshape(ns_, ns_, nb), axes=(2,), norm="ortho")
        return np.stack([iwav_l(sp[:, :, b], sl2) for b in range(nb)], axis=2)
    def ana(v3):
        w = np.stack([wav_l(v3[:, :, b], 2)[0] for b in range(nb)], axis=2)
        return dctn(w, axes=(2,), norm="ortho").ravel()
    def make_cassi(codes):
        K = len(codes)
        def fwd(c3):
            out = np.zeros((K, ns_, ns_ + nb - 1))
            for s_ in range(K):
                masked = codes[s_][:, :, None] * c3                   # รหัสช่องเปิดอยู่ในระนาบวัตถุ จึงไม่ขึ้นกับแถบคลื่น
                for b in range(nb):
                    out[s_, :, b:b + ns_] += masked[:, :, b]          # ปริซึมเฉือนแถบคลื่นที่ b ไป b จุดภาพ
            return out
        def adj(r):
            out = np.zeros((ns_, ns_, nb))
            for s_ in range(K):
                for b in range(nb):
                    out[:, :, b] += codes[s_] * r[s_][:, b:b + ns_]
            return out
        return fwd, adj
    N3 = ns_ * ns_ * nb
    rngc = np.random.default_rng(9)
    shots = [1, 2, 4, 8]
    if OPT_CASSI:
        print(f"optical: CASSI, cube {ns_}x{ns_}x{nb} (N = {N3}); each snapshot gives {ns_*(ns_+nb-1)} detector pixels")
        print(f"     LASSO and group LASSO solved by ADMM (block-diagonal A A^T) to relative duality gap {OPT_TOL:g};"
              " lambda on the 1-3-10 grid 3e-4 ... 1e-1 by continuation with warm starts, uncertified solutions excluded (oracle)")
        OPT_ITERS.clear()
    for K in shots:
        codes = [(rngc.random((ns_, ns_)) < 0.5).astype(float) for _ in range(K)]   # สุ่มทุกค่า K เพื่อคงลำดับตัวสุ่ม
        if K not in OPT_CASSI:
            continue
        fwd, adj = make_cassi(codes)
        y2 = fwd(cube)
        A = lambda c: fwd(syn(c)).ravel()
        At = lambda r: ana(adj(r.reshape(K, ns_, ns_ + nb - 1)))
        # A A^T = F F^T แยกเป็นบล็อก K x K ต่อจุดภาพของตัวรับรู้ เพราะแต่ละสมาชิกของลูกบาศก์ตกบนจุดภาพเดียวในทุกช็อต
        Csh = np.zeros((K, ns_, ns_ + nb - 1)); Mblk = np.zeros((ns_, ns_ + nb - 1, K, K))
        for b in range(nb):
            Csh[:] = 0.0
            for s_ in range(K):
                Csh[s_, :, b:b + ns_] = codes[s_]
            Mblk += np.einsum("sij,tij->ijst", Csh, Csh)
        wM, VM = np.linalg.eigh(Mblk)
        def solve(v, rho, wM=wM, VM=VM, K=K):
            V3 = v.reshape(K, ns_, ns_ + nb - 1)
            co = np.einsum("ijst,sij->ijt", VM, V3) / (rho + wM)
            return np.einsum("ijst,ijt->sij", VM, co).ravel()
        res_K = {}
        for nm in ["separable l1", "joint l2,1"]:
            grp = None if nm == "separable l1" else (ns_ * ns_, nb)
            def run(lm, z0, A=A, At=At, y=y2.ravel(), solve=solve, grp=grp):
                z = solve_lasso(A, At, y, N3, lm, solve, None, groups=grp, z0=z0)
                return psnr(np.clip(syn(z), 0, 1), cube), z, solve_lasso.last_gap
            pb, zb, lb, allres = lam_cont(run, tag=f"CASSI K = {K} {nm}")
            gaps = {l: v[2] for l, v in allres.items()}
            res_K[nm] = [pb, lb, gaps[lb], {f"{l:.0e}": [round(v[0], 3), v[2]] for l, v in sorted(allres.items())}]
            if K == 4:
                recs = dict(np.load(OPT_RECS)) if os.path.exists(OPT_RECS) else {}
                recs["sep" if grp is None else "joint"] = np.clip(syn(zb), 0, 1)
                np.savez(OPT_RECS, **recs)                               # มีเพียงส่วน K = 4 ที่เขียนไฟล์นี้
            print(f"   K = {K}: compression {K*y2[0].size/N3:.3f}  {nm:13s} best lambda {lb:.0e}  PSNR {pb:5.2f} dB"
                  f"  (relative gap {gaps[lb]:.1e})   PSNR [gap] per lambda: "
                  + "  ".join(f"{l:.0e}: {v[0]:.2f} [{v[2]:.0e}]" for l, v in sorted(allres.items())), flush=True)
        _opt_save({f"cassi_K{K}": res_K})
    if OPT_CASSI:
        print(f"   solver iterations: median {int(np.median(OPT_ITERS))}, max {max(OPT_ITERS)} ({len(OPT_ITERS)} solves)")
    if OPT_PLOT:
        R_ = _opt_load(); recs = np.load(OPT_RECS)
        fracs, curves = R_["fracs"], R_["curves"]
        sigmas, out_cs, out_dir = R_["sigmas"], R_["out_cs"], R_["out_dir"]
        cassi_curves = {nm: [R_[f"cassi_K{K}"][nm][0] for K in shots] for nm in ["separable l1", "joint l2,1"]}
        rec, rec_sep = recs["joint"], recs["sep"]
        print("optical: figure from logs/fig06_optical_results.json")
        print("   CASSI PSNR (separable l1 / joint l2,1) at K = " + "  ".join(
            f"{K}: {cassi_curves['separable l1'][i]:.2f} / {cassi_curves['joint l2,1'][i]:.2f}" for i, K in enumerate(shots)))
        print("   per-band PSNR at K = 4 (joint): " + " ".join(f"{psnr(rec[:,:,b], cube[:,:,b]):.1f}" for b in range(0, nb, 3)))
        fig, axes = plt.subplots(2, 3, figsize=(6.2, 4.0))
        axes[0, 0].plot(fracs, curves["Bernoulli"], "o-", color=ACC, ms=3, label="Bernoulli")
        axes[0, 0].plot(fracs, curves["Hadamard"], "s--", color=GRN, ms=3, label="Hadamard")
        axes[0, 0].plot(fracs, curves["Hadamard, no DC row"], "v:", color=GRY, ms=3, label="Hadamard, no DC")
        axes[0, 0].set_xlabel("$m/n$"); axes[0, 0].set_ylabel("PSNR (dB)"); axes[0, 0].set_ylim(3, 46); axes[0, 0].legend(frameon=False, fontsize=6, loc="upper left", handlelength=1.6)
        axes[0, 0].set_title("(a)", fontsize=9, loc="left")
        axes[0, 1].semilogx(sigmas, out_cs, "o-", color=ACC, ms=3, label="multiplexed")
        axes[0, 1].semilogx(sigmas, out_dir, "s--", color=RED, ms=3, label="direct pixels")
        axes[0, 1].set_xlabel(r"detector noise $\sigma$"); axes[0, 1].set_ylabel("PSNR (dB)")
        axes[0, 1].legend(frameon=False, fontsize=6); axes[0, 1].set_title("(b)", fontsize=9, loc="left")
        axes[0, 2].plot(shots, cassi_curves["joint l2,1"], "o-", color=ACC, ms=3, label=r"joint $\ell_{2,1}$")
        axes[0, 2].plot(shots, cassi_curves["separable l1"], "s--", color=GRY, ms=3, label=r"separable $\ell_1$")
        axes[0, 2].set_xlabel(r"snapshots $n_{\mathrm{s}}$"); axes[0, 2].set_ylabel("PSNR (dB)")
        axes[0, 2].legend(frameon=False, fontsize=6); axes[0, 2].set_title("(c)", fontsize=9, loc="left")
        show(axes[1, 0], cube[:, :, 6], "(d) band 7, truth")
        show(axes[1, 1], rec[:, :, 6], f"(e) joint, {psnr(rec[:,:,6], cube[:,:,6]):.1f} dB")
        show(axes[1, 2], rec_sep[:, :, 6], f"(f) separable, {psnr(rec_sep[:,:,6], cube[:,:,6]):.1f} dB")
        fig.tight_layout(h_pad=1.0, w_pad=0.8)
        save(fig, "fig06-optical")
