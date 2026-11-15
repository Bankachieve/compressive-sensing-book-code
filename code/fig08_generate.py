# -*- coding: utf-8 -*-
r"""สร้างรูปประกอบและผลการทดลองเชิงตัวเลขทั้งหมดของบทที่ 8 -> figures/fig08-*.pdf
ส่วนที่เลือกรันได้: gridless onebit pnp unroll  เช่น `python fig08_generate.py onebit unroll`
ตัวเลขทุกค่าที่อ้างในเนื้อหาและคำบรรยายภาพพิมพ์ออกทางจอเมื่อรันสคริปต์นี้
ต้องการ numpy scipy matplotlib pywt scikit-image cvxpy (SCS) autograd"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle  # กริดพื้นหลังและตำแหน่ง legend แบบเดียวกันทั้งเล่ม
import os, sys, time, warnings, importlib.util
warnings.filterwarnings("ignore")

RUN = set(sys.argv[1:])
def want(name):
    return (not RUN) or (name in RUN)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "figures")
os.makedirs(OUT, exist_ok=True)
INK, ACC, GRY, RED, GRN, ORG, PUR = "#1F2A37", "#1D4ED8", "#9AA3AF", "#B91C1C", "#047857", "#B45309", "#6D28D9"
plt.rcParams.update({
    "font.size": 9, "axes.linewidth": 0.6, "axes.edgecolor": INK,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.labelcolor": INK, "lines.linewidth": 1.3,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "image.cmap": "gray",
})

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf")
    plt.close(fig)
    print("saved", name)

def soft(z, t):
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)

def hard_k(z, k):
    out = np.zeros_like(z)
    if k <= 0:                                   # ป้องกัน slice [-0:] ซึ่งเลือกทุกสมาชิก
        return out
    idx = np.argsort(np.abs(z))[-min(k, z.size):]
    out[idx] = z[idx]
    return out

def db(v):
    return 10 * np.log10(v)

# ============================================================
# fig08-gridless : การกู้คืนสเปกตรัมเส้นแบบไม่ใช้กริดด้วยนอร์มอะตอม เทียบกับ OMP บนกริด
#   x_j = sum_l c_l exp(i 2 pi f_l j), j = 0..n-1  สังเกต m ตัวอย่างที่สุ่มตำแหน่ง
# ============================================================
def lsig(freqs, amps, n):
    j = np.arange(n)
    return np.exp(2j * np.pi * np.outer(j, freqs)) @ amps

def draw_freqs(k, sep, rng):
    while True:
        f = np.sort(rng.random(k))
        d = np.diff(np.concatenate([f, [f[0] + 1]]))
        if d.min() >= sep:
            return f

_ANM = {}
ANM_STATUS = []                                                           # สถานะของ SDP ทุกครั้งที่แก้ (optimal / optimal_inaccurate)
def anm_sdp(yobs, T, n):
    """แก้ปัญหานอร์มอะตอมแบบไร้สัญญาณรบกวนด้วย SDP ตาม Tang และคณะ:
       min (1/2n) tr Toep(b) + t/2  s.t. [[Toep(b), x],[x^H, t]] >= 0,  x_T = y
       สร้างปัญหาครั้งเดียวด้วยพารามิเตอร์ (หน้ากากของตำแหน่งที่สังเกต) แล้วแก้ซ้ำ"""
    import cvxpy as cp
    if n not in _ANM:
        u = cp.Variable(n, complex=True); x = cp.Variable(n, complex=True); t = cp.Variable()
        Toe = cp.real(u[0]) * np.eye(n)
        for d in range(1, n):
            Sd = np.eye(n, k=-d)                                          # สมาชิก (i+d, i) = 1
            Toe = Toe + u[d] * Sd + cp.conj(u[d]) * Sd.T
        Z = cp.bmat([[Toe, cp.reshape(x, (n, 1), order="F")], [cp.reshape(cp.conj(x), (1, n), order="F"), cp.reshape(t, (1, 1), order="F")]])
        msk = cp.Parameter(n, nonneg=True); ym = cp.Parameter(n, complex=True)
        prob = cp.Problem(cp.Minimize(cp.real(u[0]) / 2 + t / 2),
                          [Z == Z.H, Z >> 0, cp.multiply(msk, x) == ym])
        _ANM[n] = (prob, u, x, msk, ym)
    prob, u, x, msk, ym = _ANM[n]
    mv = np.zeros(n); mv[T] = 1.0
    yv = np.zeros(n, complex); yv[T] = yobs
    msk.value = mv; ym.value = yv
    prob.solve(solver="SCS", eps=1e-8, max_iters=20000)
    ANM_STATUS.append(prob.status)                                        # บันทึกสถานะของตัวแก้ทุกครั้ง
    if prob.status not in ("optimal", "optimal_inaccurate") or u.value is None or x.value is None:
        raise RuntimeError(f"ANM SDP failed: status {prob.status}")       # ไม่ใช้ผลจากตัวแก้ที่ล้มเหลว
    uv = u.value
    Tu = np.array([[uv[i - j] if i >= j else np.conj(uv[j - i]) for j in range(n)] for i in range(n)])
    Tu[np.diag_indices(n)] = np.real(uv[0])
    return np.asarray(x.value), Tu

def esprit_freqs(Tu, tol=1e-3):
    """ประมาณความถี่จากการแยกแวนเดอร์มอนด์ของ Toep(b) ด้วย ESPRIT บนปริภูมิสัญญาณ"""
    w, V = np.linalg.eigh(Tu)
    if w.max() <= 0:                                                      # Toep(b) เป็นศูนย์: ไม่มีความถี่
        return np.array([])
    r = int(np.sum(w > tol * w.max()))
    if r == 0:                                                            # ป้องกัน slice [-0:] ซึ่งเลือกทุกเวกเตอร์ลักษณะเฉพาะ
        return np.array([])
    Us = V[:, -r:]
    Phi = np.linalg.lstsq(Us[:-1], Us[1:], rcond=None)[0]
    z = np.linalg.eigvals(Phi)
    return np.sort(np.mod(np.angle(z) / (2 * np.pi), 1.0))

def omp_c(A, y, k):
    r = y.copy(); S = []
    nrm = np.linalg.norm(A, axis=0)
    for _ in range(k):
        j = int(np.argmax(np.abs(A.conj().T @ r) / nrm))
        if j in S: break
        S.append(j)
        c = np.linalg.lstsq(A[:, S], y, rcond=None)[0]; r = y - A[:, S] @ c
    return S, c

def omp_grid(yobs, T, n, k, r):
    grid = np.arange(r * n) / (r * n)
    Afull = np.exp(2j * np.pi * np.outer(np.arange(n), grid))
    S, c = omp_c(Afull[T], yobs, k)
    return Afull[:, S] @ c, grid[S], c

def fdist(a, b):
    d = np.abs(a - b)
    return np.minimum(d, 1 - d)

if want("gridless"):
    n, k = 64, 5
    sep = 1.0 / np.floor((n - 1) / 4)
    rng = np.random.default_rng(80)
    # (ก) ตัวอย่างหนึ่งกรณี
    f0 = draw_freqs(k, sep, rng)
    a0 = (1 + 0.5 * np.abs(rng.standard_normal(k))) * np.exp(2j * np.pi * rng.random(k))
    x0 = lsig(f0, a0, n)
    m0 = 32
    T0 = np.sort(rng.choice(n, m0, replace=False))
    t1 = time.time()
    xa, Tu = anm_sdp(x0[T0], T0, n)
    t_sdp = time.time() - t1
    fa = esprit_freqs(Tu)
    xg1, fg1, cg1 = omp_grid(x0[T0], T0, n, k, 1)
    xg4, fg4, cg4 = omp_grid(x0[T0], T0, n, k, 4)
    print(f"gridless: n = {n}, k = {k}, minimum separation {sep:.4f} = 1/floor((n-1)/4), m = {m0} random samples, noiseless")
    print(f"   true frequencies      {np.round(f0, 4)}")
    print(f"   ANM (ESPRIT on Toep)  {np.round(fa, 4)}   max error {max(fdist(fa[i], f0[i]) for i in range(k)) if len(fa)==k else float('nan'):.2e}   SDP time {t_sdp:.1f} s")
    print(f"   OMP grid r=1          {np.round(np.sort(fg1), 4)}")
    print(f"   OMP grid r=4          {np.round(np.sort(fg4), 4)}")
    for nm, xr in [("ANM", xa), ("OMP r=1", xg1), ("OMP r=4", xg4)]:
        print(f"   {nm:8s} NMSE of full signal {db(np.sum(np.abs(xr - x0)**2) / np.sum(np.abs(x0)**2)):7.1f} dB")
    # (ข) NMSE เทียบกับจำนวนตัวอย่าง
    ms = [12, 16, 20, 24, 32, 40]
    trials = 15
    res = {"ANM": [], "OMP r=1": [], "OMP r=4": [], "OMP r=16": []}
    succ = {kk: [] for kk in res}
    t1 = time.time()
    for m in ms:
        acc = {kk: [] for kk in res}
        for tr in range(trials):
            rr = np.random.default_rng(1000 * m + tr)
            f = draw_freqs(k, sep, rr)
            a = (1 + 0.5 * np.abs(rr.standard_normal(k))) * np.exp(2j * np.pi * rr.random(k))
            x = lsig(f, a, n)
            T = np.sort(rr.choice(n, m, replace=False))
            xr, _ = anm_sdp(x[T], T, n)
            acc["ANM"].append(np.sum(np.abs(xr - x)**2) / np.sum(np.abs(x)**2))
            for r in (1, 4, 16):
                xg, _, _ = omp_grid(x[T], T, n, k, r)
                acc[f"OMP r={r}"].append(np.sum(np.abs(xg - x)**2) / np.sum(np.abs(x)**2))
        for kk in res:
            v = np.array(acc[kk])
            res[kk].append(db(np.median(v)))
            succ[kk].append(np.mean(v < 1e-4))
        print(f"   ... m = {m} done ({time.time()-t1:.0f} s)", flush=True)
    print(f"gridless: NMSE (median over {trials} trials, dB) and success rate (NMSE < -40 dB) versus m   [{time.time()-t1:.0f} s]")
    print(f"   SDP solver status over {len(ANM_STATUS)} solves: " + ", ".join(f"{s_} {ANM_STATUS.count(s_)}" for s_ in sorted(set(ANM_STATUS))))
    print("   m           " + " ".join(f"{m:6d}" for m in ms))
    for kk in res:
        print(f"   {kk:10s}  " + " ".join(f"{v:6.1f}" for v in res[kk]) + "   | " + " ".join(f"{v:4.2f}" for v in succ[kk]))
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.45))
    ax = axes[0]
    ax.vlines(f0, 0, np.abs(a0), color=INK, lw=1.0)
    ax.plot(f0, np.abs(a0), "o", color=INK, ms=4, mfc="white", label="true")
    ax.plot(fg1, np.abs(cg1), "s", color=GRY, ms=3.5, label="OMP, grid $r=1$")
    ax.plot(fg4, np.abs(cg4), "^", color=ORG, ms=3.5, label="OMP, grid $r=4$")
    if len(fa) == k:
        ca = np.linalg.lstsq(np.exp(2j * np.pi * np.outer(T0, fa)), x0[T0], rcond=None)[0]
        ax.plot(fa, np.abs(ca), "x", color=ACC, ms=5, mew=1.2, label="atomic norm")
    ax.set_xlim(0, 1); ax.set_ylim(0, 2.6)
    ax.set_xlabel("frequency $f$"); ax.set_ylabel("amplitude $|c|$")
    ax.set_title("(a)", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", ncol=2)
    ax = axes[1]
    sty = {"ANM": ("o-", ACC, "atomic norm"), "OMP r=1": ("s-", GRY, "OMP, $r=1$"),
           "OMP r=4": ("^-", ORG, "OMP, $r=4$"), "OMP r=16": ("v-", RED, "OMP, $r=16$")}
    for kk in res:
        s, c, lb = sty[kk]
        ax.plot(ms, res[kk], s, color=c, ms=3, label=lb)
    ax.set_xlabel("number of samples $m$"); ax.set_ylabel("median NMSE (dB)")
    ax.set_title("(b)", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig08-gridless")

# ============================================================
# fig08-onebit : การรับรู้เชิงบีบอัดแบบหนึ่งบิต
#   (ก) ความคลาดเคลื่อนเชิงทิศทางเทียบกับ m/n ของตัวประมาณฉายกลับ + H_k และ BIHT
#   (ข) งบประมาณบิตคงที่: จำนวนบิตต่อการวัดที่เหมาะสมขึ้นกับ SNR ก่อนการควอนไทซ์
# ============================================================
def unit(v):
    return v / max(np.linalg.norm(v), 1e-300)

def passive_1bit(Phi, s, k):
    m = Phi.shape[0]
    return unit(hard_k(Phi.T @ s / m, k))

def biht(Phi, s, k, iters=200):
    m = Phi.shape[0]
    tau = np.sqrt(np.pi / 2) / m
    x = passive_1bit(Phi, s, k)
    for _ in range(iters):
        x = hard_k(x + 0.5 * tau * Phi.T @ (s - np.sign(Phi @ x)), k)
        x = unit(x)
    return x

def quantize(v, b, vmax):
    """ตัวควอนไทซ์เอกรูปแบบ midrise ที่มี 2^b ระดับบนช่วง [-vmax, vmax] (ค่าที่เกินช่วงถูกตัด)"""
    L = 2 ** b; d = 2 * vmax / L
    q = np.floor(np.clip(v, -vmax, vmax - 1e-12) / d) + 0.5
    return q * d

def iht_k(Phi, y, k, iters=300):
    L = np.linalg.norm(Phi, 2) ** 2
    x = np.zeros(Phi.shape[1])
    for _ in range(iters):
        x = hard_k(x + Phi.T @ (y - Phi @ x) / L, k)
    S = np.flatnonzero(x)
    if S.size:                                                       # ประมาณกำลังสองน้อยที่สุดบนตัวรองรับ (HTP หนึ่งขั้น)
        x = np.zeros_like(x); x[S] = np.linalg.lstsq(Phi[:, S], y, rcond=None)[0]
    return x

if want("onebit"):
    n, k, trials = 256, 8, 100
    ratios = [0.5, 1, 2, 4, 8, 16]
    E = {"passive": [], "BIHT": []}
    t1 = time.time()
    for rt in ratios:
        m = int(rt * n)
        acc = {kk: [] for kk in E}
        for tr in range(trials):
            rr = np.random.default_rng(5000 + 37 * m + tr)
            x = np.zeros(n); S = rr.choice(n, k, replace=False); x[S] = rr.standard_normal(k); x = unit(x)
            Phi = rr.standard_normal((m, n))
            s = np.sign(Phi @ x)
            acc["passive"].append(np.linalg.norm(passive_1bit(Phi, s, k) - x))
            acc["BIHT"].append(np.linalg.norm(biht(Phi, s, k) - x))
        for kk in E:
            E[kk].append(np.mean(acc[kk]))
    print(f"onebit: n = {n}, k = {k}, Gaussian Phi, ||x||_2 = 1, noiseless sign measurements, {trials} trials   [{time.time()-t1:.0f} s]")
    print("   m/n            " + " ".join(f"{v:7.1f}" for v in ratios))
    for kk in E:
        print(f"   {kk:8s} error  " + " ".join(f"{v:7.4f}" for v in E[kk]))
    sl = np.polyfit(np.log(ratios[2:]), np.log(E["passive"][2:]), 1)[0]
    sl2 = np.polyfit(np.log(ratios[2:]), np.log(E["BIHT"][2:]), 1)[0]
    print(f"   log-log slope for m/n >= 2: passive {sl:.2f} (theory -0.5), BIHT {sl2:.2f}")
    # (ข) งบประมาณบิต
    B = 4 * n
    bits = [1, 2, 3, 4, 6, 8]
    snrs = [40, 10]
    Q = {s_: [] for s_ in snrs}
    t1 = time.time()
    for snr in snrs:
        for b in bits:
            m = B // b
            errs = []
            for tr in range(trials):
                rr = np.random.default_rng(9000 + 11 * b + tr + snr)
                x = np.zeros(n); S = rr.choice(n, k, replace=False); x[S] = rr.standard_normal(k); x = unit(x)
                Phi = rr.standard_normal((m, n))
                z = Phi @ x + 10 ** (-snr / 20) * rr.standard_normal(m)     # ||x|| = 1 จึงให้กำลังของ Phi x เท่ากับหนึ่ง
                if b == 1:
                    xh = biht(Phi, np.sign(z), k)
                else:
                    xh = unit(iht_k(Phi, quantize(z, b, 3.0), k))
                errs.append(np.linalg.norm(xh - x))
            Q[snr].append(np.mean(errs))
    print(f"onebit: fixed budget B = {B} bits (B/n = {B//n}), b bits per measurement so m = B/b, uniform quantizer on [-3, 3]   [{time.time()-t1:.0f} s]")
    print("   b                 " + " ".join(f"{v:7d}" for v in bits))
    print("   m                 " + " ".join(f"{B//v:7d}" for v in bits))
    for snr in snrs:
        print(f"   SNR {snr:2d} dB error  " + " ".join(f"{v:7.4f}" for v in Q[snr]) + f"   best b = {bits[int(np.argmin(Q[snr]))]}")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.45))
    ax = axes[0]
    ax.loglog(ratios, E["passive"], "s-", color=GRY, ms=3, label="back-projection + $H_k$")
    ax.loglog(ratios, E["BIHT"], "o-", color=ACC, ms=3, label="BIHT")
    ref = E["passive"][2] * (np.array(ratios) / ratios[2]) ** -0.5
    ax.loglog(ratios, ref, ":", color=INK, lw=0.9, label=r"slope $-1/2$")
    ax.set_xlabel("$m/n$"); ax.set_ylabel(r"$\|\hat{x}-x\|_2$  ($\|x\|_2=1$)")
    ax.set_title("(a)", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=6.5)
    ax = axes[1]
    for snr, st in [(40, ("o-", ACC)), (10, ("s-", RED))]:
        ax.semilogy(bits, Q[snr], st[0], color=st[1], ms=3, label=f"SNR {snr} dB")
    ax.set_xlabel("bits per measurement $b$  ($m = B/b$)"); ax.set_ylabel("error")
    ax.set_title("(b)", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig08-onebit")

# ============================================================
# fig08-pnp : ADMM แบบเสียบแล้วใช้ (PnP) สำหรับ MRI
#   ใช้ตัวดำเนินการ แฟนทอม ภาพสนามสุ่ม หน้ากาก และ ADMM ของบทที่ 6 โดยตรง
#   ตัวกำจัดสัญญาณรบกวนคือ non-local means ของ scikit-image (สัญญาอนุญาต BSD) เพื่อให้รูปทุกรูปสร้างจากซอฟต์แวร์ที่อนุญาตการใช้งานทุกประเภท
# ============================================================
def load_ch6():
    spec = importlib.util.spec_from_file_location("fig06", os.path.join(HERE, "fig06_generate.py"))
    mod = importlib.util.module_from_spec(spec)
    argv = sys.argv; sys.argv = ["fig06_generate.py", "__none__"]
    spec.loader.exec_module(mod)
    sys.argv = argv
    return mod

def nlm(z, s):
    from skimage.restoration import denoise_nl_means
    return denoise_nl_means(z, patch_size=5, patch_distance=6, h=0.8 * s, sigma=s, fast_mode=True)

def pnp_admm_mri(c6, y, M, denoise, sigma_d, rho=1.0, iters=100, x_true=None, sigma0=None, decay=0.85):
    """PnP-ADMM: ขั้นปรับปรุง x แม่นตรงตามบทตั้งของบทที่ 6, ขั้นปรับปรุง v คือ denoise(x + u, sigma_t)
       sigma0 = None: ระดับคงที่ sigma_t = sigma_d ; มิฉะนั้นใช้การต่อเนื่อง sigma_t = max(sigma_d, sigma0 decay^t)"""
    Mf = M.astype(float)
    chib = 0.5 * (Mf + c6.flip_freq(Mf))
    ybar = c6.herm(np.where(M, y, 0).astype(complex))
    x = c6.Ft(np.where(M, y, 0)); v = x.copy(); u = np.zeros_like(x)
    hist, dif = [], []
    for t_ in range(iters):
        x_old = x
        x = c6.Ft((ybar + rho * c6.F(v - u)) / (chib + rho))
        sd = sigma_d if sigma0 is None else max(sigma_d, sigma0 * decay ** t_)
        v = denoise(x + u, sd)
        u = u + x - v
        dif.append(np.linalg.norm(x - x_old) / np.linalg.norm(x))
        if x_true is not None:
            hist.append(c6.psnr(np.clip(v, 0, 1), x_true))
    return np.clip(v, 0, 1), hist, dif

if want("pnp"):
    c6 = load_ch6()
    n, R = 128, 4
    M = c6.mask_vardens(n, R, np.random.default_rng(3))
    imgs = {"phantom": c6.phantom(n), "textured": c6.textured_image(n)}
    SIGS = [0.005, 0.01, 0.02, 0.04]
    print(f"pnp: {n}x{n} images, n/m = {R}, variable-density rows, k-space noise sigma = {c6.SIG_K}; all parameters tuned on a grid (oracle)")
    print("     denoiser: non-local means (scikit-image, patch 5, distance 6, h = 0.8 sigma); rho = 1; 100 iterations; wavelet and TV baselines solved to convergence (ch. 6 criterion)")
    print("     PnP-fixed: sigma_t = sigma_d constant; PnP-cont.: sigma_t = max(sigma_d, 0.2 x 0.85^t)")
    panels, hists = {}, {}
    t1 = time.time()
    for nm, img in imgs.items():
        y = c6.acquire(img, M, 21)
        zf = np.clip(c6.Ft(y), 0, 1)
        bw = c6.tune(y, M, img, "wavelet"); bt = c6.tune(y, M, img, "TV")
        best = bc = None
        for sd in SIGS:
            r_, h_, d_ = pnp_admm_mri(c6, y, M, nlm, sd, x_true=img)
            if best is None or c6.psnr(r_, img) > best[0]:
                best = (c6.psnr(r_, img), r_, sd, h_, d_)
            r_, h_, d_ = pnp_admm_mri(c6, y, M, nlm, sd, x_true=img, sigma0=0.2)
            if bc is None or c6.psnr(r_, img) > bc[0]:
                bc = (c6.psnr(r_, img), r_, sd, h_, d_)
        panels[nm] = (img, zf, bw[1], bt[1], bc[1]); hists[nm] = (best[3], bc[3])
        print(f"   {nm:9s} zero-filled {c6.psnr(zf, img):5.2f}  wavelet {bw[0]:5.2f} (lam_w {bw[2]:.0e})  TV {bt[0]:5.2f} (lam_t {bt[3]:.0e})  "
              f"PnP-fixed {best[0]:5.2f} (sigma_d {best[2]})  PnP-cont. {bc[0]:5.2f} dB (sigma_end {bc[2]})")
        print(f"   {nm:9s} PSNR after 10 / 20 / 50 / 100 iterations: fixed {best[3][9]:.2f} / {best[3][19]:.2f} / {best[3][49]:.2f} / {best[3][99]:.2f};"
              f"  cont. {bc[3][9]:.2f} / {bc[3][19]:.2f} / {bc[3][49]:.2f} / {bc[3][99]:.2f}")
        print(f"   {nm:9s} relative change of x at the last iteration: fixed {best[4][-1]:.1e}, cont. {bc[4][-1]:.1e}")
        if nm == "phantom":
            # การตรวจรหัส: PnP ที่ใช้ตัวดำเนินการใกล้เคียงของ lam_t TV เป็นตัวกำจัดสัญญาณรบกวนต้องให้ผลเท่ากับ ADMM ของ TV (ปัญหาที่ 8.4)
            lt = 3e-3                                                 # ค่าคงที่สำหรับการตรวจรหัส (ทั้งสองวิธีลู่เข้าได้เร็วที่ค่านี้)
            r_tv, _, _ = pnp_admm_mri(c6, y, M, lambda z, s_: c6.tv_denoise(z, lt, iters=200), 0.0, iters=800)
            r_ref, _ = c6.admm_mri(y, M, lam_t=lt, tol=1e-7, max_iters=20000); it_ref = c6.admm_mri.last_iters
            r_200, _ = c6.admm_mri(y, M, lam_t=lt, iters=200)
            print(f"   check at lam_t = {lt:.0e}: PnP with prox_(lam_t TV), 800 iterations {c6.psnr(r_tv, img):.2f} dB vs converged TV-ADMM "
                  f"({it_ref} iterations, eps_rel 1e-7) {c6.psnr(r_ref, img):.2f} dB (TV-ADMM with fixed rho = 0.5, 200 iterations {c6.psnr(r_200, img):.2f} dB), "
                  f"||difference|| / ||image|| = {np.linalg.norm(r_tv - r_ref) / np.linalg.norm(r_ref):.1e}")
    print(f"   [{time.time()-t1:.0f} s]")
    fig = plt.figure(figsize=(6.6, 3.9))
    gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 0.95])
    titles = ["truth", "zero-filled", "wavelet $\\ell_1$", "TV", "PnP-NLM"]
    for i, nm in enumerate(["phantom", "textured"]):
        for j in range(5):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(panels[nm][j], vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f"({'abcde'[j]}) {titles[j]}", fontsize=7.5)
    ax = fig.add_subplot(gs[2, :])
    for nm, col in [("phantom", ACC), ("textured", RED)]:
        hf, hc = hists[nm]
        ax.plot(range(1, 101), hc, "-", color=col, label=f"{nm}, continuation")
        ax.plot(range(1, 101), hf, "--", color=col, alpha=0.6, label=f"{nm}, fixed $\\sigma$")
    ax.set_xlabel("iteration"); ax.set_ylabel("PSNR (dB)")
    ax.set_title("(f)", loc="left", fontsize=8)
    ax.legend(frameon=False, fontsize=6.3, ncol=2, loc="lower right")
    fig.tight_layout(h_pad=0.4, w_pad=0.2)
    save(fig, "fig08-pnp")

# ============================================================
# fig08-unroll : การคลี่ขั้นตอนวิธี ISTA เป็นโครงข่าย (LISTA-CP, ALISTA) เทียบกับ ISTA, FISTA และ STA-MPI
#   ฝึกด้วย autograd (numpy) และ Adam ที่เขียนเอง
# ============================================================
def alista_weight(Phi):
    """ผลเฉลยรูปปิดของ min ||W^T Phi||_F^2 s.t. diag(W^T Phi) = 1 : W = (Phi^+)^T diag(1/l), l_i = (Phi^+ Phi)_ii"""
    Pp = np.linalg.pinv(Phi)
    lev = np.sum(Pp * Phi.T, axis=1)                                   # สมาชิกแนวทแยงของ Phi^+ Phi
    return Pp.T / lev[None, :], lev, Pp

def ista_path(Phi, Y, lam, T, keep=True):
    """คืนรายการของผลทุกรอบเมื่อ keep=True หรือคืนเฉพาะผลรอบสุดท้าย (ประหยัดหน่วยความจำสำหรับการวนซ้ำยาว)"""
    L = np.linalg.norm(Phi, 2) ** 2
    X = np.zeros((Phi.shape[1], Y.shape[1])); out = []
    for t_ in range(T):
        X = soft(X + Phi.T @ (Y - Phi @ X) / L, lam / L)
        if keep:
            out.append(X)
    return out if keep else [X]

def fista_path(Phi, Y, lam, T):
    L = np.linalg.norm(Phi, 2) ** 2
    X = np.zeros((Phi.shape[1], Y.shape[1])); Z = X.copy(); tk = 1.0; out = []
    for _ in range(T):
        Xn = soft(Z + Phi.T @ (Y - Phi @ Z) / L, lam / L)
        tn = (1 + np.sqrt(1 + 4 * tk ** 2)) / 2
        Z = Xn + (tk - 1) / tn * (Xn - X); X = Xn; tk = tn; out.append(X)
    return out

def stampi_path(Phi, Y, gam, T, Pp):
    """STA-MPI ตามขั้นตอนวิธีของบทที่ 5: x <- S_theta(x + Phi^+ (y - Phi x)) แล้ว theta <- gam theta
       เริ่มที่ theta_0 = ||Phi^+ y||_inf  (ไม่ใช้เกณฑ์หยุด k_max เพื่อเปรียบเทียบที่จำนวนรอบเท่ากัน)"""
    X = np.zeros((Phi.shape[1], Y.shape[1])); th = np.max(np.abs(Pp @ Y), axis=0, keepdims=True); out = []
    for _ in range(T):
        X = soft(X + Pp @ (Y - Phi @ X), th); out.append(X)
        th = th * gam
    return out

def nmse_db(Xh, X):
    return db(np.sum((Xh - X) ** 2) / np.sum(X ** 2))

def bern_gauss(n, N, p, rng):
    return (rng.random((n, N)) < p) * rng.standard_normal((n, N))

if want("unroll"):
    import autograd.numpy as anp
    from autograd import grad as agrad
    n, m, T, p, snr = 200, 100, 16, 0.1, 40.0
    rng = np.random.default_rng(88)
    Phi = rng.standard_normal((m, n)) / np.sqrt(m)
    Phi = Phi / np.linalg.norm(Phi, axis=0)
    sig = lambda Xc: np.sqrt(np.mean((Phi @ Xc) ** 2) * 10 ** (-snr / 10))
    def make(N, pp, seed):
        r = np.random.default_rng(seed)
        X = bern_gauss(n, N, pp, r)
        Y = Phi @ X
        return X, Y + sig(X) * r.standard_normal(Y.shape)
    Xv, Yv = make(1000, p, 2)                                            # ชุดตรวจสอบสำหรับปรับพารามิเตอร์ของวิธีดั้งเดิม
    Xte, Yte = make(2000, p, 3)                                          # ชุดทดสอบ
    W_al, lev, Pp = alista_weight(Phi)
    # ตรวจรูปปิดของ W: เงื่อนไขทแยงและความสอดคล้องร่วมแบบทั่วไป
    G = W_al.T @ Phi
    off = G - np.diag(np.diag(G))
    G0 = Phi.T @ Phi; off0 = G0 - np.diag(np.diag(G0))
    # เปรียบเทียบกับการลดนอร์มโฟรเบนิอุสเชิงตัวเลข (เกรเดียนต์ฉาย) จากจุดเริ่ม Phi
    Wn = Phi.copy()
    PPt = Phi @ Phi.T
    step = 1.0 / np.linalg.norm(PPt, 2)
    for _ in range(3000):
        Wn = Wn - step * (PPt @ Wn)
        Wn = Wn + Phi * ((1 - np.sum(Wn * Phi, axis=0)) / np.sum(Phi * Phi, axis=0))[None, :]
    print(f"unroll: n = {n}, m = {m}, Gaussian Phi with unit-norm columns, Bernoulli-Gaussian x (p = {p}), SNR {snr:.0f} dB, T = {T} layers")
    print(f"   closed-form W: max |diag(W^T Phi) - 1| = {np.max(np.abs(np.diag(G) - 1)):.1e}; ||W^T Phi||_F^2 = {np.sum(G**2):.2f} "
          f"(projected gradient {np.sum((Wn.T @ Phi)**2):.2f}, ||W - W_pg||/||W|| = {np.linalg.norm(W_al - Wn)/np.linalg.norm(W_al):.1e})")
    print(f"   leverage scores l_i = (Phi^+ Phi)_ii: mean {lev.mean():.3f} (m/n = {m/n:.3f}), min {lev.min():.3f}, max {lev.max():.3f}")
    print(f"   generalized coherence max_(i!=j) |w_i^T phi_j|: W = Phi {np.max(np.abs(off0)):.3f}, W = closed form {np.max(np.abs(off)):.3f}")
    # ปรับพารามิเตอร์ของวิธีดั้งเดิมบนชุดตรวจสอบ ที่ T ชั้น
    lams = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
    lam_i = min(lams, key=lambda l: nmse_db(ista_path(Phi, Yv, l, T)[-1], Xv))
    lam_f = min(lams, key=lambda l: nmse_db(fista_path(Phi, Yv, l, T)[-1], Xv))
    gams = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    gam_s = min(gams, key=lambda g: nmse_db(stampi_path(Phi, Yv, g, T, Pp)[-1], Xv))
    print(f"   tuned on validation at T = {T}: ISTA lambda {lam_i}, FISTA lambda {lam_f}, STA-MPI gamma_h {gam_s}")
    curves = {
        "ISTA": [nmse_db(Xh, Xte) for Xh in ista_path(Phi, Yte, lam_i, T)],
        "FISTA": [nmse_db(Xh, Xte) for Xh in fista_path(Phi, Yte, lam_f, T)],
        "STA-MPI": [nmse_db(Xh, Xte) for Xh in stampi_path(Phi, Yte, gam_s, T, Pp)],
    }
    lam_long = min([0.005, 0.01, 0.02, 0.05, 0.1], key=lambda l: nmse_db(ista_path(Phi, Yv, l, 1000, keep=False)[-1], Xv))
    print(f"   ISTA run to 1000 iterations (lambda tuned separately on validation: {lam_long}): "
          f"{nmse_db(ista_path(Phi, Yte, lam_long, 100, keep=False)[-1], Xte):.1f} dB after 100, "
          f"{nmse_db(ista_path(Phi, Yte, lam_long, 1000, keep=False)[-1], Xte):.1f} dB after 1000 iterations")

    # ---- โครงข่ายแบบคลี่ ----
    L0 = np.linalg.norm(Phi, 2) ** 2
    Phi_a = anp.array(Phi)
    def sft(z, t):
        return anp.sign(z) * anp.maximum(anp.abs(z) - t, 0.0)
    def fwd_alista(par, Y, all_layers=False):
        X = anp.zeros((n, Y.shape[1])); outs = []
        for t in range(T):
            X = sft(X + par["g"][t] * anp.dot(W_al.T, Y - anp.dot(Phi_a, X)), anp.exp(par["s"][t]))
            outs.append(X)
        return outs if all_layers else X
    def fwd_lista(par, Y, all_layers=False):
        X = anp.zeros((n, Y.shape[1])); outs = []
        for t in range(T):
            X = sft(X + anp.dot(par["W"][t].T, Y - anp.dot(Phi_a, X)), anp.exp(par["s"][t]))
            outs.append(X)
        return outs if all_layers else X
    def loss_fn(fwd):
        def f(par, X, Y):
            outs = fwd(par, Y, all_layers=True)
            tot = 0.0
            for t, Xh in enumerate(outs):                               # การกำกับทุกชั้น โดยถ่วงชั้นท้ายมากกว่า
                tot = tot + (t + 1) * anp.sum((Xh - X) ** 2)
            return tot / (anp.sum(X ** 2) * T * (T + 1) / 2)
        return f
    def adam(par, lossf, steps, lr, seed, batch=256):
        gfun = agrad(lossf)
        mom = {k_: [np.zeros_like(v) for v in par[k_]] if isinstance(par[k_], list) else np.zeros_like(par[k_]) for k_ in par}
        vel = {k_: [np.zeros_like(v) for v in par[k_]] if isinstance(par[k_], list) else np.zeros_like(par[k_]) for k_ in par}
        r = np.random.default_rng(seed)
        for it in range(1, steps + 1):
            X = bern_gauss(n, batch, p, r); Y = Phi @ X
            Y = Y + sig(X) * r.standard_normal(Y.shape)
            g = gfun(par, X, Y)
            for k_ in par:
                if isinstance(par[k_], list):
                    for j in range(len(par[k_])):
                        mom[k_][j] = 0.9 * mom[k_][j] + 0.1 * g[k_][j]
                        vel[k_][j] = 0.999 * vel[k_][j] + 0.001 * g[k_][j] ** 2
                        par[k_][j] = par[k_][j] - lr * (mom[k_][j] / (1 - 0.9 ** it)) / (np.sqrt(vel[k_][j] / (1 - 0.999 ** it)) + 1e-8)
                else:
                    mom[k_] = 0.9 * mom[k_] + 0.1 * g[k_]
                    vel[k_] = 0.999 * vel[k_] + 0.001 * g[k_] ** 2
                    par[k_] = par[k_] - lr * (mom[k_] / (1 - 0.9 ** it)) / (np.sqrt(vel[k_] / (1 - 0.999 ** it)) + 1e-8)
        return par
    t1 = time.time()
    par_al = {"g": np.full(T, 1.0), "s": np.full(T, np.log(lam_i / L0 * 2))}
    # ค่าเริ่มของ ALISTA: ขนาดก้าว 1 กับ W ที่ diag(W^T Phi) = 1 และขีดเริ่มเปลี่ยนเท่ากับ ISTA
    par_al = adam(par_al, loss_fn(fwd_alista), 3000, 0.01, 7)
    t_al = time.time() - t1
    t1 = time.time()
    par_li = {"W": [Phi / L0 for _ in range(T)], "s": np.full(T, np.log(lam_i / L0))}
    par_li = adam(par_li, loss_fn(fwd_lista), 3000, 0.001, 8)
    t_li = time.time() - t1
    curves["ALISTA"] = [nmse_db(np.asarray(Xh), Xte) for Xh in fwd_alista(par_al, Yte, True)]
    curves["LISTA-CP"] = [nmse_db(np.asarray(Xh), Xte) for Xh in fwd_lista(par_li, Yte, True)]
    nparam = {"ALISTA": 2 * T, "LISTA-CP": T * (m * n + 1)}
    print(f"   training: Adam, batch 256, 3000 steps each, fresh samples every step; ALISTA {t_al:.0f} s ({nparam['ALISTA']} parameters), "
          f"LISTA-CP {t_li:.0f} s ({nparam['LISTA-CP']} parameters)")
    print(f"   learned ALISTA step sizes gamma_t: {np.round(par_al['g'], 3)}")
    print(f"   learned ALISTA thresholds theta_t: {np.round(np.exp(par_al['s']), 4)}")
    print("   NMSE (dB) on test set at layer t = 1, 2, 4, 8, 12, 16")
    for kk, cv in curves.items():
        print(f"   {kk:9s} " + " ".join(f"{cv[t-1]:7.1f}" for t in [1, 2, 4, 8, 12, 16]))
    # ---- ความทนทานต่อการเปลี่ยนการแจกแจง: ระดับความเบาบางของชุดทดสอบเปลี่ยนไป ----
    ps = [0.05, 0.10, 0.15, 0.20, 0.25]
    rob = {kk: [] for kk in ["ISTA (1000 it.)", "FISTA", "STA-MPI", "ALISTA", "LISTA-CP"]}
    for pt in ps:
        Xs, Ys = make(2000, pt, 40 + int(100 * pt))
        rob["ISTA (1000 it.)"].append(nmse_db(ista_path(Phi, Ys, lam_long, 1000, keep=False)[-1], Xs))
        rob["FISTA"].append(nmse_db(fista_path(Phi, Ys, lam_f, T)[-1], Xs))
        rob["STA-MPI"].append(nmse_db(stampi_path(Phi, Ys, gam_s, T, Pp)[-1], Xs))
        rob["ALISTA"].append(nmse_db(np.asarray(fwd_alista(par_al, Ys)), Xs))
        rob["LISTA-CP"].append(nmse_db(np.asarray(fwd_lista(par_li, Ys)), Xs))
    print("   distribution shift: NMSE (dB) at T = 16 when the test sparsity p differs from training p = 0.10")
    print("   p              " + " ".join(f"{v:7.2f}" for v in ps))
    for kk, cv in rob.items():
        print(f"   {kk:15s}" + " ".join(f"{v:7.1f}" for v in cv))
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
    ax = axes[0]
    sty = {"ISTA": ("-", GRY), "FISTA": ("-", ORG), "STA-MPI": ("-", GRN), "ALISTA": ("-", ACC), "LISTA-CP": ("--", RED)}
    for kk, cv in curves.items():
        ax.plot(range(1, T + 1), cv, sty[kk][0], color=sty[kk][1], marker="o", ms=2.2, label=kk)
    ax.set_xlabel("layer / iteration $t$"); ax.set_ylabel("NMSE (dB)")
    ax.set_title("(a)", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=6.5)
    ax = axes[1]
    sty2 = {"ISTA (1000 it.)": ("s:", GRY), "FISTA": ("^-", ORG), "STA-MPI": ("v-", GRN), "ALISTA": ("o-", ACC), "LISTA-CP": ("d--", RED)}
    for kk, cv in rob.items():
        ax.plot(ps, cv, sty2[kk][0], color=sty2[kk][1], ms=3, label=kk)
    ax.axvline(p, color=INK, lw=0.6, ls=":")
    ax.set_xlabel("test sparsity $p$ (training $p = 0.1$)"); ax.set_ylabel("NMSE at $t = 16$ (dB)")
    ax.set_title("(b)", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=6.0)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig08-unroll")

print("DONE")
