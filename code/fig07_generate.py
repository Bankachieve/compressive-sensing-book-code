# -*- coding: utf-8 -*-
r"""สร้างรูปประกอบและผลการทดลองเชิงตัวเลขทั้งหมดของบทที่ 7 -> figures/fig07-*.pdf
ส่วนที่เลือกรันได้: channel spectrum doa fdia  เช่น `python fig07_generate.py doa fdia`
ข้อมูลระบบไฟฟ้ากำลังสร้างจากกรณีทดสอบ IEEE 9 บัสและ 14 บัสของ PYPOWER (แบบจำลองกระแสตรง)
ตัวเลขทุกค่าที่อ้างในเนื้อหาและคำบรรยายภาพพิมพ์ออกทางจอเมื่อรันสคริปต์นี้"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle  # กริดพื้นหลังและตำแหน่ง legend แบบเดียวกันทั้งเล่ม
import os, sys, time, warnings
warnings.filterwarnings("ignore")

RUN = set(sys.argv[1:])
def want(name):
    return (not RUN) or (name in RUN)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT, exist_ok=True)
INK, ACC, GRY, RED, GRN, ORG = "#1F2A37", "#1D4ED8", "#9AA3AF", "#B91C1C", "#047857", "#B45309"
plt.rcParams.update({
    "font.size": 9, "axes.linewidth": 0.6, "axes.edgecolor": INK,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.labelcolor": INK, "lines.linewidth": 1.3,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf")
    plt.close(fig)
    print("saved", name)

# ============================================================
# ขั้นตอนวิธีพื้นฐานที่ใช้ร่วมกัน (เขียนตามบทที่ 4 และ 5)
# ============================================================
def omp(A, y, k=None, eps=None, kmax=None):
    """OMP เชิงซ้อน หยุดเมื่อครบ k อะตอม หรือเมื่อ ||r||_2 <= eps"""
    m, n = A.shape; kmax = kmax or m
    r = y.copy(); S = []; x = np.zeros(n, complex)
    nrm = np.linalg.norm(A, axis=0)
    while True:
        if k is not None and len(S) >= k: break
        if eps is not None and np.linalg.norm(r) <= eps: break
        if len(S) >= kmax: break
        j = int(np.argmax(np.abs(A.conj().T @ r) / nrm))
        if j in S: break
        S.append(j)
        c = np.linalg.lstsq(A[:, S], y, rcond=None)[0]; r = y - A[:, S] @ c
    if S:
        x[S] = c
    return x, S

def somp(A, Y, k):
    """simultaneous OMP: เลือกแถวที่ ||A_j^H R||_2 มากที่สุด"""
    R = Y.copy(); S = []
    nrm = np.linalg.norm(A, axis=0)
    for _ in range(k):
        j = int(np.argmax(np.linalg.norm(A.conj().T @ R, axis=1) / nrm))
        if j in S: break
        S.append(j)
        C = np.linalg.lstsq(A[:, S], Y, rcond=None)[0]; R = Y - A[:, S] @ C
    return S

GAP_TOL = 1e-6          # ช่องว่างคู่กันสัมพัทธ์ที่ใช้หยุด FISTA (แก้ปัญหาจนลู่เข้า)
L21_ITERS = []          # จำนวนรอบที่ใช้จริงของทุกการเรียก fista_l21

def fista_l21(A, Y, lam, iters=None, tol=GAP_TOL, max_iters=200000, check=10):
    """min 1/2 ||Y - A X||_F^2 + lam ||X||_{2,1} (กลุ่มคือแถวของ X)
       iters ไม่เป็น None: ทำซ้ำตายตัวตามจำนวนนั้น (ใช้เทียบกับการหยุดก่อนลู่เข้าเท่านั้น)
       มิฉะนั้น: เริ่มโมเมนตัมใหม่ตามเงื่อนไขเกรเดียนต์ (O'Donoghue และ Candes) และหยุดเมื่อช่องว่างคู่กันสัมพัทธ์ไม่เกิน tol
       โดยจุดคู่กัน U = R min(1, lam / max_i ||(A^H R)_i||_2) จากเศษเหลือ R = Y - A X"""
    Lc = np.linalg.norm(A, 2) ** 2
    X = np.zeros((A.shape[1], Y.shape[1]), complex); V = X.copy(); t = 1.0
    n_it = iters if iters is not None else max_iters
    for it in range(1, n_it + 1):
        Z = V - A.conj().T @ (A @ V - Y) / Lc
        nr = np.linalg.norm(Z, axis=1, keepdims=True)
        Xn = Z * np.maximum(1 - (lam / Lc) / np.maximum(nr, 1e-15), 0)
        if iters is None and np.real(np.vdot(V - Xn, Xn - X)) > 0:
            t = 1.0
        tn = (1 + np.sqrt(1 + 4 * t * t)) / 2
        V = Xn + (t - 1) / tn * (Xn - X); X, t = Xn, tn
        if iters is None and it % check == 0:
            R = Y - A @ X
            P = 0.5 * np.linalg.norm(R) ** 2 + lam * np.sum(np.linalg.norm(X, axis=1))
            U = R * min(1.0, lam / max(np.max(np.linalg.norm(A.conj().T @ R, axis=1)), 1e-300))
            if P - (np.real(np.vdot(U, Y)) - 0.5 * np.linalg.norm(U) ** 2) <= tol * P:
                break
    L21_ITERS.append(it)
    return X

# ============================================================
# fig07-channel : การประมาณช่องสัญญาณเบาบางในระบบ OFDM
#   (ก) NMSE เทียบกับจำนวนนำร่อง เมื่อวิถีอยู่บนกริด  (ข) ความไม่ตรงกันของฐานเมื่อวิถีอยู่นอกกริด
# ============================================================
def nmse_db(a, b):
    return 10 * np.log10(np.sum(np.abs(a - b) ** 2) / np.sum(np.abs(b) ** 2))

if want("channel"):
    Nc, Lcp, kp, snr = 256, 64, 6, 20.0
    sig = 10 ** (-snr / 20)                                         # สัญญาณรบกวนต่อซับแคร์ริเออร์ (|H|^2 เฉลี่ย = 1)
    rng = np.random.default_rng(11)
    Ffull = np.exp(-2j * np.pi * np.outer(np.arange(Nc), np.arange(Lcp)) / Nc)
    trials = 300
    pilots = [12, 16, 20, 24, 32, 48, 64, 96, 128]
    res = {"LS": [], "OMP": [], "oracle": []}
    for P in pilots:
        acc = {q: [] for q in res}
        for tr in range(trials):
            h = np.zeros(Lcp, complex); d = rng.choice(Lcp, kp, replace=False)
            h[d] = (rng.standard_normal(kp) + 1j * rng.standard_normal(kp)) / np.sqrt(2 * kp)
            Htrue = Ffull @ h
            e = sig * (rng.standard_normal(Nc) + 1j * rng.standard_normal(Nc)) / np.sqrt(2)
            Pr = np.sort(rng.choice(Nc, P, replace=False))         # นำร่องแบบสุ่มสำหรับการรับรู้เชิงบีบอัด
            y = Htrue[Pr] + e[Pr]
            x, _ = omp(Ffull[Pr], y, eps=1.1 * sig * np.sqrt(P), kmax=P // 2)
            acc["OMP"].append(nmse_db(Ffull @ x, Htrue))
            xo = np.zeros(Lcp, complex); xo[d] = np.linalg.lstsq(Ffull[Pr][:, d], y, rcond=None)[0]
            acc["oracle"].append(nmse_db(Ffull @ xo, Htrue))
            if P >= Lcp:                                           # LS ต้องการนำร่องอย่างน้อย L ตำแหน่ง ใช้แบบระยะเท่ากัน
                Pe = np.round(np.linspace(0, Nc, P, endpoint=False)).astype(int); ye = Htrue[Pe] + e[Pe]
                hl = np.linalg.lstsq(Ffull[Pe], ye, rcond=None)[0]
                acc["LS"].append(nmse_db(Ffull @ hl, Htrue))
        for q in res:
            res[q].append(np.mean(acc[q]) if acc[q] else np.nan)
    print(f"channel: N = {Nc} subcarriers, L = {Lcp} taps, k = {kp} paths on the delay grid, SNR {snr} dB, {trials} trials")
    for q in res:
        print(f"   {q:6s} NMSE (dB) at P = " + " ".join(f"{p}:{v:6.1f}" for p, v in zip(pilots, res[q])))
    print(f"   theory for LS with equispaced pilots: NMSE = L/P x 1/SNR -> " +
          " ".join(f"{p}:{10*np.log10(Lcp / p * 10 ** (-snr / 10)):6.1f}" for p in pilots if p >= Lcp))
    # (ข) วิถีนอกกริดและพจนานุกรมที่สุ่มละเอียดขึ้น
    P = 32; overs = [1, 2, 4, 8, 16]
    mm = {"on-grid": [], "off-grid": []}
    for r_ in overs:
        grid = np.arange(0, Lcp, 1 / r_)
        D = np.exp(-2j * np.pi * np.outer(np.arange(Nc), grid) / Nc)
        for case in mm:
            vals = []
            for tr in range(trials):
                tau = rng.choice(Lcp - 1, kp, replace=False).astype(float)
                if case == "off-grid":
                    tau = tau + rng.random(kp)
                g = (rng.standard_normal(kp) + 1j * rng.standard_normal(kp)) / np.sqrt(2 * kp)
                Htrue = np.exp(-2j * np.pi * np.outer(np.arange(Nc), tau) / Nc) @ g
                e = sig * (rng.standard_normal(Nc) + 1j * rng.standard_normal(Nc)) / np.sqrt(2)
                Pr = np.sort(rng.choice(Nc, P, replace=False)); y = Htrue[Pr] + e[Pr]
                x, _ = omp(D[Pr], y, eps=1.1 * sig * np.sqrt(P), kmax=P // 2)
                vals.append(nmse_db(D @ x, Htrue))
            mm[case].append(np.mean(vals))
    coh = []
    for r_ in overs:
        grid = np.arange(0, Lcp, 1 / r_); a0 = np.exp(-2j * np.pi * np.arange(Nc) * grid[0] / Nc)
        a1 = np.exp(-2j * np.pi * np.arange(Nc) * grid[1] / Nc); coh.append(np.abs(a0.conj() @ a1) / Nc)
    print(f"channel: basis mismatch, P = {P} random pilots, OMP with delay grid refined by factor r")
    for q in mm:
        print(f"   {q:9s} NMSE (dB) at r = " + " ".join(f"{o}:{v:6.1f}" for o, v in zip(overs, mm[q])))
    print("   coherence of neighbouring atoms (full band) at r = " + " ".join(f"{o}:{c:.3f}" for o, c in zip(overs, coh)))
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.5))
    axes[0].plot(pilots, res["OMP"], "o-", color=ACC, ms=3, label="OMP (random pilots)")
    axes[0].plot(pilots, res["oracle"], "--", color=GRY, label="oracle LS")
    axes[0].plot(pilots, res["LS"], "s-", color=RED, ms=3, label="LS (equispaced)")
    axes[0].set_xscale("log", base=2); axes[0].set_xticks(pilots[::2]); axes[0].set_xticklabels([str(p) for p in pilots[::2]])
    axes[0].set_xlabel("number of pilots $m$"); axes[0].set_ylabel("NMSE (dB)"); axes[0].legend(frameon=False, fontsize=6.5)
    axes[0].set_title("(a)", fontsize=9, loc="left")
    axes[1].plot(overs, mm["on-grid"], "o-", color=ACC, ms=3, label="on-grid delays")
    axes[1].plot(overs, mm["off-grid"], "s-", color=RED, ms=3, label="off-grid delays")
    axes[1].set_xscale("log", base=2); axes[1].set_xticks(overs); axes[1].set_xticklabels([str(o) for o in overs])
    axes[1].set_xlabel("grid refinement factor $r$"); axes[1].set_ylabel("NMSE (dB)"); axes[1].legend(frameon=False, fontsize=6.5)
    axes[1].set_title("(b)", fontsize=9, loc="left")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig07-channel")

# ============================================================
# fig07-spectrum : การชักตัวอย่างแบบหลายโคเซตและการกู้คืนตัวรองรับเชิงสเปกตรัมแบบ MMV
# ============================================================
if want("spectrum"):
    Lc_, K, T, snr = 64, 6, 20, 20.0
    rng = np.random.default_rng(12)
    Fl = np.exp(2j * np.pi * np.outer(np.arange(Lc_), np.arange(Lc_)) / Lc_)
    ps = [6, 8, 10, 12, 14, 16, 20, 24, 28, 32]
    trials = 400
    out = {"SOMP (all columns)": [], "l21 (all columns)": [], "OMP (one column)": []}
    for p in ps:
        ok = {q: 0 for q in out}
        for tr in range(trials):
            C = np.sort(rng.choice(Lc_, p, replace=False))         # ดัชนีโคเซตที่เลือก
            A = Fl[C] / np.sqrt(p)
            S = np.sort(rng.choice(Lc_, K, replace=False))
            X = np.zeros((Lc_, T), complex)
            X[S] = (rng.standard_normal((K, T)) + 1j * rng.standard_normal((K, T))) / np.sqrt(2)
            sg = np.sqrt(np.mean(np.abs(A @ X) ** 2) * 10 ** (-snr / 10))
            Y = A @ X + sg * (rng.standard_normal((p, T)) + 1j * rng.standard_normal((p, T))) / np.sqrt(2)
            ok["SOMP (all columns)"] += set(somp(A, Y, K)) == set(S)
            Xh = fista_l21(A, Y, lam=2.0 * sg * (np.sqrt(T) + np.sqrt(2 * np.log(Lc_))))
            ok["l21 (all columns)"] += set(np.argsort(np.linalg.norm(Xh, axis=1))[::-1][:K]) == set(S)
            _, S1 = omp(A, Y[:, 0], k=K)
            ok["OMP (one column)"] += set(S1) == set(S)
        for q in out:
            out[q].append(ok[q] / trials)
    print(f"spectrum: L = {Lc_} spectral cells, K = {K} occupied, T = {T} columns, SNR {snr} dB, {trials} trials")
    for q in out:
        print(f"   {q:20s} success at p = " + " ".join(f"{p}:{v:.2f}" for p, v in zip(ps, out[q])))
    print(f"   l21 solved to relative duality gap {GAP_TOL:g}: iterations median {int(np.median(L21_ITERS))}, max {max(L21_ITERS)} ({len(L21_ITERS)} solves)")
    L21_ITERS.clear()
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for (q, st) in zip(out, [("o-", ACC), ("^-", GRN), ("s--", RED)]):
        ax.plot(ps, out[q], st[0], color=st[1], ms=3, label=q.replace("l21", r"$\ell_{2,1}$"))
    ax.axvline(K, color=GRY, ls=":", lw=1.0); ax.axvline(2 * K, color=GRY, ls="--", lw=1.0)
    ax.text(K + 0.3, 0.05, "$k$", fontsize=7, color=GRY); ax.text(2 * K + 0.3, 0.05, "$2k$", fontsize=7, color=GRY)
    ax.set_xlabel("number of cosets $m$"); ax.set_ylabel("support recovery rate"); ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig07-spectrum")

# ============================================================
# fig07-doa : การประมาณทิศทางการมาถึงด้วยแถวลำดับเชิงเส้นสม่ำเสมอ
#   (ก) สเปกตรัมเชิงพื้นที่จากการทดลองหนึ่งครั้ง  (ข) ความน่าจะเป็นในการแยกแหล่งกำเนิดเทียบกับ SNR
# ============================================================
def steer(M, ang_deg):
    return np.exp(1j * np.pi * np.outer(np.arange(M), np.sin(np.deg2rad(np.atleast_1d(ang_deg)))))

def music(Y, A, K):
    R = Y @ Y.conj().T / Y.shape[1]; w, V = np.linalg.eigh(R); En = V[:, :-K]
    return 1 / np.sum(np.abs(En.conj().T @ A) ** 2, axis=0)

_L1SVD_CACHE = {}
L1SVD_GAP = []          # ช่องว่างคู่กันสัมพัทธ์ของผลเฉลยทุกครั้ง (ใช้รับรองการลู่เข้า)

def l21_ipm(A, Ysv, lam):
    """min 1/2 ||Ysv - A X||_F^2 + lam ||X||_{2,1} แบบจำนวนเชิงซ้อน แก้ด้วยวิธีจุดภายใน (Clarabel ผ่าน CVXPY)
       ในรูปจำนวนจริง (ส่วนจริงและส่วนจินตภาพของแถวเดียวกันอยู่ในกลุ่มเดียวกัน) โดยสร้างปัญหาครั้งเดียวแล้วเปลี่ยนเฉพาะพารามิเตอร์
       FISTA ใช้ได้ในหลักการ แต่กรณีแหล่งกำเนิดสอดคล้องกันที่ SNR สูงต้องใช้เกิน 10^5 รอบ"""
    import cvxpy as cp
    key = (id(A), Ysv.shape[1])
    if key not in _L1SVD_CACHE:
        M, G = A.shape; Kc = Ysv.shape[1]
        Xr = cp.Variable((G, Kc)); Xi = cp.Variable((G, Kc))
        Yr = cp.Parameter((M, Kc)); Yi = cp.Parameter((M, Kc)); lp = cp.Parameter(nonneg=True)
        Rr = Yr - (A.real @ Xr - A.imag @ Xi); Ri = Yi - (A.real @ Xi + A.imag @ Xr)
        obj = 0.5 * (cp.sum_squares(Rr) + cp.sum_squares(Ri)) + lp * cp.sum(cp.norm(cp.hstack([Xr, Xi]), 2, axis=1))
        _L1SVD_CACHE[key] = (cp.Problem(cp.Minimize(obj)), Xr, Xi, Yr, Yi, lp)
    prob, Xr, Xi, Yr, Yi, lp = _L1SVD_CACHE[key]
    Yr.value = np.ascontiguousarray(Ysv.real); Yi.value = np.ascontiguousarray(Ysv.imag); lp.value = lam
    prob.solve(solver="CLARABEL", tol_gap_rel=1e-10, tol_gap_abs=1e-12, tol_feas=1e-10)
    X = Xr.value + 1j * Xi.value
    # ใบรับรอง: ช่องว่างคู่กันสัมพัทธ์ของผลเฉลย (จุดคู่กันจากการปรับมาตราส่วนเศษเหลือ)
    R = Ysv - A @ X
    P = 0.5 * np.linalg.norm(R) ** 2 + lam * np.sum(np.linalg.norm(X, axis=1))
    U = R * min(1.0, lam / np.max(np.linalg.norm(A.conj().T @ R, axis=1)))
    L1SVD_GAP.append((P - (np.real(np.vdot(U, Ysv)) - 0.5 * np.linalg.norm(U) ** 2)) / P)
    nr = np.linalg.norm(X, axis=1)
    X[nr <= 1e-6 * nr.max()] = 0.0                                 # ตัดเศษระดับความคลาดเคลื่อนของวิธีจุดภายในให้เป็นศูนย์แม่นตรง
    return X

def l1svd(Y, A, K, sigma):
    U, s, Vh = np.linalg.svd(Y, full_matrices=False)
    Ysv = Y @ Vh[:K].conj().T                                      # ลดมิติข้อมูลเหลือ K สดมภ์ (Malioutov และคณะ)
    M, G = A.shape
    lam = sigma * np.sqrt(M) * (np.sqrt(K) + np.sqrt(2 * np.log(G)))
    X = l21_ipm(A, Ysv, lam)
    return np.linalg.norm(X, axis=1)

def peaks(spec, grid, K):
    idx = [i for i in range(1, len(spec) - 1) if spec[i] >= spec[i - 1] and spec[i] >= spec[i + 1] and spec[i] > 0]
    idx = sorted(idx, key=lambda i: spec[i], reverse=True)[:K]
    return np.sort(grid[idx]) if len(idx) == K else None

if want("doa"):
    M, T, K = 10, 50, 2
    grid = np.arange(-90, 90.01, 0.5); A = steer(M, grid)
    th_true = np.array([0.25, 6.25])                                 # อยู่นอกกริดครึ่งช่อง และห่างกันน้อยกว่าความกว้างลำคลื่น
    bw = np.rad2deg(2 / M)                                           # ความกว้างลำคลื่นโดยประมาณที่แนวตั้งฉาก (เรเดียน -> องศา)
    rng = np.random.default_rng(13)
    def snapshot(snr, coherent):
        S = (rng.standard_normal((K, T)) + 1j * rng.standard_normal((K, T))) / np.sqrt(2)
        if coherent:
            S[1] = S[0] * np.exp(1j * 0.7)
        sg = 10 ** (-snr / 20)
        N = sg * (rng.standard_normal((M, T)) + 1j * rng.standard_normal((M, T))) / np.sqrt(2)
        return steer(M, th_true) @ S + N, sg
    Y, sg = snapshot(10.0, False)
    bf = np.real(np.sum(np.abs(A.conj().T @ Y) ** 2, axis=1))
    mu = music(Y, A, K); l1 = l1svd(Y, A, K, sg)
    print(f"doa: ULA M = {M}, T = {T} snapshots, sources at {th_true} deg (beamwidth ~ {bw:.1f} deg), grid step 0.5 deg")
    snrs = [-5, 0, 5, 10, 15, 20, 25]; trials = 200
    prob = {}
    for coh in [False, True]:
        for nm in ["beamforming", "MUSIC", "l1-SVD"]:
            prob[(nm, coh)] = []
        for snr in snrs:
            hit = {nm: 0 for nm in ["beamforming", "MUSIC", "l1-SVD"]}
            for tr in range(trials):
                Yt, sgt = snapshot(snr, coh)
                specs = {"beamforming": np.real(np.sum(np.abs(A.conj().T @ Yt) ** 2, axis=1)),
                         "MUSIC": music(Yt, A, K), "l1-SVD": l1svd(Yt, A, K, sgt)}
                for nm, sp in specs.items():
                    pk = peaks(sp, grid, K)
                    hit[nm] += pk is not None and np.all(np.abs(pk - th_true) <= 1.5)
            for nm in hit:
                prob[(nm, coh)].append(hit[nm] / trials)
        for nm in ["beamforming", "MUSIC", "l1-SVD"]:
            print(f"   {'coherent' if coh else 'uncorrelated':12s} {nm:12s} P(resolve) at SNR = " +
                  " ".join(f"{s}:{v:.2f}" for s, v in zip(snrs, prob[(nm, coh)])))
    print(f"   l1-SVD solved by interior point (Clarabel): largest relative duality gap {max(L1SVD_GAP):.1e} ({len(L1SVD_GAP)} solves)")
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.5))
    for sp, st, lb in [(bf, ("-", GRY), "beamforming"), (mu, ("-", RED), "MUSIC"), (l1, ("-", ACC), r"$\ell_1$-SVD")]:
        v = 10 * np.log10(np.maximum(sp / sp.max(), 1e-4))
        axes[0].plot(grid, v, st[0], color=st[1], lw=1.0, label=lb)
    for t_ in th_true:
        axes[0].axvline(t_, color=INK, ls=":", lw=0.7)
    axes[0].set_xlim(-30, 36); axes[0].set_ylim(-40, 2)
    axes[0].set_xlabel("angle (deg)"); axes[0].set_ylabel("normalized spectrum (dB)")
    axes[0].legend(frameon=False, fontsize=6.5, loc="lower left"); axes[0].set_title("(a)", fontsize=9, loc="left")
    for nm, st in [("beamforming", ("v-", GRY)), ("MUSIC", ("s-", RED)), ("l1-SVD", ("o-", ACC))]:
        lb = r"$\ell_1$-SVD" if nm == "l1-SVD" else nm
        axes[1].plot(snrs, prob[(nm, False)], st[0], color=st[1], ms=3, label=lb)
        axes[1].plot(snrs, prob[(nm, True)], st[0].replace("-", "--"), color=st[1], ms=3, alpha=0.6)
    axes[1].set_xlabel("SNR (dB)"); axes[1].set_ylabel("probability of resolution")
    axes[1].legend(frameon=False, fontsize=6.5, loc="center right"); axes[1].set_title("(b)", fontsize=9, loc="left")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig07-doa")

# ============================================================
# fig07-fdia : การประมาณสถานะภายใต้การโจมตีแบบเบาบาง (หนึ่งช่วงเวลา) และดัชนีความมั่นคง
# ============================================================
def build_grid(name):
    from pypower.api import case9, case14, ext2int, makeBdc, rundcpf, ppoption
    case = {"case9": case9, "case14": case14}[name]
    ppc = ext2int(case()); Bbus, Bf, _, _ = makeBdc(ppc["baseMVA"], ppc["bus"], ppc["branch"])
    Bbus = Bbus.toarray(); Bf = Bf.toarray(); keep = np.arange(1, Bbus.shape[0])   # บัสที่ 1 เป็นบัสอ้างอิง
    H = np.vstack([Bbus[keep][:, keep], Bf[:, keep], -Bf[:, keep]])                # การฉีดกำลัง + การไหลต้นทาง + การไหลปลายทาง
    r = rundcpf(case(), ppoption(VERBOSE=0, OUT_ALL=0))[0]
    th = np.deg2rad(r["bus"][:, 8])[keep] - np.deg2rad(r["bus"][0, 8])
    return H, th

def security_index(H):
    """alpha = min ||H c||_0 เหนือ c != 0 ด้วยกำหนดการจำนวนเต็มผสม (HiGHS) โดยตรึง c_j = 1 ทีละตัว"""
    from scipy.optimize import milp, LinearConstraint, Bounds
    m, n = H.shape; best = (m + 1, None); status, gaps = [], []
    Cb = 1e3; Mb = np.abs(H).sum(axis=1).max() * Cb
    for j in range(n):
        # ตัวแปร [c (n), z (m) binary]; minimize sum z  s.t.  -Mb z <= H c <= Mb z,  c_j = 1
        cvec = np.concatenate([np.zeros(n), np.ones(m)])
        A1 = np.hstack([H, -Mb * np.eye(m)]); A2 = np.hstack([-H, -Mb * np.eye(m)])
        cons = [LinearConstraint(A1, -np.inf, 0), LinearConstraint(A2, -np.inf, 0)]
        lb = np.concatenate([-Cb * np.ones(n), np.zeros(m)]); ub = np.concatenate([Cb * np.ones(n), np.ones(m)])
        lb[j] = ub[j] = 1.0
        res = milp(cvec, constraints=cons, bounds=Bounds(lb, ub), integrality=np.concatenate([np.zeros(n), np.ones(m)]))
        status.append(res.status); gaps.append(getattr(res, "mip_gap", np.nan))
        if res.status != 0 or res.x is None:
            # ค่าต่ำสุดเหนือ j ทุกตัวจะไม่ได้รับการรับรองหากมี MILP ใดล้มเหลวหรือถึงขีดจำกัด
            raise RuntimeError(f"security_index uncertified: MILP for c_{j} = 1 returned status {res.status} ({res.message})")
        c = res.x[:n]; val = int(np.sum(np.abs(H @ c) > 1e-6))
        if val < best[0]:
            best = (val, c)
    security_index.last_check = (len(status), int(np.sum(np.array(status) == 0)), float(np.nanmax(gaps)) if gaps else np.nan)
    return best

def lad(H, y):
    from scipy.optimize import linprog
    m, n = H.shape
    cvec = np.concatenate([np.zeros(n), np.ones(m)])
    A = np.block([[H, -np.eye(m)], [-H, -np.eye(m)]]); b = np.concatenate([y, -y])
    res = linprog(cvec, A_ub=A, b_ub=b, bounds=[(None, None)] * n + [(0, None)] * m, method="highs")
    if res.status != 0:
        raise RuntimeError(f"LAD linear program failed: status {res.status} ({res.message})")
    return res.x[:n]

def lnr(H, y, sigma, thr=3.0):
    """การทดสอบส่วนเหลือบรรทัดฐานที่ใหญ่ที่สุด (largest normalized residual): ตัดหน่วยวัดทีละหนึ่งจนผ่านการทดสอบ"""
    keep = np.arange(H.shape[0])
    while len(keep) > H.shape[1]:
        Hk = H[keep]; th = np.linalg.lstsq(Hk, y[keep], rcond=None)[0]
        Om = np.eye(len(keep)) - Hk @ np.linalg.pinv(Hk)
        rn = np.abs(y[keep] - Hk @ th) / (sigma * np.sqrt(np.maximum(np.diag(Om), 1e-12)))
        if rn.max() <= thr:
            return th
        keep = np.delete(keep, int(np.argmax(rn)))
    return np.linalg.lstsq(H[keep], y[keep], rcond=None)[0]

if want("fdia"):
    print("fdia: DC measurement models built from PYPOWER (bus 1 = reference)")
    alphas = {}
    for name in ["case9", "case14"]:
        H, th0 = build_grid(name); m, n = H.shape
        t0 = time.time(); a, c = security_index(H); alphas[name] = a
        print(f"   {name}: H is {m} x {n}, rank {np.linalg.matrix_rank(H)}, sigma_max/min = "
              f"{np.linalg.svd(H, compute_uv=False)[0]:.2f}/{np.linalg.svd(H, compute_uv=False)[-1]:.3f},"
              f" ||theta0|| = {np.linalg.norm(th0):.3f}; security index alpha = {a} ({time.time()-t0:.1f} s)")
        ns, nok, gmax = security_index.last_check
        print(f"      security index certified: {nok}/{ns} MILPs optimal, max MIP gap {gmax:.1e}")
        print(f"      theta0 = {np.round(th0, 4)}")
    H, th0 = build_grid("case14"); m, n = H.shape; z0 = H @ th0
    rng = np.random.default_rng(14)
    snr = 30.0; sigma = np.linalg.norm(z0) / np.sqrt(m) * 10 ** (-snr / 20)
    ks = list(range(0, 15)); trials = 300
    amp = 3.0 * np.abs(z0).max()                                      # ขนาดของการโจมตีแต่ละหน่วยวัด
    tau = sigma ** 2 * (m - n + 3 * np.sqrt(2 * (m - n)))               # เกณฑ์ไคกำลังสองโดยประมาณ (ประมาณ 3 ส่วนเบี่ยงเบนมาตรฐาน)
    out = {"LS": [], "LNR": [], "LAD": []}; det = {"random": [], "stealth": []}
    for k in ks:
        acc = {q: [] for q in out}; dr = 0; ds = 0
        for tr in range(trials):
            a = np.zeros(m)
            if k > 0:
                rows = rng.choice(m, k, replace=False); a[rows] = amp * rng.uniform(0.5, 1.5, k) * rng.choice([-1, 1], k)
            y = z0 + a + sigma * rng.standard_normal(m)
            for q in out:
                est = {"LS": lambda: np.linalg.lstsq(H, y, rcond=None)[0], "LNR": lambda: lnr(H, y, sigma),
                       "LAD": lambda: lad(H, y)}[q]()
                acc[q].append(np.linalg.norm(est - th0) / np.linalg.norm(th0) < 0.02)
            r = y - H @ np.linalg.lstsq(H, y, rcond=None)[0]; dr += np.sum(r ** 2) > tau
            # การโจมตีแบบซ่อนเร้น a = H c ที่มีขนาดพลังงานเท่ากัน
            cst = rng.standard_normal(n); ast = H @ cst; ast *= np.linalg.norm(a if k > 0 else z0) / np.linalg.norm(ast)
            ys = z0 + ast + sigma * rng.standard_normal(m); rs = ys - H @ np.linalg.lstsq(H, ys, rcond=None)[0]
            ds += np.sum(rs ** 2) > tau
        for q in out:
            out[q].append(np.mean(acc[q]))
        det["random"].append(dr / trials); det["stealth"].append(ds / trials)
    print(f"fdia: IEEE 14-bus, single snapshot, SNR {snr} dB, attack amplitude {amp:.2f} p.u. per attacked meter, "
          f"success = relative error < 2%, {trials} trials")
    for q in out:
        print(f"   {q:4s} success at k = " + " ".join(f"{k}:{v:.2f}" for k, v in zip(ks, out[q])))
    for q in det:
        print(f"   chi-square detector, {q:7s} attack: detection rate at k = " + " ".join(f"{k}:{v:.2f}" for k, v in zip(ks, det[q])))
    # การโจมตีแบบซ่อนเร้นที่เบาบางที่สุด (ใช้ alpha หน่วยวัด) หลบได้ทั้งการทดสอบและ LAD
    a_min, c_min = security_index(H); ast = H @ c_min; ast = ast / np.abs(ast).max() * amp
    ys = z0 + ast; th_lad = lad(H, ys); th_ls = np.linalg.lstsq(H, ys, rcond=None)[0]
    print(f"   sparsest stealth attack: ||a||_0 = {int(np.sum(np.abs(ast) > 1e-9))}, residual of LS = "
          f"{np.linalg.norm(ys - H @ th_ls):.2e}, LAD estimate error = {np.linalg.norm(th_lad - th0)/np.linalg.norm(th0):.3f},"
          f" ||H^+ a|| / ||theta0|| = {np.linalg.norm(np.linalg.pinv(H) @ ast)/np.linalg.norm(th0):.3f}")
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.5))
    for q, st in [("LS", ("v-", GRY)), ("LNR", ("s-", RED)), ("LAD", ("o-", ACC))]:
        lb = {"LS": "LS", "LNR": "LS + LNR test", "LAD": r"$\ell_1$ regression"}[q]
        axes[0].plot(ks, out[q], st[0], color=st[1], ms=3, label=lb)
    axes[0].axvline(alphas["case14"] / 2, color=GRY, ls=":", lw=1.0)
    axes[0].text(alphas["case14"] / 2 + 0.2, 0.5, r"$\alpha_{\mathbf{H}}/2$", fontsize=7, color=GRY)
    axes[0].set_xlabel("number of attacked meters $k$"); axes[0].set_ylabel("success rate")
    axes[0].legend(frameon=False, fontsize=6.5); axes[0].set_title("(a)", fontsize=9, loc="left")
    axes[1].plot(ks, det["random"], "o-", color=ACC, ms=3, label="random sparse attack")
    axes[1].plot(ks, det["stealth"], "s--", color=RED, ms=3, label=r"stealth attack $\mathbf{a} = \mathbf{H}\mathbf{c}$")
    axes[1].set_xlabel("number of attacked meters $k$"); axes[1].set_ylabel("detection rate")
    axes[1].set_ylim(-0.03, 1.05); axes[1].legend(frameon=False, fontsize=6.5, loc="center right")
    axes[1].set_title("(b)", fontsize=9, loc="left")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig07-fdia")
