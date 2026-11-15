# ตรวจการลู่เข้าของการทดลองกล้องจุดภาพเดียวแบบเดิม (LASSO ด้วย FISTA 600 รอบ เทียบกับ 3000 รอบ)
# รัน: python3 fig06_optical_iterprobe.py
"""Convergence probe for the ch6 single-pixel experiment: FISTA 600 iterations (as in the book) vs 3000."""
import sys, numpy as np, importlib.util
sys.argv = ["x", "__none__"]
spec = importlib.util.spec_from_file_location("f6", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "fig06_generate.py"))
f6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(f6)

n1 = 64; n = n1 * n1; lev = 3
img = f6.textured_image(n1, 2)
arr, sl = f6.wav_l(img, lev)
rng = np.random.default_rng(8)
lam_grid = [3e-4, 1e-3, 3e-3]

def obj(A, y, c, lam):
    return 0.5 * np.sum((A(c) - y) ** 2) + lam * np.sum(np.abs(c))

for fr in [0.1, 0.3, 0.6]:
    m = int(fr * n)
    for nm in ["Bernoulli", "Hadamard"]:
        if nm == "Bernoulli":
            Phi = rng.choice([-1.0, 1.0], (m, n)) / np.sqrt(m)
        else:
            rows = np.concatenate(([0], rng.choice(np.arange(1, n), m - 1, replace=False)))
            Phi = f6.hadamard_rows(n, rows) / np.sqrt(m)
        A = lambda c: Phi @ f6.iwav_l(c.reshape(n1, n1), sl).ravel()
        At = lambda r: f6.wav_l((Phi.T @ r).reshape(n1, n1), lev)[0].ravel()
        y = Phi @ img.ravel(); Lc = f6.op_norm(A, At, n) ** 2
        out = []
        for lm in lam_grid:
            c6 = f6.fista_l1(A, At, y, n, lm, 600, Lc)
            c30 = f6.fista_l1(A, At, y, n, lm, 3000, Lc)
            p6 = f6.psnr(np.clip(f6.iwav_l(c6.reshape(n1, n1), sl), 0, 1), img)
            p30 = f6.psnr(np.clip(f6.iwav_l(c30.reshape(n1, n1), sl), 0, 1), img)
            o6, o30 = obj(A, y, c6, lm), obj(A, y, c30, lm)
            out.append((lm, p6, p30, (o6 - o30) / o30))
        best6 = max(o[1] for o in out); best30 = max(o[2] for o in out)
        print(f"m/n={fr:.2f} {nm:9s} best PSNR 600it {best6:5.2f}  3000it {best30:5.2f}  | "
              + "  ".join(f"lam {o[0]:.0e}: {o[1]:.2f}->{o[2]:.2f} relgap {o[3]:.1e}" for o in out), flush=True)
