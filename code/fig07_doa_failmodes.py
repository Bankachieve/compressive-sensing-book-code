# การวินิจฉัยประกอบภาพ fig07-doa: รูปแบบความล้มเหลวของ l1-SVD (แก้ด้วยวิธีจุดภายใน) เมื่อแหล่งกำเนิดสอดคล้องกันที่ SNR สูง
# รัน: python3 fig07_doa_failmodes.py
"""Failure modes of converged l1-SVD for coherent sources at high SNR."""
import sys, numpy as np, importlib.util, collections
sys.argv = ["x", "__none__"]
spec = importlib.util.spec_from_file_location("f7", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "fig07_generate.py"))
f7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(f7)
M, T, K = 10, 50, 2
grid = np.arange(-90, 90.01, 0.5); A = f7.steer(M, grid); th = np.array([0.25, 6.25])
rng = np.random.default_rng(5)
for snr in [15, 25, 35]:
    modes = collections.Counter(); ex = []
    for _ in range(60):
        S = (rng.standard_normal((K, T)) + 1j * rng.standard_normal((K, T))) / np.sqrt(2); S[1] = S[0] * np.exp(1j * 0.7)
        sg = 10 ** (-snr / 20)
        Y = f7.steer(M, th) @ S + sg * (rng.standard_normal((M, T)) + 1j * rng.standard_normal((M, T))) / np.sqrt(2)
        sp = f7.l1svd(Y, A, K, sg)
        nzr = grid[sp > 0]
        pk = f7.peaks(sp, grid, K)
        ok = pk is not None and np.all(np.abs(pk - th) <= 1.5)
        if ok: modes["resolved"] += 1; continue
        if pk is None: modes["fewer than 2 peaks"] += 1
        else:
            d = np.abs(pk[:, None] - th[None, :]).min(axis=1)
            if np.all(d <= 1.5): modes["both peaks near one source"] += 1
            elif np.any(d > 3): modes["spurious peak away from sources"] += 1
            else: modes["peak 1.5-3 deg off"] += 1
        if len(ex) < 2: ex.append((pk, np.round(nzr, 1)[:12]))
    print(f"coherent SNR {snr}: {dict(modes)}  examples {ex}", flush=True)
