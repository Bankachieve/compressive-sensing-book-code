# จำนวนรอบที่ FISTA (เกณฑ์ช่องว่างคู่กัน 1e-6, ไม่เกิน 200000 รอบ) ต้องใช้แก้ l1-SVD เมื่อแหล่งกำเนิดสอดคล้องกัน
# รัน: python3 fig07_doa_fista_iters.py
import sys, time, numpy as np, importlib.util
sys.argv = ["x", "__none__"]
spec = importlib.util.spec_from_file_location("f7", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "fig07_generate.py"))
f7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(f7)
M, T, K = 10, 50, 2
grid = np.arange(-90, 90.01, 0.5); A = f7.steer(M, grid); th = np.array([0.25, 6.25])
rng = np.random.default_rng(1)
for coh in [True]:
    for snr in [0, 10, 20, 25]:
        its = []; t0 = time.time()
        for _ in range(5):
            S = (rng.standard_normal((K, T)) + 1j * rng.standard_normal((K, T))) / np.sqrt(2)
            if coh: S[1] = S[0] * np.exp(1j * 0.7)
            sg = 10 ** (-snr / 20)
            Y = f7.steer(M, th) @ S + sg * (rng.standard_normal((M, T)) + 1j * rng.standard_normal((M, T))) / np.sqrt(2)
            U_, s_, Vh = np.linalg.svd(Y, full_matrices=False); Ysv = Y @ Vh[:K].conj().T
            lam = sg * np.sqrt(M) * (np.sqrt(K) + np.sqrt(2 * np.log(A.shape[1])))
            f7.L21_ITERS.clear(); f7.fista_l21(A, Ysv, lam); its.append(f7.L21_ITERS[-1])
        print(f"coherent={coh} snr {snr}: iterations {its}  ({(time.time()-t0)/5:.1f} s/solve)", flush=True)
