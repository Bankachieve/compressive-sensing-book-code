# ตรวจซ้ำกรณีของกล้องจุดภาพเดียว (ภาพ fig06-optical(ก)) ที่ ADMM หยุดก่อนช่องว่างคู่กันถึง 1e-6
# โดยสร้างเมทริกซ์การวัดชุดเดิม (เล่นลำดับตัวสุ่มซ้ำตาม fig06_generate.py) แล้วแก้ basis pursuit ด้วยกำหนดการเชิงเส้นแบบแม่นตรง
# รัน: python3 fig06_optical_lpcheck.py 0.30:0:Hadamard 0.30:1:nodc ...
import sys, time, numpy as np, importlib.util, warnings
warnings.filterwarnings("ignore")
ARGS = sys.argv[1:]
sys.argv = ["x", "__none__"]
spec = importlib.util.spec_from_file_location("f6", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "fig06_generate.py"))
f6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(f6)
n1 = 64; n = n1 * n1; lev = 3
img = f6.textured_image(n1, 2); arr, sl = f6.wav_l(img, lev)
synth = lambda c: f6.iwav_l(c.reshape(n1, n1), sl).ravel()
Wsyn = np.column_stack([synth(e) for e in np.eye(n)])
want = {(float(a.split(":")[0]), int(a.split(":")[1]), a.split(":")[2]) for a in ARGS}
rng = np.random.default_rng(8)
for fr in [0.1, 0.2, 0.3, 0.45, 0.6]:
    m = int(fr * n)
    for tr in range(3):
        for nm in ["Bernoulli", "Hadamard", "nodc"]:
            if nm == "Bernoulli":
                Phi = rng.choice([-1.0, 1.0], (m, n)) / np.sqrt(m)
            elif nm == "nodc":
                Phi = f6.hadamard_rows(n, rng.choice(np.arange(1, n), m, replace=False)) / np.sqrt(m)
            else:
                Phi = f6.hadamard_rows(n, np.concatenate(([0], rng.choice(np.arange(1, n), m - 1, replace=False)))) / np.sqrt(m)
            if (round(fr, 2), tr, nm) in {(round(a, 2), b, c) for a, b, c in want}:
                y = Phi @ img.ravel(); t0 = time.time()
                c = f6.bp_lp(Phi @ Wsyn, y)
                print(f"m/n = {fr:.2f} {nm:9s} trial {tr}: LP {time.time()-t0:.0f} s  ||c||_1 {np.abs(c).sum():.8f}  "
                      f"PSNR {f6.psnr(np.clip(synth(c).reshape(n1, n1), 0, 1), img):.2f} dB", flush=True)
