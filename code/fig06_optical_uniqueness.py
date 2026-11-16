# -*- coding: utf-8 -*-
"""วินิจฉัยความต่างของ PSNR อาดามาร์ที่ m/n = 0.10 (ประเด็น E03 ของ Technical Audit 01)
สร้างปัญหาสามครั้งเดิมของ fig06_generate.py (เล่นตัวสุ่ม seed 8 ซ้ำตามลำดับเดิม) แล้ว
  (1) แก้ basis pursuit ด้วยตัวแก้สามแบบ: HiGHS จุดภายใน (แบบในเล่ม), HiGHS ซิมเพล็กซ์คู่กัน และ Clarabel
  (2) หาช่วงของ PSNR บนเซตผลเฉลยเหมาะที่สุด: ตรึง ||c||_1 ไว้ที่ค่าต่ำสุด (เผื่อสัมพัทธ์ 1e-9)
      แล้วหาผลเฉลยที่ใกล้และไกลภาพจริงที่สุดในทิศทาง <c, c_true>
รัน: python3 fig06_optical_uniqueness.py"""
import os, sys, importlib.util
import numpy as np
from scipy.optimize import linprog
ARGS = sys.argv[1:]                               # เลือกสัดส่วนที่จะแก้ เช่น 0.2 (ค่าเริ่มต้น: ทั้ง 0.1 และ 0.2)
sys.argv = ["x", "__none__"]
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("f6", os.path.join(HERE, "fig06_generate.py"))
f6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(f6)

n1 = 64; n = n1 * n1; lev = 3
img = f6.textured_image(n1, 2)
arr, sl = f6.wav_l(img, lev)
c_true = arr.ravel()
synth = lambda c: f6.iwav_l(c.reshape(n1, n1), sl).ravel()
Wsyn = np.column_stack([synth(e) for e in np.eye(n)])
rng = np.random.default_rng(8)
m = int(0.1 * n)
FRACS = [0.1, 0.2]                                # m <= 1024 ใช้กำหนดการเชิงเส้นในเล่ม

def draw(nm):                                     # ลำดับการใช้ตัวสุ่มเดียวกับ sp_draw ใน fig06_generate.py
    if nm == "Bernoulli":
        return rng.choice([-1.0, 1.0], (m, n)) / np.sqrt(m)
    if nm == "Hadamard, no DC row":
        return rng.choice(np.arange(1, n), m, replace=False)
    return np.concatenate(([0], rng.choice(np.arange(1, n), m - 1, replace=False)))

def lp(Amat, y, method, cost=None, l1cap=None):
    """min cost^T [p; q] s.t. A (p - q) = y, p, q >= 0 และ (ถ้ากำหนด) 1^T (p + q) <= l1cap"""
    N = Amat.shape[1]
    cvec = np.ones(2 * N) if cost is None else cost
    kw = dict(A_eq=np.hstack([Amat, -Amat]), b_eq=y, bounds=(0, None), method=method)
    if l1cap is not None:
        kw.update(A_ub=np.ones((1, 2 * N)), b_ub=[l1cap])
    opts = {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10} if method == "highs-ipm" else {}
    res = linprog(cvec, options=opts, **kw)
    assert res.status == 0, res.message
    return res.x[:N] - res.x[N:]

def clarabel(Amat, y):
    import cvxpy as cp
    c = cp.Variable(Amat.shape[1])
    prob = cp.Problem(cp.Minimize(cp.norm1(c)), [Amat @ c == y]); prob.solve(solver="CLARABEL")
    assert prob.status == "optimal", prob.status
    return np.asarray(c.value)

def ps(c):
    return f6.psnr(np.clip(synth(c).reshape(n1, n1), 0, 1), img)

def face_qp(Amat, y, cap, target):
    """min ||c - target||_2 บนเซตผลเฉลยเกือบเหมาะที่สุด {A c = y, ||c||_1 <= cap} (target = 0 คือผลเฉลยนอร์ม l2 น้อยที่สุด)"""
    import cvxpy as cp
    c = cp.Variable(Amat.shape[1])
    prob = cp.Problem(cp.Minimize(cp.sum_squares(c - target)), [Amat @ c == y, cp.norm1(c) <= cap]); prob.solve(solver="CLARABEL")
    assert prob.status in ("optimal", "optimal_inaccurate"), prob.status
    return np.asarray(c.value)

print(f"optical-uniqueness: Hadamard rows (DC row kept), noiseless basis pursuit in the db4 domain; ||c_true||_1 = {np.abs(c_true).sum():.4f}")
for fr in FRACS:
  m = int(fr * n); solve_fr = (not ARGS) or (f"{fr:g}" in ARGS)
  if solve_fr: print(f"m/n = {fr:.2f} (m = {m})")
  for tr in range(3):
    D = {nm: draw(nm) for nm in ["Bernoulli", "Hadamard", "Hadamard, no DC row"]}["Hadamard"]
    if not solve_fr:                              # เล่นตัวสุ่มซ้ำโดยไม่แก้ปัญหา
        continue
    Amat = (f6.hadamard_rows(n, D) / np.sqrt(m)) @ Wsyn
    y = Amat @ c_true
    sols = {"HiGHS-IPM (book)": lp(Amat, y, "highs-ipm"), "HiGHS dual simplex": lp(Amat, y, "highs-ds"), "Clarabel": clarabel(Amat, y)}
    print(f"   trial {tr}:")
    for k_, c in sols.items():
        print(f"      {k_:20s} ||c||_1 = {np.abs(c).sum():.8f}   residual {np.linalg.norm(Amat @ c - y) / np.linalg.norm(y):.1e}"
              f"   nnz(|c|>1e-8) = {int(np.sum(np.abs(c) > 1e-8)):4d}   PSNR {ps(c):6.2f} dB")
    opt = min(np.abs(c).sum() for c in sols.values())
    cap = opt * (1 + 1e-9)
    g = np.concatenate([-c_true, c_true])                                 # <c, c_true> = c_true^T (p - q)
    near = lp(Amat, y, "highs-ds", cost=g, l1cap=cap)                     # ใกล้ทิศทางภาพจริงที่สุดในเซตเหมาะที่สุด
    far = lp(Amat, y, "highs-ds", cost=-g, l1cap=cap)                     # ไกลที่สุด
    print(f"      optimal set (||c||_1 <= {cap:.8f}): PSNR from {ps(far):6.2f} to {ps(near):6.2f} dB along <c, c_true>;"
          f" ||c_near - c_far|| / ||c_true|| = {np.linalg.norm(near - far) / np.linalg.norm(c_true):.3f}", flush=True)
    if fr > 0.1:                                  # ปัญหากำลังสองที่ m = 819 ใช้เวลานาน จึงตรวจเฉพาะ m/n = 0.1
        continue
    cap7 = opt * (1 + 1e-7)                                               # ตัวแก้กรวยต้องการเผื่อ 1e-7
    print(f"      near-optimal set (||c||_1 <= opt (1 + 1e-7)): minimum-l2-norm solution PSNR {ps(face_qp(Amat, y, cap7, 0.0)):6.2f} dB;"
          f" best attainable PSNR {ps(face_qp(Amat, y, cap7, c_true)):6.2f} dB", flush=True)
