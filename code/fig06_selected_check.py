# -*- coding: utf-8 -*-
"""ตรวจการลู่เข้าของโครงแบบที่ได้รับเลือกในการทดลอง MRI ของบทที่ 6 (ตารางที่ 6.2, 6.3 และภาพตัวประกอบการเร่ง)
หลังแก้ส่วนเหลือคู่กันของ ADMM ให้เป็นนอร์มของผลรวม (C01) โดยแก้ซ้ำด้วยตัวคูณที่ได้รับเลือก แล้วรายงานจำนวนรอบ
สถานะลู่เข้า และส่วนเหลือสุดท้ายเทียบกับขีดจำกัด
  ส่วน mri    : ตัวคูณที่เลือกอ่านจาก logs/fig06_mri_rerun-2569-11-15.log (9 การแก้)
  ส่วน mriacc : ปรับตัวคูณบนกริดใหม่ทุกจุดเหมือน fig06_generate.py mriacc แล้วรายงานโครงแบบที่ได้รับเลือก
รัน: python3 fig06_selected_check.py [mri] [mriacc]"""
import os, sys, importlib.util
import numpy as np
SEL = set(sys.argv[1:]) or {"mri", "mriacc"}
sys.argv = ["x", "__none__"]
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("f6", os.path.join(HERE, "fig06_generate.py"))
f6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(f6)

def solve(y, M, lw, lt):
    r, _ = f6.admm_mri(y, M, lam_w=lw, lam_t=lt, tol=f6.TOL)
    rp, sd, ep, ed = f6.admm_mri.last_res
    return r, f6.admm_mri.last_iters, f6.admm_mri.last_converged, rp / ep, sd / ed

def report(label, img, y, M, lw, lt):
    r, it, conv, a, b = solve(y, M, lw, lt)
    print(f"   {label:34s} lam_w {lw:.0e} lam_t {lt:.0e}: PSNR {f6.psnr(r, img):5.2f} dB, iterations {it:5d},"
          f" converged {conv}, r_pri/eps_pri {a:.2f}, s_dual/eps_dual {b:.2f}", flush=True)
    return it, conv

n = 256
x0 = f6.phantom(n); xt = f6.textured_image(n)
if "mri" in SEL:
    R = 4
    M = f6.mask_vardens(n, R, np.random.default_rng(3)); Mu = f6.mask_random(n, R, np.random.default_rng(3)); Me = f6.mask_uniform(n, R)
    y = f6.acquire(x0, M, 5); yt = f6.acquire(xt, M, 7)
    print("selected-check mri: selected multipliers from logs/fig06_mri_rerun-2569-11-15.log")
    res = []
    for lab, lw, lt in [("phantom wavelet", 3e-3, 0.0), ("phantom TV", 0.0, 1e-3), ("phantom TV + wavelet", 1e-6, 1e-3)]:
        res.append(report(lab, x0, y, M, lw, lt))
    for lab, MM in [("phantom equispaced TV", Me), ("phantom uniform random TV", Mu)]:
        res.append(report(lab, x0, f6.acquire(x0, MM, 5), MM, 0.0, 1e-3))
    for lab, lw, lt in [("textured wavelet", 3e-5, 0.0), ("textured TV", 0.0, 1e-3), ("textured TV + wavelet", 3e-5, 1e-3)]:
        res.append(report(lab, xt, yt, M, lw, lt))
    print(f"   selected solves: {len(res)}, converged {sum(c for _, c in res)}, max iterations {max(i for i, _ in res)}")
if "mriacc" in SEL:
    rec = {}
    orig = f6.admm_mri
    def wrapped(y, M, **kw):                     # บันทึกจำนวนรอบและสถานะของทุกการแก้ เพื่อดูของโครงแบบที่ได้รับเลือก
        out = orig(y, M, **kw)                   # admm_mri เขียนสถานะไว้ที่ชื่อ admm_mri ในโมดูล ซึ่งตอนนี้คือ wrapped
        rec[(id(y), round(kw.get("lam_w", 0.0), 12), round(kw.get("lam_t", 0.0), 12))] = (wrapped.last_iters, wrapped.last_converged)
        return out
    f6.admm_mri = wrapped
    print("selected-check mriacc: lambda_t re-tuned on the grid at every acceleration (as in fig06_generate.py mriacc)")
    sel = []
    for nm, img in [("phantom", x0), ("textured", xt)]:
        for Rk in [2, 3, 4, 6, 8, 12]:
            Mk = f6.mask_vardens(n, Rk, np.random.default_rng(3)); yk = f6.acquire(img, Mk, 11)
            p, _, lw, lt = f6.tune(yk, Mk, img, "TV")
            it, conv = rec[(id(yk), round(lw, 12), round(lt, 12))]
            sel.append((it, conv))
            print(f"   {nm:9s} n/m = {Rk:2d}: TV {p:5.1f} dB, lam_t {lt:.0e}, iterations {it:5d}, converged {conv}", flush=True)
    print(f"   selected solves: {len(sel)}, converged {sum(c for _, c in sel)}, max iterations {max(i for i, _ in sel)};"
          f" all tuning solves: {len(f6.CONV_LOG)}, converged {sum(f6.CONV_LOG)}, median iterations {int(np.median(f6.ITER_LOG))}, max {max(f6.ITER_LOG)}")
