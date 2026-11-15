# -*- coding: utf-8 -*-
"""ตัวเลขประกอบเฉลยปัญหาท้ายบทที่ต้องรันโปรแกรม (ปัญหาที่ 1.6 และ 4.4)
ทุกส่วนกำหนด seed ไว้ จึงทำซ้ำได้ตรงตัว เลือกรันบางส่วนได้ เช่น `python exercises_check.py ex1-6`
ส่วนของปัญหาที่ 4.6 อยู่ใน `fig04_generate.py rwl1` เพราะใช้ขั้นตอนวิธี IRLS ของบทที่ 4"""
import numpy as np
from scipy.optimize import linprog
import sys

RUN = set(sys.argv[1:])
def want(name):
    return (not RUN) or (name in RUN)

def gaussian(m, n, rng):
    return rng.standard_normal((m, n)) / np.sqrt(m)

def sparse_signal(n, k, rng):
    x = np.zeros(n); S = rng.choice(n, k, replace=False); x[S] = rng.standard_normal(k)
    return x

def rel_err(xh, x):
    return np.linalg.norm(xh - x) / np.linalg.norm(x)

def basis_pursuit(Phi, y):
    """(P1) ผ่านกำหนดการเชิงเส้น z = u - v, u, v >= 0; คืน None หากตัวแก้ไม่สำเร็จ"""
    m, n = Phi.shape
    res = linprog(np.ones(2 * n), A_eq=np.hstack([Phi, -Phi]), b_eq=y, bounds=(0, None), method="highs")
    return None if res.status != 0 else res.x[:n] - res.x[n:]

def soft(z, lam):
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0.0)

if want("ex1-6"):
    # ปัญหาที่ 1.6: อัตราความสำเร็จของ (P1) เมื่อ n = 256, k = 8 และ m = 16, 24, ..., 128
    rng = np.random.default_rng(16); n, k, trials = 256, 8, 100
    ms = np.arange(16, 129, 8); rates = []
    for m in ms:
        ok = 0
        for _ in range(trials):
            Phi = gaussian(m, n, rng); x = sparse_signal(n, k, rng); xh = basis_pursuit(Phi, Phi @ x)
            ok += xh is not None and rel_err(xh, x) < 1e-3
        rates.append(ok / trials)
    print(f"ex1-6: n={n} k={k} trials={trials} seed 16; success = rel. err < 1e-3")
    print("   " + "  ".join(f"m={m}:{r:.2f}" for m, r in zip(ms, rates)))
    first = next((int(m) for m, r in zip(ms, rates) if r > 0.95), None)
    print(f"   smallest m with success > 95%: {first};  2 k log(n/k) = {2 * k * np.log(n / k):.1f}")

if want("ex4-4"):
    # ปัญหาที่ 4.4: ADMM สำหรับ basis pursuit (ขั้น x คือภาพฉายลงบน {x : Phi x = y}) เทียบกับกำหนดการเชิงเส้น
    rng = np.random.default_rng(44); m, n, k, trials, beta, tmax = 64, 256, 8, 10, 1.0, 20000
    its, e_admm, e_lp = [], [], []
    for _ in range(trials):
        Phi = gaussian(m, n, rng); x = sparse_signal(n, k, rng); y = Phi @ x
        G = np.linalg.solve(Phi @ Phi.T, np.eye(m))                 # (Phi Phi^T)^{-1}; Phi มีแรงก์แถวเต็มด้วยความน่าจะเป็นหนึ่ง
        z = np.zeros(n); u = np.zeros(n); t = tmax
        for t in range(1, tmax + 1):
            v = z - u
            xa = v - Phi.T @ (G @ (Phi @ v - y))                      # ภาพฉายเชิงตั้งฉากลงบนปริภูมิย่อยสัมพรรค
            z = soft(xa + u, 1.0 / beta)
            u = u + xa - z
            if np.linalg.norm(xa - z) < 1e-8:
                break
        its.append(t); e_admm.append(rel_err(z, x))
        xl = basis_pursuit(Phi, y); e_lp.append(rel_err(xl, x) if xl is not None else np.inf)
    print(f"ex4-4: m,n,k = {m},{n},{k}, beta = {beta}, {trials} trials, seed 44; stop when ||x - z|| < 1e-8 (cap {tmax})")
    print(f"   ADMM iterations: median {int(np.median(its))}, max {max(its)}, all = {its}")
    print(f"   median relative error: ADMM {np.median(e_admm):.1e}, LP {np.median(e_lp):.1e}")
