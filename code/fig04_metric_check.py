# -*- coding: utf-8 -*-
"""ตรวจความสอดคล้องของตัวชี้วัดในตารางที่ 4.3 (ประเด็น E01 ของ Technical Audit 01)
(1) เชิงวิเคราะห์: ถ้าความคลาดเคลื่อนบรรทัดฐานใช้นอร์มโฟรเบนิอุสและ PSNR ใช้สมการ (26) ของบทความบนภาพเดียวกัน
    PSNR + 20 log10(nerr) = 20 log10(255 / RMS(X0)) ซึ่งคงที่ทุกแถวและไม่ติดลบสำหรับภาพ 8 บิต
(2) จำลองขนาดเล็ก: สร้างภาพคืนด้วย IHT, IRLS และ IRLS-DR บนปัญหาแบบเดียวกับบทความ (ฐาน DCT, M/N = 0.5)
    แล้วคำนวณผลรวมข้างต้นภายใต้นิยามที่เป็นไปได้สามแบบ เพื่อดูว่านิยามใดให้รูปแบบเดียวกับตาราง
รัน: python3 fig04_metric_check.py"""
import numpy as np
from scipy.fft import idct
from skimage import data, transform

# --- (1) ค่าจากตาราง I, III, IV, V ของบทความ (คัดตามต้นฉบับ)
table = {"IHT": (0.546294822456923, 13.419542261080572), "IRLS": (0.13054135154075275, 20.261949757237478),
         "IRLS-DR": (0.07993784956037091, 21.194630017158865), "IRL1": (0.07596854561346658, 21.64997462101646)}
print("metric-check (1): PSNR + 20 log10(nerr) from the published tables")
for k, (ne, ps) in table.items():
    off = ps + 20 * np.log10(ne)
    print(f"   {k:8s} offset {off:6.2f} dB  -> implied RMS(X0) = 255 / 10^(offset/20) = {255 / 10 ** (off / 20):6.1f}")
print("   with Frobenius-norm nerr and eq. (26) on one 8-bit image the offset is one constant >= 0 (RMS(X0) <= 255)")

# --- (2) การจำลองขนาดเล็ก
rng = np.random.default_rng(43)
h, w = 48, 44; N = h * w; M = N // 2
X0 = transform.resize(data.camera().astype(float), (h, w), anti_aliasing=True)      # ช่วง 0..255
CM = idct(np.eye(h), axis=0, norm="ortho"); CN = idct(np.eye(w), axis=0, norm="ortho")
Psi = np.kron(CN, CM)                                                               # vec(X) = Psi s (ลำดับสดมภ์)
Phi = rng.standard_normal((M, N)) / np.sqrt(M)
A = Phi @ Psi; b = A @ (Psi.T @ X0.ravel(order="F"))

def iht(A, b, K, iters=100):
    s = np.ones(A.shape[1]); mu = 1 / np.linalg.norm(A, 2) ** 2
    for _ in range(iters):
        v = s + mu * A.T @ (b - A @ s); S = np.argsort(np.abs(v))[::-1][:K]
        s = np.zeros_like(v); s[S] = v[S]
    return s

def irls(A, b, p=0.9, iters=100):                                                   # IRLS แบบ Chartrand--Yin
    s = A.T @ np.linalg.solve(A @ A.T, b); eps = 1.0
    for _ in range(iters):
        d = (s ** 2 + eps) ** (1 - p / 2); s_new = d * (A.T @ np.linalg.solve((A * d) @ A.T, b))
        if np.linalg.norm(s_new - s) < np.sqrt(eps) / 100: eps /= 10
        s = s_new
    return s

def irls_dr(A, b, p=0.9, xi=1e-6, iters=100):                                       # ขั้นตอนวิธีที่ 2 ของบทความ
    s = np.ones(A.shape[1])
    for _ in range(iters):
        d = np.abs(s) ** (2 - p); s = d * (A.T @ np.linalg.solve((A * d) @ A.T + xi * np.eye(A.shape[0]), b))
    return s

recs = {"IHT (K = M)": iht(A, b, M), "IRLS": irls(A, b), "IRLS-DR": irls_dr(A, b)}
c0 = 20 * np.log10(255 / np.sqrt(np.mean(X0 ** 2)))
print(f"metric-check (2): {h}x{w} test image, M/N = 0.5, DCT basis; constant 20 log10(255/RMS(X0)) = {c0:.2f} dB")
print("   definition used for nerr / PSNR          " + "  ".join(f"{k:>12s}" for k in recs))
rows = {"Frobenius / eq.(26)": [], "Frobenius / eq.(26) on clipped image": [], "spectral norm(ord=2) / eq.(26)": []}
for k, s in recs.items():
    Xh = (Psi @ s).reshape((h, w), order="F"); E = Xh - X0; Ec = np.clip(Xh, 0, 255) - X0
    psnr = lambda e: 20 * np.log10(255 / np.sqrt(np.mean(e ** 2)))
    rows["Frobenius / eq.(26)"].append(psnr(E) + 20 * np.log10(np.linalg.norm(E) / np.linalg.norm(X0)))
    rows["Frobenius / eq.(26) on clipped image"].append(psnr(Ec) + 20 * np.log10(np.linalg.norm(E) / np.linalg.norm(X0)))
    rows["spectral norm(ord=2) / eq.(26)"].append(psnr(E) + 20 * np.log10(np.linalg.norm(E, 2) / np.linalg.norm(X0, 2)))
for lab, v in rows.items():
    print(f"   {lab:40s} " + "  ".join(f"{x:12.2f}" for x in v))
print("   nerr (Frobenius): " + "  ".join(f"{k} {np.linalg.norm((Psi @ s).reshape((h, w), order='F') - X0) / np.linalg.norm(X0):.3f}" for k, s in recs.items()))
