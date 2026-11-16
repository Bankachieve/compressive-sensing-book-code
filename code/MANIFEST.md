# MANIFEST: ภาพและตารางในเล่ม → ไฟล์ต้นฉบับ สคริปต์ ส่วน seed และบันทึกผล

สร้างอัตโนมัติจาก `main.aux` ต้นฉบับ LaTeX และโค้ดใน `code/` (หมายเลขหน้าเป็นหน้าพิมพ์ของ PDF ฉบับ 299 หน้า)
รันส่วนใดส่วนหนึ่งด้วย `python code/<สคริปต์> <ส่วน>`; สคริปต์บทที่ 1–3 รันทั้งไฟล์ ค่า seed คือ `np.random.default_rng(seed)` ที่ปรากฏในส่วนนั้น

## ภาพ

| ภาพ | หน้า | ป้าย | ไฟล์ภาพ | สคริปต์/ต้นฉบับ | ส่วน | seed | บันทึกผล |
|---|---|---|---|---|---|---|---|
| 1.1 | 3 | fig:sampling | fig01-sampling | fig01_generate.py | – | 2026 | stdout ของ fig01_generate.py |
| 1.2 | 4 | fig:spectrum | fig01-spectrum | fig01_generate.py | – | 2026 | stdout ของ fig01_generate.py |
| 1.3 | 5 | fig:timeline | fig01-timeline | fig01_generate.py | – | 2026 | stdout ของ fig01_generate.py |
| 1.4 | 8 | fig:pipeline | fig01-pipeline | figures/tikz/fig01-pipeline.tex | TikZ | – | แผนภาพเชิงแนวคิด ไม่มีการคำนวณ |
| 1.5 | 10 | fig:measurements | fig01-measurements | fig01_generate.py | – | 2026 | stdout ของ fig01_generate.py |
| 1.6 | 10 | fig:paradigm | fig01-paradigm | figures/tikz/fig01-paradigm.tex | TikZ | – | แผนภาพเชิงแนวคิด ไม่มีการคำนวณ |
| 1.7 | 12 | fig:demo | fig01-demo | fig01_generate.py | – | 2026 | stdout ของ fig01_generate.py |
| 1.8 | 13 | fig:regression | fig01-regression | fig_structured_generate.py | – | 7 | stdout ของ fig_structured_generate.py |
| 1.9 | 19 | fig:roadmap | fig01-roadmap | figures/tikz/fig01-roadmap.tex | TikZ | – | แผนภาพเชิงแนวคิด ไม่มีการคำนวณ |
| 2.1 | 27 | fig:unit-balls | fig02-unit-balls | fig02_generate.py | – | – | stdout ของ fig02_generate.py |
| 2.2 | 29 | fig:projection | fig02-projection | fig02_generate.py | – | – | stdout ของ fig02_generate.py |
| 2.3 | 34 | fig:mb-frame | fig02-mb-frame | fig02_generate.py | – | – | stdout ของ fig02_generate.py |
| 2.4 | 36 | fig:dct-basis | fig02-dct-basis | fig02_generate.py | – | – | stdout ของ fig02_generate.py |
| 2.5 | 37 | fig:dct-demo | fig02-dct-demo | fig02_generate.py | – | – | stdout ของ fig02_generate.py |
| 2.6 | 38 | fig:wavelet2d | fig02-wavelet2d | fig02_generate.py | – | – | stdout ของ fig02_generate.py |
| 2.7 | 40 | fig:sigma2 | fig02-sigma2 | figures/tikz/fig02-sigma2.tex | TikZ | – | แผนภาพเชิงแนวคิด ไม่มีการคำนวณ |
| 2.8 | 41 | fig:decay | fig02-decay | fig02_generate.py | – | – | stdout ของ fig02_generate.py |
| 2.9 | 42 | fig:bestk | fig02-bestk | fig02_generate.py | – | – | stdout ของ fig02_generate.py |
| 2.10 | 43 | fig:structured | fig02-structured | fig_structured_generate.py | – | 3, 7 | stdout ของ fig_structured_generate.py |
| 2.11 | 45 | fig:lowrank | fig02-lowrank | fig_structured_generate.py | – | 3, 7 | stdout ของ fig_structured_generate.py |
| 3.1 | 51 | fig:geometry | fig03-geometry | fig03_generate.py | – | – | stdout ของ fig03_generate.py |
| 3.2 | 52 | fig:implications | fig03-implications | figures/tikz/fig03-implications.tex | TikZ | – | แผนภาพเชิงแนวคิด ไม่มีการคำนวณ |
| 3.3 | 62 | fig:rip | fig03-rip | fig03_generate.py | – | 5, 11 | stdout ของ fig03_generate.py |
| 3.4 | 65 | fig:matrices | fig03-matrices | fig03_generate.py | – | 2, 5, 11 | stdout ของ fig03_generate.py |
| 3.5 | 67 | fig:coherence | fig03-coherence | fig03_generate.py | – | 11 | stdout ของ fig03_generate.py |
| 3.6 | 71 | fig:phase | fig03-phase | fig03_generate.py | – | 1, 2, 5, 11 | stdout ของ fig03_generate.py |
| 3.7 | 72 | fig:noisy | fig03-noisy | fig03_generate.py | – | 1, 2, 5, 11, 21 | stdout ของ fig03_generate.py |
| 4.1 | 80 | fig:prox | fig04-prox | fig04_generate.py | prox | – | stdout ของ fig04_generate.py |
| 4.2 | 80 | fig:algo-map | fig04-algo-map | figures/tikz/fig04-algo-map.tex | TikZ | – | แผนภาพเชิงแนวคิด ไม่มีการคำนวณ |
| 4.3 | 84 | fig:certificate | fig04-certificate | fig04_generate.py | certificate | 52 | stdout ของ fig04_generate.py |
| 4.4 | 88 | fig:lasso-path | fig04-lasso-path | fig04_generate.py | lasso-path | 11 | stdout ของ fig04_generate.py |
| 4.5 | 89 | fig:convergence | fig04-convergence | fig04_generate.py | convergence | 7 | stdout ของ fig04_generate.py |
| 4.6 | 95 | fig:irls | fig04-irls | fig04_generate.py | irls | 3 | stdout ของ fig04_generate.py |
| 4.7 | 104 | fig:fnr | fig04-fnr | fig04_generate.py | fnr | 1, 100, 300 | stdout ของ fig04_generate.py |
| 4.8 | 107 | fig:focuss-mse | fig04-focuss-mse | fig04_generate.py | focuss-mse | 9 | stdout ของ fig04_generate.py |
| 4.9 | 110 | fig:runtime | fig04-runtime | fig04_generate.py | runtime | 13 | stdout ของ fig04_generate.py |
| 5.1 | 117 | fig:omp-trace | fig05-omp-trace | fig05_generate.py | omp-trace | 21 | stdout ของ fig05_generate.py |
| 5.2 | 125 | fig:amp | fig05-amp | fig05_generate.py | amp | 31, 32, 33 | fig05_cosamp-stop-2569-11-15.log |
| 5.3 | 131 | fig:author-threshold | fig05-author | fig05_generate.py | author | 7 | stdout ของ fig05_generate.py |
| 5.4 | 135 | fig:recovery | fig05-recovery | fig05_generate.py | recovery | 1000 | stdout ของ fig05_generate.py |
| 5.5 | 136 | fig:noisy-greedy | fig05-noisy | fig05_generate.py | noisy | 2000 | stdout ของ fig05_generate.py |
| 5.6 | 138 | fig:mmv | fig05-mmv | fig05_generate.py | mmv | 3000 | stdout ของ fig05_generate.py |
| 5.7 | 140 | fig:dictionary | fig05-dictionary | fig05_generate.py | dictionary | 4000, 5000 | stdout ของ fig05_generate.py |
| 5.8 | 143 | fig:runtime-greedy | fig05-runtime | fig05_generate.py | runtime | 13 | stdout ของ fig05_generate.py |
| 6.1 | 152 | fig:fourier-wavelet | fig06-coherence | fig06_generate.py | coherence | – | stdout ของ fig06_generate.py |
| 6.2 | 155 | fig:kspace-sampling | fig06-sampling | fig06_generate.py | sampling | 3 | stdout ของ fig06_generate.py |
| 6.3 | 160 | fig:mri | fig06-mri | fig06_generate.py | mri | 3 | fig06_mri.log, fig06_mri_200iter_old.log, fig06_mri_rerun-2569-11-15.log |
| 6.4 | 162 | fig:acceleration | fig06-acceleration | fig06_generate.py | mriacc | 3 | fig06_mriacc.log, fig06_selected_check_mriacc-2569-11-15.log |
| 6.5 | 164 | fig:threshold | fig06-threshold | fig06_generate.py | denoise | 1, 4 | stdout ของ fig06_generate.py |
| 6.6 | 166 | fig:denoise | fig06-denoise | fig06_generate.py | denoise | 1, 4 | stdout ของ fig06_generate.py |
| 6.7 | 168 | fig:irls-scale | fig06-irls | fig06_generate.py | irls | 6 | fig06_irls.log |
| 6.8 | 174 | fig:optical | fig06-optical | fig06_generate.py | irls | 6, 8, 9 | fig06_irls.log |
| 7.1 | 180 | fig:channel | fig07-channel | fig07_generate.py | channel | 11 | fig07_channel.log |
| 7.2 | 182 | fig:spectrum-sensing | fig07-spectrum | fig07_generate.py | spectrum | 12 | fig07_spectrum_conv.log |
| 7.3 | 185 | fig:doa | fig07-doa | fig07_generate.py | doa | 13 | fig07_doa.log, fig07_doa_600iter_old.log, fig07_doa_conv.log |
| 7.4 | 190 | fig:fdia | fig07-fdia | fig07_generate.py | fdia | 14 | fig07_fdia.log, fig07_fdia_rerun-2569-11-15.log |
| 8.1 | 206 | fig:gridless | fig08-gridless | fig08_generate.py | gridless | 80, 1000 | fig08_gridless.log, fig08_gridless_rerun-2569-11-15.log, fig08_gridless_run1.log |
| 8.2 | 210 | fig:onebit | fig08-onebit | fig08_generate.py | onebit | 5000, 9000 | fig08_onebit.log |
| 8.3 | 214 | fig:pnp | fig08-pnp | fig08_generate.py | pnp | 3 | fig08_pnp.log, fig08_pnp_200iter_old.log, fig08_pnp_rerun-2569-11-15.log |
| 8.4 | 218 | fig:unroll | fig08-unroll | fig08_generate.py | unroll | 88 | fig08_unroll.log |

## ตาราง

| ตาราง | หน้า | ป้าย | ประเภท |
|---|---|---|---|
| 1.1 | 11 | tab:paradigm | สรุปจากเอกสารอ้างอิง |
| 1.2 | 16 | tab:family | จัดทำโดยผู้แต่ง (ตัวเลขจากการทดลองพิมพ์ออกโดยสคริปต์ของบทเดียวกัน) |
| 1.3 | 17 | tab:applications | จัดทำโดยผู้แต่ง (ตัวเลขจากการทดลองพิมพ์ออกโดยสคริปต์ของบทเดียวกัน) |
| 1.4 | 20 | tab:roadmap | จัดทำโดยผู้แต่ง (ตัวเลขจากการทดลองพิมพ์ออกโดยสคริปต์ของบทเดียวกัน) |
| 2.1 | 27 | tab:norm-summary | จัดทำโดยผู้แต่ง (ตัวเลขจากการทดลองพิมพ์ออกโดยสคริปต์ของบทเดียวกัน) |
| 2.2 | 31 | tab:vector-matrix | จัดทำโดยผู้แต่ง (ตัวเลขจากการทดลองพิมพ์ออกโดยสคริปต์ของบทเดียวกัน) |
| 2.3 | 39 | tab:basis-guide | จัดทำโดยผู้แต่ง (ตัวเลขจากการทดลองพิมพ์ออกโดยสคริปต์ของบทเดียวกัน) |
| 3.1 | 66 | tab:ensembles | จัดทำโดยผู้แต่ง (ตัวเลขจากการทดลองพิมพ์ออกโดยสคริปต์ของบทเดียวกัน) |
| 3.2 | 69 | tab:conditions | จัดทำโดยผู้แต่ง (ตัวเลขจากการทดลองพิมพ์ออกโดยสคริปต์ของบทเดียวกัน) |
| 3.3 | 70 | tab:statdim | จัดทำโดยผู้แต่ง (ตัวเลขจากการทดลองพิมพ์ออกโดยสคริปต์ของบทเดียวกัน) |
| 4.1 | 77 | tab:problem-forms | จัดทำโดยผู้แต่ง (ตัวเลขจากการทดลองพิมพ์ออกโดยสคริปต์ของบทเดียวกัน) |
| 4.2 | 96 | tab:paper-notation | จัดทำโดยผู้แต่ง (ตัวเลขจากการทดลองพิมพ์ออกโดยสคริปต์ของบทเดียวกัน) |
| 4.3 | 98 | tab:irls-image | ตัวเลขจากบทความต้นทาง (ข้อมูลดิบไม่รวมในแพ็กเกจ) |
| 4.4 | 103 | tab:reweighting-family | จัดทำโดยผู้แต่ง (ตัวเลขจากการทดลองพิมพ์ออกโดยสคริปต์ของบทเดียวกัน) |
| 4.5 | 104 | tab:fnr-noisy | จัดทำโดยผู้แต่ง (ตัวเลขจากการทดลองพิมพ์ออกโดยสคริปต์ของบทเดียวกัน) |
| 4.6 | 109 | tab:algo-compare | จัดทำโดยผู้แต่ง (ตัวเลขจากการทดลองพิมพ์ออกโดยสคริปต์ของบทเดียวกัน) |
| 5.1 | 115 | tab:greedy-taxonomy | จัดทำโดยผู้แต่ง (ตัวเลขจากการทดลองพิมพ์ออกโดยสคริปต์ของบทเดียวกัน) |
| 5.2 | 126 | tab:paper-notation-greedy | จัดทำโดยผู้แต่ง (ตัวเลขจากการทดลองพิมพ์ออกโดยสคริปต์ของบทเดียวกัน) |
| 5.3 | 127 | tab:homotopy-family | สรุปจากเอกสารอ้างอิง |
| 5.4 | 130 | tab:soft-real | ตัวเลขจากบทความต้นทาง (ข้อมูลดิบไม่รวมในแพ็กเกจ) |
| 5.5 | 132 | tab:homotopy-scale | จัดทำโดยผู้แต่ง (ตัวเลขจากการทดลองพิมพ์ออกโดยสคริปต์ของบทเดียวกัน) |
| 5.6 | 134 | tab:iht-fnr | จัดทำโดยผู้แต่ง (ตัวเลขจากการทดลองพิมพ์ออกโดยสคริปต์ของบทเดียวกัน) |
| 5.7 | 142 | tab:grand-compare | จัดทำโดยผู้แต่ง (ตัวเลขจากการทดลองพิมพ์ออกโดยสคริปต์ของบทเดียวกัน) |
| 6.1 | 149 | tab:imaging-operators | จัดทำโดยผู้แต่ง (ตัวเลขจากการทดลองพิมพ์ออกโดยสคริปต์ของบทเดียวกัน) |
| 6.2 | 156 | tab:tpsf | จัดทำโดยผู้แต่ง (ตัวเลขจากการทดลองพิมพ์ออกโดยสคริปต์ของบทเดียวกัน) |
| 6.3 | 161 | tab:mri | จัดทำโดยผู้แต่ง (ตัวเลขจากการทดลองพิมพ์ออกโดยสคริปต์ของบทเดียวกัน) |
| 6.4 | 166 | tab:denoise | จัดทำโดยผู้แต่ง (ตัวเลขจากการทดลองพิมพ์ออกโดยสคริปต์ของบทเดียวกัน) |
| 6.5 | 168 | tab:irls-scale | จัดทำโดยผู้แต่ง (ตัวเลขจากการทดลองพิมพ์ออกโดยสคริปต์ของบทเดียวกัน) |
| 6.6 | 174 | tab:cassi | จัดทำโดยผู้แต่ง (ตัวเลขจากการทดลองพิมพ์ออกโดยสคริปต์ของบทเดียวกัน) |
| 7.1 | 191 | tab:paper-notation-fdia | จัดทำโดยผู้แต่ง (ตัวเลขจากการทดลองพิมพ์ออกโดยสคริปต์ของบทเดียวกัน) |
| 7.2 | 197 | tab:fdia-reported | ตัวเลขจากบทความต้นทาง (ข้อมูลดิบไม่รวมในแพ็กเกจ) |
| 8.1 | 219 | tab:frontier-map | จัดทำโดยผู้แต่ง (ตัวเลขจากการทดลองพิมพ์ออกโดยสคริปต์ของบทเดียวกัน) |

## ข้อจำกัดด้านข้อมูล

- ตารางที่ 4.3, 5.4 และ 7.2 คัดตัวเลขจากบทความที่ผู้แต่งเป็นผู้เขียนร่วม (อ้างอิงไว้ในคำบรรยาย) ข้อมูลดิบและโค้ดต้นทางของบทความเหล่านั้นไม่รวมอยู่ในแพ็กเกจนี้ จึงทำซ้ำได้เฉพาะเมื่อได้ข้อมูลจากผู้เขียนบทความ ส่วนตัวเลขที่หนังสือคำนวณเองทุกค่ามีสคริปต์และ seed ตามตารางข้างต้น
- ตัวเลขในเฉลยปัญหาที่ 1.6, 4.4 และ 4.6 มาจาก `exercises_check.py` (`ex1-6`, `ex4-4`) และ `fig04_generate.py rwl1`
