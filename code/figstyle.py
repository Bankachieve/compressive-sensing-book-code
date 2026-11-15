# -*- coding: utf-8 -*-
"""รูปแบบกลางของกราฟทั้งเล่ม (นำเข้าโดย fig*_generate.py ทุกไฟล์)

ทำงานอัตโนมัติก่อนบันทึกภาพทุกครั้ง (ครอบ Figure.savefig):
  1) เส้นกริดพื้นหลังสีเทาอ่อนบนทุกแกนที่เป็นกราฟ (ไม่ใส่บนภาพ แผนที่ความร้อน แถบสี หรือแกนที่ไม่มีขีดบอกค่า)
  2) ย้ายคำอธิบายสัญลักษณ์ (legend) ไปยังมุมที่ไม่ทับเส้นกราฟ จุดข้อมูล แท่ง หรือข้อความ
     ถ้าไม่มีมุมว่าง จะจัดเป็นแถวแนวนอนด้านบนแล้วขยายขอบบนของแกนให้มีที่ว่างพอ
"""
import numpy as np
import matplotlib
from matplotlib.figure import Figure
from matplotlib.collections import QuadMesh, LineCollection, PathCollection, PolyCollection
from matplotlib.transforms import Bbox

GRID = dict(color="#9AA3AF", linewidth=0.4, alpha=0.45)
LOCS = ["upper right", "upper left", "lower right", "lower left",
        "center right", "center left", "lower center", "upper center"]
PAD = 3.0   # ระยะกันชนรอบกรอบ legend (พิกเซล)


def _is_plot_axes(ax):
    if getattr(ax, "name", "") != "rectilinear" or not ax.axison:
        return False
    if ax.images or any(isinstance(c, QuadMesh) for c in ax.collections):
        return False
    return len(ax.get_xticks()) > 0 and len(ax.get_yticks()) > 0


def _densify(d, per=14):
    d = d[np.isfinite(d).all(axis=1)]
    if len(d) < 2 or len(d) > 600:
        return d
    t = np.linspace(0, 1, per)[:, None]
    seg = [d[i] + t * (d[i + 1] - d[i]) for i in range(len(d) - 1)]
    return np.vstack(seg)


def _obstacles(ax, renderer):
    """จุด (พิกัดจอ) ของเส้น/จุด/พื้นที่ และกรอบของแท่งกับข้อความ ภายในแกน"""
    pts, boxes = [], []
    for ln in ax.lines:
        if ln.get_transform() != ax.transData:
            continue    # เส้นอ้างอิงจาก axhline/axvline ลากตลอดแกน ไม่นับเป็นเส้นข้อมูล
        xy = np.asarray(ln.get_xydata(), dtype=float)
        if xy.size:
            pts.append(_densify(ln.get_transform().transform(xy)))
    for c in ax.collections:
        try:
            if isinstance(c, LineCollection):
                tr = c.get_transform()
                for s in c.get_segments():
                    if len(s):
                        pts.append(_densify(tr.transform(np.asarray(s, dtype=float))))
            elif isinstance(c, PathCollection):
                off = np.asarray(c.get_offsets(), dtype=float)
                if off.size:
                    pts.append(c.get_offset_transform().transform(off))
            elif isinstance(c, PolyCollection):
                tr = c.get_transform()
                for p in c.get_paths():
                    pts.append(_densify(tr.transform(p.vertices), per=6))
        except Exception:
            pass
    for p in ax.patches:
        try:
            boxes.append(p.get_window_extent(renderer))
        except Exception:
            pass
    for t in ax.texts:
        if t.get_text().strip():
            try:
                boxes.append(t.get_window_extent(renderer))
            except Exception:
                pass
    pts = np.vstack([p for p in pts if len(p)]) if any(len(p) for p in pts) else np.zeros((0, 2))
    axbb = ax.get_window_extent(renderer)
    keep = ((pts[:, 0] >= axbb.x0 - 1) & (pts[:, 0] <= axbb.x1 + 1) &
            (pts[:, 1] >= axbb.y0 - 1) & (pts[:, 1] <= axbb.y1 + 1)) if len(pts) else []
    return (pts[keep] if len(pts) else pts), boxes


def _overlaps(leg, ax, renderer):
    bb = leg.get_window_extent(renderer)
    bb = Bbox.from_extents(bb.x0 - PAD, bb.y0 - PAD, bb.x1 + PAD, bb.y1 + PAD)
    pts, boxes = _obstacles(ax, renderer)
    if len(pts) and np.any((pts[:, 0] > bb.x0) & (pts[:, 0] < bb.x1) &
                           (pts[:, 1] > bb.y0) & (pts[:, 1] < bb.y1)):
        return True
    return any(bb.overlaps(b) for b in boxes if b.width > 0 and b.height > 0)


def _frame(leg):
    leg.set_frame_on(True)
    fr = leg.get_frame()
    fr.set_facecolor("white"); fr.set_edgecolor("#9AA3AF")
    fr.set_linewidth(0.4); fr.set_alpha(1.0)
    leg.set_zorder(20)


def _fit_legend(ax, fig):
    leg = ax.get_legend()
    if leg is None:
        return
    canvas = fig.canvas
    _frame(leg)
    canvas.draw(); r = canvas.get_renderer()
    if not _overlaps(leg, ax, r):
        return
    for loc in LOCS:
        leg.set_loc(loc)
        canvas.draw()
        if not _overlaps(leg, ax, r):
            return
    # ไม่มีมุมว่าง: จัดแนวนอนด้านบนและขยายขอบบนของแกน
    handles = leg.legend_handles
    labels = [t.get_text() for t in leg.get_texts()]
    fs = leg.get_texts()[0].get_fontsize() if labels else 7
    axw = ax.get_window_extent(r).width
    new = None
    for ncol in range(len(labels), 0, -1):
        new = ax.legend(handles, labels, loc="upper center", ncol=ncol, fontsize=fs,
                        columnspacing=1.0, handlelength=1.8, handletextpad=0.5,
                        borderaxespad=0.35, labelspacing=0.3)
        _frame(new); canvas.draw()
        if new.get_window_extent(r).width <= 0.96 * axw:
            break
    linear = ax.get_yscale() == "linear"
    y0, y1 = ax.get_ylim()
    ticks = [t for t in ax.get_yticks() if min(y0, y1) - 1e-12 <= t <= max(y0, y1) + 1e-12]
    for _ in range(9):     # ขยายได้ไม่เกินประมาณร้อยละ 45 ของความสูงแกน
        if not _overlaps(new, ax, r):
            break
        bb = ax.get_window_extent(r)
        top = ax.transData.inverted().transform((bb.x0, bb.y1 + 0.05 * bb.height))[1]
        ax.set_ylim(ax.get_ylim()[0], top)
        if linear:
            ax.set_yticks(ticks)
        canvas.draw()


def apply(fig):
    for ax in fig.axes:
        if _is_plot_axes(ax):
            ax.grid(True, which="major", **GRID)
            ax.set_axisbelow(True)
    for ax in fig.axes:
        try:
            _fit_legend(ax, fig)
        except Exception as e:      # ไม่ให้การจัดรูปแบบทำให้การสร้างภาพล้มเหลว
            print("figstyle: legend not adjusted:", e)


_orig_savefig = Figure.savefig

def _savefig(self, *args, **kwargs):
    if not getattr(self, "_figstyle_done", False):
        self._figstyle_done = True
        try:
            apply(self)
        except Exception as e:
            print("figstyle: skipped:", e)
    return _orig_savefig(self, *args, **kwargs)

Figure.savefig = _savefig
