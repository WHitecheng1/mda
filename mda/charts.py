"""图表绘制：SVG（零依赖，可嵌进 HTML 报告）与 PNG（Pillow）双后端。

设计：先把所有图元记录成"显示列表"（display list），再分别交给
SVG 渲染器或 Pillow 渲染器输出，因此两种后端图形完全一致，
且不依赖 matplotlib。

支持的图形：折线/散点（带误差棒）、柱状（带误差棒）、直方图（含正态曲线）、
箱线图、热力图（偏差矩阵）、Bland-Altman、残差图、帕累托（不确定度贡献）。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = ["Plot", "PALETTE", "render_svg", "render_png", "find_cjk_font", "figure_html"]

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
           "#8c564b", "#17becf", "#e377c2", "#7f7f7f", "#bcbd22"]

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\Deng.ttf", r"C:\Windows\Fonts\simsun.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]


def find_cjk_font() -> str | None:
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


# --------------------------------------------------------------------------
# 显示列表与渲染器
# --------------------------------------------------------------------------
@dataclass
class Prim:
    kind: str                       # line | poly | rect | circle | text
    pts: list
    style: dict = field(default_factory=dict)


def _esc(s: Any) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def render_svg(prims: Sequence[Prim], width: int, height: int, bg: str = "#ffffff") -> str:
    """把显示列表渲染成 SVG 文本。"""
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" font-family="Microsoft YaHei, SimHei, '
           f'Arial, sans-serif">',
           f'<rect x="0" y="0" width="{width}" height="{height}" fill="{bg}"/>']
    for p in prims:
        st = p.style
        if p.kind == "line":
            (x1, y1), (x2, y2) = p.pts
            dash = f' stroke-dasharray="{st["dash"]}"' if st.get("dash") else ""
            out.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                       f'stroke="{st.get("color", "#333")}" stroke-width="{st.get("width", 1)}" '
                       f'opacity="{st.get("opacity", 1)}"{dash} stroke-linecap="round"/>')
        elif p.kind == "poly":
            pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in p.pts)
            tag = "polygon" if st.get("closed") else "polyline"
            dash = f' stroke-dasharray="{st["dash"]}"' if st.get("dash") else ""
            fill = st.get("fill", "none")
            out.append(f'<{tag} points="{pts}" fill="{fill}" stroke="{st.get("color", "#333")}" '
                       f'stroke-width="{st.get("width", 2)}" opacity="{st.get("opacity", 1)}"'
                       f'{dash} stroke-linejoin="round"/>')
        elif p.kind == "rect":
            x, y, w, h = p.pts
            if w < 0:
                x, w = x + w, -w
            if h < 0:
                y, h = y + h, -h
            fill = st.get("fill", "none")
            stroke = st.get("stroke", "none")
            out.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(w, 0):.2f}" height="{max(h, 0):.2f}" '
                       f'fill="{fill}" stroke="{stroke}" stroke-width="{st.get("width", 1)}" '
                       f'opacity="{st.get("opacity", 1)}" rx="{st.get("rx", 0)}"/>')
        elif p.kind == "circle":
            cx, cy, r = p.pts
            out.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{max(r, 0.5):.2f}" '
                       f'fill="{st.get("fill", "#333")}" stroke="{st.get("stroke", "none")}" '
                       f'stroke-width="{st.get("width", 1)}" opacity="{st.get("opacity", 1)}"/>')
        elif p.kind == "text":
            x, y = p.pts
            anchor = {"start": "start", "middle": "middle", "end": "end"}.get(st.get("anchor", "start"), "start")
            baseline = {"baseline": "auto", "middle": "middle", "top": "hanging"}.get(
                st.get("baseline", "baseline"), "auto")
            rot = f' transform="rotate({st["rotate"]} {x:.2f} {y:.2f})"' if st.get("rotate") else ""
            weight = ' font-weight="bold"' if st.get("bold") else ""
            out.append(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{st.get("size", 12)}" '
                       f'fill="{st.get("color", "#222")}" text-anchor="{anchor}" '
                       f'dominant-baseline="{baseline}" opacity="{st.get("opacity", 1)}"{weight}{rot}>'
                       f'{_esc(st.get("text", ""))}</text>')
    out.append("</svg>")
    return "\n".join(out)


_FONT_CACHE: dict[tuple, Any] = {}


def _pil_font(size: int, bold: bool = False):
    from PIL import ImageFont
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    path = find_cjk_font()
    font = None
    if path:
        try:
            font = ImageFont.truetype(path, size)
        except Exception:
            font = None
    if font is None:
        font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


def render_png(prims: Sequence[Prim], width: int, height: int, path: str,
               bg: str = "#ffffff", scale: int = 2) -> str:
    """把显示列表渲染成 PNG（默认 2 倍超采样后缩小，线条更平滑）。"""
    from PIL import Image, ImageDraw
    ss = max(1, int(scale))
    W, H = width * ss, height * ss
    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img, "RGBA")

    def sc(pts):
        return [(float(x) * ss, float(y) * ss) for x, y in pts]

    for p in prims:
        st = p.style
        color = st.get("color", "#333")
        lw = max(1, int(round(st.get("width", 1) * ss)))
        if p.kind == "line":
            (x1, y1), (x2, y2) = sc(p.pts)
            if st.get("dash"):
                _dashed(draw, (x1, y1, x2, y2), color, lw, st["dash"], ss)
            else:
                draw.line([x1, y1, x2, y2], fill=color, width=lw)
        elif p.kind == "poly":
            pts = sc(p.pts)
            if st.get("closed"):
                draw.polygon(pts, fill=st.get("fill", None), outline=color)
                if lw > 1 and st.get("fill", "none") != "none":
                    draw.line(pts + [pts[0]], fill=color, width=lw)
            else:
                draw.line(pts, fill=color, width=lw, joint="curve")
        elif p.kind == "rect":
            x, y, w, h = p.pts
            box = [min(x, x + w) * ss, min(y, y + h) * ss,
                   max(x, x + w) * ss, max(y, y + h) * ss]
            fill = st.get("fill", None)
            outline = st.get("stroke", None)
            if (fill and fill != "none") or (outline and outline != "none"):
                draw.rectangle(box, fill=None if fill in (None, "none") else fill,
                               outline=None if outline in (None, "none") else outline,
                               width=lw)
        elif p.kind == "circle":
            cx, cy, r = p.pts
            box = [(cx - r) * ss, (cy - r) * ss, (cx + r) * ss, (cy + r) * ss]
            draw.ellipse(box, fill=st.get("fill", "#333"),
                         outline=None if st.get("stroke") in (None, "none") else st.get("stroke"),
                         width=lw)
        elif p.kind == "text":
            x, y = p.pts[0] * ss, p.pts[1] * ss
            text = str(st.get("text", ""))
            size = int(st.get("size", 12) * ss)
            font = _pil_font(size, bool(st.get("bold")))
            anchor = {"start": "l", "middle": "m", "end": "r"}.get(st.get("anchor", "start"), "l")
            vmap = {"baseline": "s", "middle": "m", "top": "a"}
            anchor += vmap.get(st.get("baseline", "baseline"), "s")
            color = st.get("color", "#222")
            if st.get("rotate"):
                ang = float(st["rotate"])
                tmp = Image.new("RGBA", (int(size * len(text) * 1.2) + 40, int(size * 2.2) + 20), (0, 0, 0, 0))
                td = ImageDraw.Draw(tmp)
                td.text((tmp.width // 2, tmp.height // 2), text, font=font, fill=color, anchor="mm")
                tmp = tmp.rotate(ang, expand=True, resample=Image.BICUBIC)
                img.paste(tmp, (int(x - tmp.width / 2), int(y - tmp.height / 2)), tmp)
                draw = ImageDraw.Draw(img, "RGBA")
            else:
                draw.text((x, y), text, font=font, fill=color, anchor=anchor)
    if ss > 1:
        img = img.resize((width, height), Image.LANCZOS)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    img.save(path)
    return path


def _dashed(draw, box, color, width, dash, ss):
    x1, y1, x2, y2 = box
    dx, dy = x2 - x1, y2 - y1
    dist = math.hypot(dx, dy)
    if dist == 0:
        return
    on, off = (dash if isinstance(dash, (tuple, list)) else (4, 3))[:2]
    on, off = on * ss, off * ss
    pos, draw_on = 0.0, True
    while pos < dist:
        seg = on if draw_on else off
        end = min(pos + seg, dist)
        if draw_on:
            draw.line([x1 + dx * pos / dist, y1 + dy * pos / dist,
                       x1 + dx * end / dist, y1 + dy * end / dist], fill=color, width=width)
        pos = end
        draw_on = not draw_on


# --------------------------------------------------------------------------
# 坐标轴数学
# --------------------------------------------------------------------------
def nice_ticks(lo: float, hi: float, target: int = 6) -> list[float]:
    if not np.isfinite(lo) or not np.isfinite(hi) or hi < lo:
        return []
    if hi == lo:
        if lo == 0:
            return [0.0]
        lo, hi = lo - abs(lo) * 0.5, hi + abs(hi) * 0.5
    raw = (hi - lo) / max(target, 2)
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            step = m * mag
            break
    else:
        step = 10 * mag
    start = math.ceil(lo / step) * step
    ticks, v = [], start
    while v <= hi + step * 1e-6:
        ticks.append(round(v, 12))
        v += step
    return ticks


def fmt_num(v: float, span: float = 0.0) -> str:
    if not np.isfinite(v):
        return "—"
    if v == 0:
        return "0"
    a = abs(v)
    if (span and span < 1e-3) or a >= 1e5 or a < 1e-4:
        return f"{v:.3g}"
    if span and span < 0.1:
        return f"{v:.4g}"
    return f"{v:g}"


def _cmap(name: str, t: float) -> str:
    t = float(np.clip(t, 0.0, 1.0))
    if name in ("RdBu", "RdBu_r", "bwr"):
        if t < 0.5:
            f = t / 0.5
            a, b = (0.13, 0.30, 0.66), (1.0, 1.0, 1.0)
        else:
            f = (t - 0.5) / 0.5
            a, b = (1.0, 1.0, 1.0), (0.70, 0.02, 0.15)
        rgb = [a[i] + (b[i] - a[i]) * f for i in range(3)]
    elif name == "Blues":
        a, b = (0.97, 0.97, 0.98), (0.03, 0.19, 0.42)
        rgb = [a[i] + (b[i] - a[i]) * t for i in range(3)]
    else:  # viridis 近似
        stops = [(0.267, 0.005, 0.329), (0.188, 0.408, 0.556), (0.208, 0.718, 0.472),
                 (0.993, 0.906, 0.144)]
        pos = t * (len(stops) - 1)
        i = min(int(pos), len(stops) - 2)
        f = pos - i
        rgb = [stops[i][k] + (stops[i + 1][k] - stops[i][k]) * f for k in range(3)]
    return "#%02x%02x%02x" % tuple(int(round(c * 255)) for c in rgb)


def _marker_prims(cx: float, cy: float, marker: str, size: float, color: str) -> list[Prim]:
    if not marker or marker in ("none", ""):
        return []
    s = size
    if marker == "o":
        return [Prim("circle", [cx, cy, s], {"fill": color, "stroke": "#fff", "width": 0.8})]
    if marker == "s":
        return [Prim("rect", [cx - s, cy - s, 2 * s, 2 * s], {"fill": color, "stroke": "#fff", "width": 0.8})]
    if marker == "^":
        return [Prim("poly", [(cx, cy - s * 1.2), (cx + s * 1.1, cy + s), (cx - s * 1.1, cy + s)],
                     {"closed": True, "fill": color, "color": "#fff", "width": 0.8})]
    if marker == "v":
        return [Prim("poly", [(cx, cy + s * 1.2), (cx + s * 1.1, cy - s), (cx - s * 1.1, cy - s)],
                     {"closed": True, "fill": color, "color": "#fff", "width": 0.8})]
    if marker == "D":
        return [Prim("poly", [(cx, cy - s * 1.2), (cx + s * 1.2, cy), (cx, cy + s * 1.2), (cx - s * 1.2, cy)],
                     {"closed": True, "fill": color, "color": "#fff", "width": 0.8})]
    if marker == "x":
        return [Prim("line", [(cx - s, cy - s), (cx + s, cy + s)], {"color": color, "width": 1.8}),
                Prim("line", [(cx - s, cy + s), (cx + s, cy - s)], {"color": color, "width": 1.8})]
    if marker == "+":
        return [Prim("line", [(cx - s, cy), (cx + s, cy)], {"color": color, "width": 1.8}),
                Prim("line", [(cx, cy - s), (cx, cy + s)], {"color": color, "width": 1.8})]
    return [Prim("circle", [cx, cy, s], {"fill": color})]


# --------------------------------------------------------------------------
# 绘图对象
# --------------------------------------------------------------------------
class Plot:
    """一张图。先用 add_* 描述数据，再 render() 输出。"""

    def __init__(self, title: str = "", xlabel: str = "", ylabel: str = "",
                 width: int = 860, height: int = 520, *, log_x: bool = False,
                 log_y: bool = False, grid: bool = True, legend: bool = True,
                 xlim: tuple | None = None, ylim: tuple | None = None,
                 x_tick_rotate: float = 0.0, legend_loc: str = "auto"):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.width, self.height = width, height
        self.log_x, self.log_y = log_x, log_y
        self.grid = grid
        self.show_legend = legend
        self.xlim, self.ylim = xlim, ylim
        self.x_tick_rotate = x_tick_rotate
        self.legend_loc = legend_loc
        self.items: list[dict] = []
        self._cat_labels: list[str] | None = None
        self._cat_labels_y: list[str] | None = None
        self._notes: list[str] = []

    # ---------------- 数据描述 ----------------
    def line(self, x, y, label: str | None = None, color: str | None = None,
             width: float = 2.0, dash=None, marker: str | None = None, markersize: float = 3.5):
        self.items.append(dict(kind="line", x=np.asarray(x, dtype="float64"),
                               y=np.asarray(y, dtype="float64"), label=label, color=color,
                               width=width, dash=dash, marker=marker, markersize=markersize))
        return self

    def errorbar(self, x, y, yerr=None, label: str | None = None, color: str | None = None,
                 marker: str = "o", markersize: float = 4.0, capsize: float = 5.0,
                 connect: bool = True, dash=None):
        self.items.append(dict(kind="errorbar", x=np.asarray(x, dtype="float64"),
                               y=np.asarray(y, dtype="float64"),
                               yerr=None if yerr is None else np.asarray(yerr, dtype="float64"),
                               label=label, color=color, marker=marker, markersize=markersize,
                               capsize=capsize, connect=connect, dash=dash))
        return self

    def scatter(self, x, y, label: str | None = None, color: str | None = None,
                marker: str = "o", markersize: float = 4.0, opacity: float = 1.0):
        self.items.append(dict(kind="scatter", x=np.asarray(x, dtype="float64"),
                               y=np.asarray(y, dtype="float64"), label=label, color=color,
                               marker=marker, markersize=markersize, opacity=opacity))
        return self

    def bars(self, labels: Sequence[str], values, errors=None, label: str | None = None,
             colors: Sequence[str] | str | None = None, horizontal: bool = False):
        self.items.append(dict(kind="bars", labels=[str(s) for s in labels],
                               values=np.asarray(values, dtype="float64"),
                               errors=None if errors is None else np.asarray(errors, dtype="float64"),
                               label=label, colors=colors, horizontal=horizontal))
        if horizontal:
            self._cat_labels_y = [str(s) for s in labels]
        else:
            self._cat_labels = [str(s) for s in labels]
        return self

    def hist(self, values, bins: int | Sequence[float] = 12, label: str | None = None,
             color: str | None = None, density: bool = True, normal_curve: bool = True):
        self.items.append(dict(kind="hist", values=np.asarray(values, dtype="float64"),
                               bins=bins, label=label, color=color, density=density,
                               normal_curve=normal_curve))
        return self

    def box(self, groups: Mapping[str, Any], label: str | None = None, colors=None):
        self.items.append(dict(kind="box", groups={str(k): np.asarray(v, dtype="float64")
                                                   for k, v in groups.items()},
                               label=label, colors=colors))
        self._cat_labels = [str(k) for k in groups.keys()]
        return self

    def heatmap(self, matrix, row_labels: Sequence[str], col_labels: Sequence[str],
                cell_fmt: str = "{:.2f}", cmap: str = "RdBu", vmin=None, vmax=None,
                annotate: bool = True, row_label: str = "", col_label: str = ""):
        self.items.append(dict(kind="heatmap", matrix=np.asarray(matrix, dtype="float64"),
                               row_labels=[str(s) for s in row_labels],
                               col_labels=[str(s) for s in col_labels], cell_fmt=cell_fmt,
                               cmap=cmap, vmin=vmin, vmax=vmax, annotate=annotate,
                               row_label=row_label, col_label=col_label))
        return self

    def hline(self, y: float, label: str | None = None, color: str = "#666666", dash=(5, 4),
              width: float = 1.2):
        self.items.append(dict(kind="hline", y=float(y), label=label, color=color,
                               dash=dash, width=width))
        return self

    def vline(self, x: float, label: str | None = None, color: str = "#666666", dash=(5, 4),
              width: float = 1.2):
        self.items.append(dict(kind="vline", x=float(x), label=label, color=color,
                               dash=dash, width=width))
        return self

    def note(self, text: str):
        self._notes.append(text)
        return self

    # ---------------- 渲染 ----------------
    def render(self, path: str, fmt: str | None = None) -> str:
        """渲染到文件。fmt 只支持 'png' / 'svg'（缺省按扩展名判断）。

        以前任何非 svg 的 fmt 都会静默写成 PNG（render('a.jpg','jpg') 得到一张
        伪装成 jpg 的 PNG），现在直接报错，避免调用方误以为格式被尊重。
        """
        fmt = (fmt or os.path.splitext(path)[1].lstrip(".") or "png").lower()
        if fmt not in ("png", "svg"):
            raise ValueError(f"不支持的图片格式 {fmt!r}：只支持 'png' 与 'svg'")
        prims = self._build()
        if fmt == "svg":
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(render_svg(prims, self.width, self.height))
            return path
        return render_png(prims, self.width, self.height, path)

    def svg(self) -> str:
        return render_svg(self._build(), self.width, self.height)

    # ---------------- 内部实现 ----------------
    def _build(self) -> list[Prim]:
        heat = [it for it in self.items if it["kind"] == "heatmap"]
        if heat:
            return self._build_heatmap(heat[-1])
        return self._build_axes()

    def _data_limits(self):
        xs, ys = [], []
        for it in self.items:
            k = it["kind"]
            if k in ("line", "errorbar", "scatter"):
                xs.append(it["x"]); ys.append(it["y"])
                if k == "errorbar" and it["yerr"] is not None:
                    ye = np.abs(it["yerr"])
                    ys.extend([it["y"] - ye, it["y"] + ye])
            elif k == "hist":
                v = it["values"][np.isfinite(it["values"])]
                if v.size:
                    xs.append(np.array([v.min(), v.max()]))
                    ys.append(np.array([0.0, 1.0]))
            elif k == "hline":
                ys.append(np.array([it["y"]]))
            elif k == "vline":
                xs.append(np.array([it["x"]]))
            elif k == "bars":
                if it["horizontal"]:
                    # 横向条形图的类目在 y 轴，x 轴为数值
                    xs.append(np.array([0.0, float(np.nanmax(it["values"]))]))
                else:
                    xs.append(np.arange(len(it["labels"]), dtype="float64"))
                    ys.append(it["values"])
        return xs, ys

    def _build_axes(self) -> list[Prim]:
        W, H = self.width, self.height
        left, right, top, bottom = 84, 26, 58, 66
        if self.show_legend:
            right = 26
        pw, ph = W - left - right, H - top - bottom
        prims: list[Prim] = []

        xs_all, ys_all = self._data_limits()
        cat_mode = self._cat_labels is not None
        cat_y = self._cat_labels_y is not None

        # --- x 范围
        if cat_mode:
            xlo, xhi = -0.6, len(self._cat_labels) - 0.4
        else:
            if self.xlim:
                xlo, xhi = self.xlim
            else:
                cat = np.concatenate([a[np.isfinite(a)] for a in xs_all if a.size]) if xs_all else np.array([0.0, 1.0])
                cat = cat[np.isfinite(cat)]
                if cat.size == 0:
                    cat = np.array([0.0, 1.0])
                xlo, xhi = float(cat.min()), float(cat.max())
                if xhi == xlo:
                    xlo, xhi = xlo - 0.5, xhi + 0.5
                pad = (xhi - xlo) * 0.05
                xlo, xhi = xlo - pad, xhi + pad
        # --- y 范围
        if cat_y:
            ylo, yhi = -0.6, len(self._cat_labels_y) - 0.4
        elif self.ylim:
            ylo, yhi = self.ylim
        else:
            cat = np.concatenate([a[np.isfinite(a)] for a in ys_all if a.size]) if ys_all else np.array([0.0, 1.0])
            cat = cat[np.isfinite(cat)]
            if cat.size == 0:
                cat = np.array([0.0, 1.0])
            ylo, yhi = float(cat.min()), float(cat.max())
            if yhi == ylo:
                ylo, yhi = ylo - 0.5, yhi + 0.5
            pad = (yhi - ylo) * 0.10
            ylo, yhi = ylo - pad, yhi + pad

        fx = (lambda v: math.log10(v)) if self.log_x else (lambda v: v)
        fy = (lambda v: math.log10(v)) if self.log_y else (lambda v: v)
        x0f, x1f = fx(xlo), fx(xhi)
        y0f, y1f = fy(ylo), fy(yhi)

        def X(v):
            return left + (fx(v) - x0f) / (x1f - x0f) * pw

        def Y(v):
            return top + ph - (fy(v) - y0f) / (y1f - y0f) * ph

        # 背景与网格
        prims.append(Prim("rect", [left, top, pw, ph], {"fill": "#fcfcfd"}))
        xticks = nice_ticks(xlo, xhi, 7) if not cat_mode else []
        yticks = [] if cat_y else nice_ticks(ylo, yhi, 6)
        if self.grid:
            for t in xticks:
                prims.append(Prim("line", [(X(t), top), (X(t), top + ph)],
                                  {"color": "#e8e8ee", "width": 1}))
            for t in yticks:
                prims.append(Prim("line", [(left, Y(t)), (left + pw, Y(t))],
                                  {"color": "#e8e8ee", "width": 1}))
        # 轴框
        prims.append(Prim("rect", [left, top, pw, ph], {"fill": "none", "stroke": "#98a2b3", "width": 1.2}))

        # 刻度与标签
        span_x, span_y = xhi - xlo, yhi - ylo
        for t in xticks:
            if not (xlo - 1e-12 <= t <= xhi + 1e-12):
                continue
            prims.append(Prim("line", [(X(t), top + ph), (X(t), top + ph + 5)], {"color": "#98a2b3", "width": 1}))
            prims.append(Prim("text", [X(t), top + ph + 9], {"text": fmt_num(t, span_x), "size": 12,
                                                             "anchor": "middle", "baseline": "top",
                                                             "color": "#475467",
                                                             "rotate": self.x_tick_rotate}))
        for t in yticks:
            if not (ylo - 1e-12 <= t <= yhi + 1e-12):
                continue
            prims.append(Prim("line", [(left - 5, Y(t)), (left, Y(t))], {"color": "#98a2b3", "width": 1}))
            prims.append(Prim("text", [left - 8, Y(t)], {"text": fmt_num(t, span_y), "size": 12,
                                                         "anchor": "end", "baseline": "middle",
                                                         "color": "#475467"}))
        if cat_mode:
            for i, lab in enumerate(self._cat_labels or []):
                prims.append(Prim("text", [X(i), top + ph + 9], {"text": lab, "size": 12,
                                                                 "anchor": "end", "baseline": "top",
                                                                 "color": "#475467",
                                                                 "rotate": self.x_tick_rotate or 30}))
        if cat_y:
            for i, lab in enumerate(self._cat_labels_y or []):
                prims.append(Prim("text", [left - 8, Y(i)], {"text": lab, "size": 12,
                                                             "anchor": "end", "baseline": "middle",
                                                             "color": "#475467"}))

        # 标题与轴名
        prims.append(Prim("text", [left, 26], {"text": self.title, "size": 17, "bold": True,
                                               "anchor": "start", "baseline": "middle"}))
        prims.append(Prim("text", [left + pw / 2, H - 18], {"text": self.xlabel, "size": 13,
                                                            "anchor": "middle", "baseline": "middle",
                                                            "color": "#344054"}))
        prims.append(Prim("text", [22, top + ph / 2], {"text": self.ylabel, "size": 13,
                                                       "anchor": "middle", "baseline": "middle",
                                                       "color": "#344054", "rotate": -90}))
        if self._notes:
            for i, n in enumerate(self._notes[:2]):
                prims.append(Prim("text", [left + pw - 4, top + 14 + i * 16],
                                  {"text": n, "size": 11, "anchor": "end", "baseline": "middle",
                                   "color": "#667085"}))

        # 数据
        legend: list[tuple[str, str, str | None]] = []
        dpts: list[tuple[float, float]] = []   # 数据点屏幕坐标，用于自动摆放图例
        ci = 0
        for it in self.items:
            k = it["kind"]
            if k == "hline":
                prims.append(Prim("line", [(left, Y(it["y"])), (left + pw, Y(it["y"]))],
                                  {"color": it["color"], "width": it["width"], "dash": it["dash"]}))
                if it.get("label"):
                    legend.append((it["label"], it["color"], "line"))
                continue
            if k == "vline":
                prims.append(Prim("line", [(X(it["x"]), top), (X(it["x"]), top + ph)],
                                  {"color": it["color"], "width": it["width"], "dash": it["dash"]}))
                if it.get("label"):
                    legend.append((it["label"], it["color"], "line"))
                continue
            color = it.get("color") or PALETTE[ci % len(PALETTE)]
            if not it.get("color"):
                ci += 1
            if k == "line":
                pts = [(X(a), Y(b)) for a, b in zip(it["x"], it["y"]) if np.isfinite(a) and np.isfinite(b)]
                prims.append(Prim("poly", pts, {"color": color, "width": it["width"], "dash": it["dash"]}))
                if it.get("marker"):
                    for (a, b) in zip(it["x"], it["y"]):
                        if np.isfinite(a) and np.isfinite(b):
                            dpts.append((X(a), Y(b)))
                            prims.extend(_marker_prims(X(a), Y(b), it["marker"], it["markersize"], color))
                if it.get("label"):
                    legend.append((it["label"], color, "line"))
            elif k == "scatter":
                for a, b in zip(it["x"], it["y"]):
                    if np.isfinite(a) and np.isfinite(b):
                        dpts.append((X(a), Y(b)))
                        prims.extend(_marker_prims(X(a), Y(b), it["marker"], it["markersize"], color))
                if it.get("label"):
                    legend.append((it["label"], color, "marker"))
            elif k == "errorbar":
                if it.get("connect"):
                    pts = [(X(a), Y(b)) for a, b in zip(it["x"], it["y"]) if np.isfinite(a) and np.isfinite(b)]
                    prims.append(Prim("poly", pts, {"color": color, "width": 1.6, "dash": it.get("dash")}))
                ye = it["yerr"]
                for i, (a, b) in enumerate(zip(it["x"], it["y"])):
                    if not (np.isfinite(a) and np.isfinite(b)):
                        continue
                    if ye is not None and np.isfinite(ye[i]):
                        e = abs(float(ye[i]))
                        prims.append(Prim("line", [(X(a), Y(b - e)), (X(a), Y(b + e))],
                                          {"color": color, "width": 1.6}))
                        c = it["capsize"]
                        prims.append(Prim("line", [(X(a) - c, Y(b - e)), (X(a) + c, Y(b - e))],
                                          {"color": color, "width": 1.6}))
                        prims.append(Prim("line", [(X(a) - c, Y(b + e)), (X(a) + c, Y(b + e))],
                                          {"color": color, "width": 1.6}))
                    dpts.append((X(a), Y(b)))
                    prims.extend(_marker_prims(X(a), Y(b), it["marker"], it["markersize"], color))
                if it.get("label"):
                    legend.append((it["label"], color, "marker"))
            elif k == "bars":
                vals = it["values"]
                errs = it["errors"]
                n = len(vals)
                for i in range(n):
                    v = vals[i]
                    if not np.isfinite(v):
                        continue
                    col = it["colors"] if isinstance(it["colors"], str) else (
                        (it["colors"][i % len(it["colors"])] if it["colors"] else color))
                    if it["horizontal"]:
                        x1, x0 = X(v), X(0.0)
                        bh2 = min(14.0, max(6.0, ph / max(n, 1) * 0.32))
                        y0, y1 = (Y(i) - bh2), (Y(i) + bh2)
                        dpts.append((x1, Y(i)))
                        prims.append(Prim("rect", [min(x0, x1), y0, abs(x1 - x0), y1 - y0],
                                          {"fill": col, "opacity": 0.85}))
                        if errs is not None and np.isfinite(errs[i]):
                            e = abs(float(errs[i]))
                            prims.append(Prim("line", [(X(v - e), Y(i)), (X(v + e), Y(i))],
                                              {"color": "#333", "width": 1.5}))
                            prims.append(Prim("line", [(X(v - e), Y(i) - 6), (X(v - e), Y(i) + 6)], {"color": "#333", "width": 1.5}))
                            prims.append(Prim("line", [(X(v + e), Y(i) - 6), (X(v + e), Y(i) + 6)], {"color": "#333", "width": 1.5}))
                    else:
                        cx = X(i)
                        y0, y1 = Y(0.0), Y(v)
                        dpts.append((cx, y1))
                        bw = pw / max(n, 1) * 0.6
                        prims.append(Prim("rect", [cx - bw / 2, min(y0, y1), bw, abs(y1 - y0)],
                                          {"fill": col, "opacity": 0.85}))
                        if errs is not None and np.isfinite(errs[i]):
                            e = abs(float(errs[i]))
                            prims.append(Prim("line", [(cx, Y(v - e)), (cx, Y(v + e))], {"color": "#333", "width": 1.5}))
                            prims.append(Prim("line", [(cx - 6, Y(v - e)), (cx + 6, Y(v - e))], {"color": "#333", "width": 1.5}))
                            prims.append(Prim("line", [(cx - 6, Y(v + e)), (cx + 6, Y(v + e))], {"color": "#333", "width": 1.5}))
                if it.get("label"):
                    legend.append((it["label"], color, "bar"))
            elif k == "hist":
                v = it["values"][np.isfinite(it["values"])]
                if v.size == 0:
                    continue
                counts, edges = np.histogram(v, bins=it["bins"], density=it["density"])
                for i in range(len(counts)):
                    xa, xb = edges[i], edges[i + 1]
                    dpts.append(((X(xa) + X(xb)) / 2, Y(counts[i])))
                    prims.append(Prim("rect", [X(xa), Y(counts[i]), X(xb) - X(xa), Y(0.0) - Y(counts[i])],
                                      {"fill": color, "opacity": 0.55, "stroke": "#ffffff", "width": 1}))
                if it.get("normal_curve") and v.size > 2:
                    mu, sd = float(np.mean(v)), float(np.std(v, ddof=1))
                    if sd > 0:
                        lo, hi = float(edges[0]), float(edges[-1])
                        gx = np.linspace(lo, hi, 120)
                        gy = np.exp(-0.5 * ((gx - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
                        if not it["density"]:
                            gy = gy * v.size * (edges[1] - edges[0])
                        prims.append(Prim("poly", [(X(a), Y(b)) for a, b in zip(gx, gy)],
                                          {"color": "#d62728", "width": 2}))
                if it.get("label"):
                    legend.append((it["label"], color, "bar"))
            elif k == "box":
                for i, (name, vals) in enumerate(it["groups"].items()):
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        continue
                    q1, med, q3 = (float(t) for t in np.percentile(vals, [25, 50, 75]))
                    iqr = q3 - q1
                    lo_w = float(vals[vals >= q1 - 1.5 * iqr].min())
                    hi_w = float(vals[vals <= q3 + 1.5 * iqr].max())
                    cx = X(i)
                    bw = pw / max(len(it["groups"]), 1) * 0.45
                    col = (it["colors"][i % len(it["colors"])] if it.get("colors") else color)
                    prims.append(Prim("rect", [cx - bw, Y(q1), 2 * bw, Y(q3) - Y(q1)],
                                      {"fill": col, "opacity": 0.35, "stroke": col, "width": 1.6}))
                    prims.append(Prim("line", [(cx - bw, Y(med)), (cx + bw, Y(med))], {"color": col, "width": 2.4}))
                    prims.append(Prim("line", [(cx, Y(q3)), (cx, Y(hi_w))], {"color": col, "width": 1.4}))
                    prims.append(Prim("line", [(cx, Y(q1)), (cx, Y(lo_w))], {"color": col, "width": 1.4}))
                    prims.append(Prim("line", [(cx - bw * 0.6, Y(hi_w)), (cx + bw * 0.6, Y(hi_w))], {"color": col, "width": 1.4}))
                    prims.append(Prim("line", [(cx - bw * 0.6, Y(lo_w)), (cx + bw * 0.6, Y(lo_w))], {"color": col, "width": 1.4}))
                    for out in vals[(vals < lo_w) | (vals > hi_w)]:
                        prims.extend(_marker_prims(cx, Y(out), "o", 3.0, col))
                if it.get("label"):
                    legend.append((it["label"], color, "bar"))

        # 图例：默认自动挑一个最不遮挡数据的角落
        if self.show_legend and legend:
            lw = 0
            for text, _c, _k in legend:
                lw = max(lw, int(12 + len(str(text)) * 8.5))
            lh = 18 * len(legend)
            bw, bh = lw + 20, lh + 12
            anchors = {
                "upper right": (left + pw - bw - 6, top + 6),
                "upper left": (left + 6, top + 6),
                "lower left": (left + 6, top + ph - bh - 6),
                "lower right": (left + pw - bw - 6, top + ph - bh - 6),
            }

            def _blocked(pos):
                x0, y0 = pos
                return sum(1 for (px, py) in dpts
                           if x0 - 4 <= px <= x0 + bw + 4 and y0 - 4 <= py <= y0 + bh + 4)

            if self.legend_loc in anchors:
                bx1, by1 = anchors[self.legend_loc]
            else:
                bx1, by1 = min(anchors.values(), key=_blocked)
            bx0, by0 = bx1 + bw, by1 + bh
            prims.append(Prim("rect", [bx1, by1, bw, bh],
                              {"fill": "#ffffff", "stroke": "#d0d5dd", "width": 1, "opacity": 0.9, "rx": 4}))
            yy = by1 + 8
            for text, c, kind in legend:
                if kind == "line":
                    prims.append(Prim("line", [(bx1 + 8, yy + 6), (bx1 + 24, yy + 6)], {"color": c, "width": 2}))
                else:
                    prims.extend(_marker_prims(bx1 + 16, yy + 6, "s", 4.0, c))
                prims.append(Prim("text", [bx1 + 31, yy + 6], {"text": text, "size": 12,
                                                               "baseline": "middle", "color": "#344054"}))
                yy += 18
        return prims

    def _build_heatmap(self, it: dict) -> list[Prim]:
        W, H = self.width, self.height
        m = it["matrix"]
        rows, cols = m.shape
        left, top = 140, 70
        right, bottom = 30, 90
        cellw = (W - left - right) / max(cols, 1)
        cellh = (H - top - bottom) / max(rows, 1)
        vals = m[np.isfinite(m)]
        vmin = it["vmin"] if it["vmin"] is not None else (float(vals.min()) if vals.size else 0.0)
        vmax = it["vmax"] if it["vmax"] is not None else (float(vals.max()) if vals.size else 1.0)
        if vmax == vmin:
            vmax = vmin + 1e-12
        prims: list[Prim] = []
        prims.append(Prim("text", [left, 26], {"text": self.title, "size": 17, "bold": True,
                                               "baseline": "middle"}))
        for i in range(rows):
            for j in range(cols):
                v = m[i, j]
                t = (v - vmin) / (vmax - vmin) if np.isfinite(v) else 0.5
                col = _cmap(it["cmap"], t)
                prims.append(Prim("rect", [left + j * cellw, top + i * cellh, cellw - 1, cellh - 1],
                                  {"fill": col if np.isfinite(v) else "#f2f4f7", "stroke": "#ffffff", "width": 1}))
                if it["annotate"] and np.isfinite(v):
                    txt = it["cell_fmt"].format(v)
                    prims.append(Prim("text", [left + j * cellw + cellw / 2, top + i * cellh + cellh / 2],
                                      {"text": txt, "size": 12, "anchor": "middle", "baseline": "middle",
                                       "color": "#1d2939" if abs(t - 0.5) < 0.42 else "#ffffff"}))
            prims.append(Prim("text", [left - 10, top + i * cellh + cellh / 2],
                              {"text": it["row_labels"][i] if i < len(it["row_labels"]) else "",
                               "size": 12, "anchor": "end", "baseline": "middle", "color": "#344054"}))
        for j in range(cols):
            lab = it["col_labels"][j] if j < len(it["col_labels"]) else ""
            prims.append(Prim("text", [left + j * cellw + cellw / 2, top - 12],
                              {"text": lab, "size": 12, "anchor": "middle", "baseline": "bottom",
                               "color": "#344054", "rotate": 0 if len(lab) <= 8 else 30}))
        prims.append(Prim("text", [left + (W - left - right) / 2, H - 30],
                          {"text": self.xlabel or it.get("col_label", ""), "size": 13,
                           "anchor": "middle", "baseline": "middle", "color": "#344054"}))
        prims.append(Prim("text", [24, top + (H - top - bottom) / 2],
                          {"text": self.ylabel or it.get("row_label", ""), "size": 13,
                           "anchor": "middle", "baseline": "middle", "color": "#344054", "rotate": -90}))
        # 色标
        cbx, cby, cbw, cbh = W - right - 18, top, 14, H - top - bottom
        steps = 60
        for s in range(steps):
            t = 1 - s / (steps - 1)
            prims.append(Prim("rect", [cbx, cby + s * cbh / steps, cbw, cbh / steps + 1],
                              {"fill": _cmap(it["cmap"], t)}))
        prims.append(Prim("rect", [cbx, cby, cbw, cbh], {"fill": "none", "stroke": "#98a2b3", "width": 1}))
        for frac, val in ((0.0, vmax), (0.5, (vmax + vmin) / 2), (1.0, vmin)):
            prims.append(Prim("text", [cbx + cbw + 5, cby + frac * cbh],
                              {"text": f"{val:.3g}", "size": 11, "baseline": "middle", "color": "#475467"}))
        return prims


def figure_html(title: str, svg_blocks: Sequence[tuple[str, str]], note: str = "") -> str:
    """把若干 (标题, SVG) 组成一个 HTML 片段/页面。"""
    parts = ['<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">',
             f"<title>{_esc(title)}</title>",
             "<style>body{font-family:'Microsoft YaHei',SimHei,Arial,sans-serif;margin:24px;"
             "background:#f6f7f9;color:#1d2939}h1{font-size:22px}h2{font-size:16px;margin:18px 0 6px}"
             ".card{background:#fff;border:1px solid #e4e7ec;border-radius:10px;padding:12px 16px;"
             "margin:14px 0;box-shadow:0 1px 3px rgba(16,24,40,.06)}"
             ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:14px}"
             "svg{max-width:100%;height:auto}</style></head><body>",
             f"<h1>{_esc(title)}</h1>"]
    if note:
        parts.append(f"<p style='color:#475467'>{_esc(note)}</p>")
    parts.append("<div class='grid'>")
    for t, svg in svg_blocks:
        parts.append(f"<div class='card'><h2>{_esc(t)}</h2>{svg}</div>")
    parts.append("</div></body></html>")
    return "\n".join(parts)
