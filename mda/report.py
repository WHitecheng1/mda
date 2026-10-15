"""报告输出：Markdown 结论、JSON 机读结果、HTML 图表页、Excel 明细表。"""

from __future__ import annotations

import html
import json
import math
import os
import re
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import uncertainty as U
from .analyze import AnalysisResult, VariableResult

__all__ = ["write_markdown", "write_json", "write_html", "write_excel", "to_jsonable",
           "summary_dataframe"]


def _esc(v: Any) -> str:
    """HTML 转义：数据集名/组名/列名都来自用户数据文件，不能直接拼进 HTML。"""
    return html.escape("" if v is None else str(v), quote=True)


def _md_esc(v: Any) -> str:
    """Markdown 表格单元格转义：竖线会破坏表格结构，换行会断行。"""
    s = "" if v is None else str(v)
    return s.replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", " ")


_BAD_SHEET = re.compile(r"[\[\]:*?/\\]")


def _sheet_name(name: Any, used: set[str]) -> str:
    """把任意名字转成合法且唯一的 Excel 工作表名。

    openpyxl 对非法字符（[]:*?/\\）会报错或改名，之前只截断到 31 字符，
    导致含非法字符的组名那一片表被静默丢弃。
    """
    base = _BAD_SHEET.sub("_", str(name or "sheet")).strip() or "sheet"
    base = base[:31]
    if base not in used:
        used.add(base)
        return base
    for i in range(2, 1000):
        suffix = f"~{i}"
        cand = base[:31 - len(suffix)] + suffix
        if cand not in used:
            used.add(cand)
            return cand
    cand = base[:28] + "~x"
    used.add(cand)
    return cand


def to_jsonable(obj: Any) -> Any:
    """把 numpy / pandas / NaN 转成 JSON 友好的结构。"""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return [to_jsonable(r) for r in obj.reset_index().to_dict("records")]
    if isinstance(obj, pd.Series):
        return to_jsonable(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if not math.isfinite(f) else f
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    return obj


def _fmt(v: Any, nd: int = 6) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if not math.isfinite(v):
            return "—"
        if v != 0 and (abs(v) >= 1e6 or abs(v) < 1e-4):
            return f"{v:.4e}"
        return f"{v:.{nd}g}"
    return str(v)


def _md_table(df: pd.DataFrame, max_rows: int = 40, floatfmt: str = "{:.6g}") -> str:
    """把 DataFrame 转成 Markdown 表格。"""
    if df is None or len(df) == 0:
        return "_（无数据）_\n"
    d = df.head(max_rows).copy()
    cols = [_md_esc(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in d.iterrows():
        cells = []
        for c in d.columns:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append("—" if not np.isfinite(v) else floatfmt.format(float(v)))
            elif isinstance(v, (int, np.integer)):
                cells.append(str(int(v)))
            else:
                cells.append(_md_esc(v))
        lines.append("| " + " | ".join(cells) + " |")
    if len(df) > max_rows:
        lines.append(f"| … | 共 {len(df)} 行，完整数据见同名 Excel 明细 | " + " |" * (len(cols) - 2))
    return "\n".join(lines) + "\n"


def _html_table(df: pd.DataFrame, max_rows: int = 60) -> str:
    if df is None or len(df) == 0:
        return "<p><em>（无数据）</em></p>"
    d = df.head(max_rows)
    head = "".join(f"<th>{_esc(c)}</th>" for c in d.columns)
    rows = []
    for _, r in d.iterrows():
        tds = []
        for c in d.columns:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                tds.append("<td style='text-align:right'>" + ("—" if not np.isfinite(v) else f"{float(v):.6g}") + "</td>")
            elif isinstance(v, (int, np.integer)):
                tds.append(f"<td style='text-align:right'>{int(v)}</td>")
            else:
                tds.append(f"<td>{_esc(v)}</td>")
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return ("<table style='border-collapse:collapse;width:100%;font-size:13px'>"
            "<thead style='background:#f2f4f7'><tr>" + head + "</tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def summary_dataframe(result: AnalysisResult) -> pd.DataFrame:
    """所有变量的关键结论汇总表。"""
    rows = []
    for v in result.variables:
        cmp = v.comparison
        ov = cmp.overall if cmp else {}
        row = {
            "变量": v.label or v.variable,
            "单位": v.unit,
            "组数": len(v.groups),
            "数据点数": int(sum(g.n for g in v.groups)),
            "离群值数": int(sum(len(g.outliers) for g in v.groups)),
        }
        if ov:
            row.update({
                "推荐值": ov.get("recommended_value"),
                "推荐值u": ov.get("recommended_u"),
                "推荐值U(k=2)": (2 * ov["recommended_u"]) if ov.get("recommended_u") else None,
                "RMS偏差": ov.get("rms_dev"),
                "最大偏差": ov.get("max_abs_dev"),
                "chi2/dof": ov.get("pooled_chi2_red"),
                "Birge": ov.get("birge"),
                "一致性": "一致" if ov.get("consistent") else "存在偏差",
            })
        for d in v.derived:
            row[f"派生量:{d.get('name','')}"] = d.get("value")
        rows.append(row)
    return pd.DataFrame(rows)


def write_json(result: AnalysisResult, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = to_jsonable(result.as_dict())
    payload["generated_at"] = datetime.now().isoformat(timespec="seconds")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return path


def write_excel(result: AnalysisResult, path: str) -> str:
    """把所有明细写成多工作表 Excel。工作表名做合法化与去重，避免静默丢表。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    used: set[str] = set()
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary_dataframe(result).to_excel(writer, sheet_name=_sheet_name("汇总", used), index=False)
        for v in result.variables:
            name = _sheet_name(str(v.label or v.variable or "value")[:28], used)
            v.table.to_excel(writer, sheet_name=_sheet_name(f"{name}-统计", used), index=False)
            for g in v.groups:
                if isinstance(g.per_x, pd.DataFrame) and len(g.per_x):
                    g.per_x.to_excel(writer, sheet_name=_sheet_name(f"{name}-{g.group}", used), index=False)
            if v.alignment is not None:
                al = v.alignment
                joined = al.mean.copy()
                joined.columns = [f"{c}_均值" for c in al.mean.columns]
                for c in al.u.columns:
                    joined[f"{c}_u"] = al.u[c]
                for c in al.n.columns:
                    joined[f"{c}_n"] = al.n[c]
                if v.comparison is not None:
                    joined["参考值"] = v.comparison.reference
                    joined["参考值u"] = v.comparison.reference_u
                joined.to_excel(writer, sheet_name=_sheet_name(f"{name}-对齐", used))
            if v.comparison is not None:
                if len(v.comparison.pairwise):
                    v.comparison.pairwise.to_excel(writer, sheet_name=_sheet_name(f"{name}-两两", used), index=False)
                if len(v.comparison.group_stats):
                    v.comparison.group_stats.to_excel(writer, sheet_name=_sheet_name(f"{name}-分组偏差", used))
            if v.fits:
                fit_rows = [dict(名称=k, **v.as_dict()["fits"][k]) for k in v.fits]
                pd.DataFrame(fit_rows).to_excel(writer, sheet_name=_sheet_name(f"{name}-拟合", used), index=False)
            if v.derived:
                pd.DataFrame(v.derived).to_excel(writer, sheet_name=_sheet_name(f"{name}-派生量", used), index=False)
    return path


def write_markdown(result: AnalysisResult, path: str, *, chart_rel: str = "charts") -> str:
    cfg = result.config
    L: list[str] = []
    L.append(f"# 测量数据分析报告：{result.dataset}")
    L.append("")
    L.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"- 置信水平：{cfg.conf:.0%}；扩展不确定度包含因子 k = {cfg.k:g}")
    L.append(f"- 离群值判别：{cfg.outlier_method}（alpha={cfg.outlier_alpha}，"
             f"{'已剔除' if cfg.exclude_outliers else '仅标记'}）")
    L.append(f"- 对齐方式：{cfg.align_method}；参考值：{cfg.reference or '反方差加权平均'}")
    L.append("")
    if result.notes:
        L.append("## 解析说明")
        L.append("")
        for n in result.notes:
            L.append(f"- {n}")
        L.append("")
    L.append("## 一、总体结论")
    L.append("")
    L.append(_md_table(summary_dataframe(result)))
    L.append("")
    for v in result.variables:
        L.append(f"## 二、变量：{v.label or v.variable}" + (f"（单位 {v.unit}）" if v.unit else ""))
        L.append("")
        L.append("### 1. 各组统计与 A 类不确定度")
        L.append("")
        L.append(_md_table(v.table))
        L.append("")
        L.append("### 2. 关键结论")
        L.append("")
        for c in v.conclusions:
            L.append(f"- {c}")
        L.append("")
        if v.alignment is not None and v.comparison is not None:
            al, cmp = v.alignment, v.comparison
            L.append(f"### 3. 对齐结果（{al.method}，{len(al.mean)} 个网格点）")
            L.append("")
            joined = al.mean.copy()
            joined.columns = [f"{c}_均值" for c in al.mean.columns]
            for c in al.u.columns:
                joined[f"{c}_u"] = al.u[c]
            joined["参考值"] = cmp.reference
            L.append(_md_table(joined.reset_index(), max_rows=30))
            L.append("")
            L.append("### 4. 分组偏差统计")
            L.append("")
            L.append(_md_table(cmp.group_stats.reset_index(), max_rows=30))
            L.append("")
            if len(cmp.pairwise):
                L.append("### 5. 两两比较")
                L.append("")
                L.append(_md_table(cmp.pairwise, max_rows=30))
                L.append("")
        if v.fits:
            L.append("### 6. 线性拟合")
            L.append("")
            rows = []
            for k, f in v.fits.items():
                rows.append({"名称": k, "斜率": f.slope, "斜率u": f.u_slope,
                             "截距": f.intercept, "截距u": f.u_intercept, "R2": f.r2,
                             "残差std": f.s_resid, "chi2/dof": f.chi2_red, "n": f.n})
            L.append(_md_table(pd.DataFrame(rows), max_rows=30))
            L.append("")
        if v.derived:
            L.append("### 7. 派生量（间接测量）与不确定度预算")
            L.append("")
            rows = []
            for d in v.derived:
                if "error" in d:
                    rows.append({"名称": d.get("name"), "错误": d["error"]})
                    continue
                rows.append({"名称": d["name"], "公式": d["formula"], "结果": d["value"],
                             "标准不确定度u": d["u"], "扩展不确定度U": d["U"],
                             "k": d["k"], "单位": d.get("unit", ""),
                             "相对不确定度%": d["relative_pct"]})
            L.append(_md_table(pd.DataFrame(rows), max_rows=20))
            L.append("")
            for d in v.derived:
                contrib = d.get("contributions") if isinstance(d, dict) else None
                if contrib:
                    cdf = pd.DataFrame({"来源": list(contrib.keys()),
                                        "标准不确定度贡献": list(contrib.values())})
                    cdf["占比%"] = 100 * (cdf["标准不确定度贡献"] / d["u"]) ** 2
                    cdf = cdf.sort_values("标准不确定度贡献", ascending=False)
                    L.append(f"**{d['name']}** 不确定度预算：")
                    L.append("")
                    L.append(_md_table(cdf, max_rows=20))
                    L.append("")
        if v.chart_files:
            L.append("### 8. 图表")
            L.append("")
            names = {
                "_01_": "各组均值与不确定度", "_02_": "与参考值的偏差", "_03_": "偏差矩阵",
                "_04_": "Bland-Altman 一致性", "_05_": "分布箱线图", "_06_": "直方图",
                "_07_": "线性拟合", "_08_": "不确定度构成",
            }
            for f in v.chart_files:
                base = os.path.basename(f)
                title = next((t for k, t in names.items() if k in base), base)
                rel = os.path.relpath(f, os.path.dirname(os.path.abspath(path))).replace("\\", "/")
                L.append(f"- **{title}**：![{title}]({rel})")
            L.append("")
    L.append("---")
    L.append("")
    L.append("_本报告由 mda（测量数据处理与不确定度分析工具包）自动生成；"
             "所有不确定度按 GUM 方法评定，A 类取 t 分布区间，合成采用方和根。_")
    text = "\n".join(L)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def write_html(result: AnalysisResult, path: str, *, title: str | None = None) -> str:
    """生成自包含的 HTML 报告（SVG 内嵌，可直接发给同事看）。"""
    from . import charts as C
    ttl = title or f"测量数据分析报告 — {result.dataset}"
    cfg = result.config
    head = [
        "<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>",
        f"<title>{_esc(ttl)}</title>",
        "<style>",
        "body{font-family:'Microsoft YaHei',SimHei,Arial,sans-serif;margin:0;background:#f6f7f9;color:#1d2939}",
        ".wrap{max-width:1180px;margin:0 auto;padding:24px}",
        "h1{font-size:24px;margin:0 0 8px}h2{font-size:18px;margin:26px 0 10px;border-left:4px solid #1f77b4;padding-left:8px}",
        "h3{font-size:15px;margin:18px 0 8px;color:#344054}",
        ".card{background:#fff;border:1px solid #e4e7ec;border-radius:10px;padding:16px 18px;margin:14px 0;box-shadow:0 1px 3px rgba(16,24,40,.06)}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(470px,1fr));gap:14px}",
        "svg{max-width:100%;height:auto}",
        "table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #e4e7ec;padding:5px 8px}",
        "th{background:#f2f4f7;text-align:left}",
        "ul{line-height:1.65}code{background:#f2f4f7;padding:1px 4px;border-radius:4px}",
        ".meta{color:#475467;font-size:13px}",
        "</style></head><body><div class='wrap'>",
        f"<h1>{_esc(ttl)}</h1>",
        f"<p class='meta'>生成时间 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ｜ "
        f"置信水平 {cfg.conf:.0%} ｜ k={cfg.k:g} ｜ 离群判别 {_esc(cfg.outlier_method)}"
        f"（{'剔除' if cfg.exclude_outliers else '标记'}） ｜ 对齐 {_esc(cfg.align_method)} ｜ "
        f"参考值 {_esc(cfg.reference) if cfg.reference else '反方差加权平均'}</p>",
    ]
    if result.notes:
        head.append("<div class='card'><b>解析说明</b><ul>" +
                    "".join(f"<li>{_esc(n)}</li>" for n in result.notes) + "</ul></div>")
    head.append("<h2>总体汇总</h2><div class='card'>" + _html_table(summary_dataframe(result)) + "</div>")
    for v in result.variables:
        head.append(f"<h2>变量：{_esc(v.label or v.variable)}" + (f"（{_esc(v.unit)}）" if v.unit else "") + "</h2>")
        head.append("<div class='card'><h3>各组统计</h3>" + _html_table(v.table) + "</div>")
        head.append("<div class='card'><h3>结论</h3><ul>" +
                    "".join(f"<li>{_esc(c)}</li>" for c in v.conclusions) + "</ul></div>")
        if v.comparison is not None:
            head.append("<div class='card'><h3>分组偏差</h3>" +
                        _html_table(v.comparison.group_stats.reset_index()) + "</div>")
            if len(v.comparison.pairwise):
                head.append("<div class='card'><h3>两两比较</h3>" +
                            _html_table(v.comparison.pairwise) + "</div>")
        if v.derived:
            head.append("<div class='card'><h3>派生量</h3>" + _html_table(
                pd.DataFrame([{k: d.get(k) for k in ("name", "formula", "value", "u", "U", "unit",
                                                     "relative_pct")} for d in v.derived])) + "</div>")
        if v.charts:
            head.append("<h3>图表</h3><div class='grid'>")
            for t, svg in v.charts:
                head.append(f"<div class='card'><b>{_esc(t)}</b>{svg}</div>")
            head.append("</div>")
    head.append("<p class='meta'>由 mda 工具包自动生成：解析 → 不确定度评定 → 对齐比较 → 图表与结论。</p>")
    head.append("</div></body></html>")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(head))
    return path
