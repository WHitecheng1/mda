"""最简单的使用方式：一条命令（或双击运行）完成 解析 -> 分析 -> 报告。

用法::

    python 一键分析.py                      # 自动分析当前文件夹里的所有数据文件
    python 一键分析.py 我的数据.csv          # 只分析这一个文件
    python 一键分析.py 数据文件夹 --recursive # 递归分析子文件夹

默认行为：自动识别表头/分组列/自变量列/数值列，自动做统计与不确定度评定、
多组对齐比较、离群判别、线性拟合，输出 HTML/Markdown/JSON/Excel 报告和 PNG 图表。
不需要任何参数，也不需要懂命令行的选项。
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import webbrowser
from datetime import datetime
from typing import Sequence

import pandas as pd

from . import report as R
from .analyze import AnalysisConfig, analyse_dataset
from .parse import json_is_tabular, read_dataset

__all__ = ["main", "find_data_files", "analyze_one"]

DATA_EXT = (".csv", ".tsv", ".txt", ".dat", ".log", ".xlsx", ".xlsm", ".xls", ".json")
OUTDIR_NAME = "分析结果"


def find_data_files(targets: Sequence[str], recursive: bool = False) -> list[str]:
    """找出要分析的数据文件（自动跳过临时文件与我们自己的结果目录）。"""
    found: list[str] = []
    for t in targets:
        if os.path.isfile(t):
            found.append(t)
            continue
        if not os.path.isdir(t):
            continue
        pattern = "**/*" if recursive else "*"
        for p in sorted(glob.glob(os.path.join(t, pattern), recursive=recursive)):
            base = os.path.basename(p)
            if not os.path.isfile(p) or base.startswith("~$") or base.startswith("."):
                continue
            if not p.lower().endswith(DATA_EXT):
                continue
            if OUTDIR_NAME in os.path.normpath(p).split(os.sep):
                continue
            if p.lower().endswith(".json"):          # 跳过配置文件之类的非表格 JSON
                try:
                    import json as _json
                    with open(p, "r", encoding="utf-8-sig", errors="replace") as fh:
                        if not json_is_tabular(_json.load(fh)):
                            continue
                except Exception:
                    continue
            found.append(p)
    # 去重并保持顺序
    seen, out = set(), []
    for p in found:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            out.append(p)
    return out


def _fmt(v, nd: int = 6) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if f != f:
        return "—"
    return f"{f:.{nd}g}"


def analyze_one(path: str, outdir_root: str | None = None, *, sheet=None,
                open_report: bool = False, k: float = 2.0, conf: float = 0.95) -> dict:
    """分析单个文件并输出报告，返回简要结果字典。

    outdir_root 为空时，结果放在"数据文件旁边的 分析结果/ 文件夹"里。
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    if outdir_root is None:
        outdir_root = os.path.join(os.path.dirname(os.path.abspath(path)) or ".", OUTDIR_NAME)
    outdir = os.path.join(outdir_root, stem)
    os.makedirs(outdir, exist_ok=True)

    ds = read_dataset(path, sheet=sheet)
    cfg = AnalysisConfig(outdir=outdir, chart_backend="both", k=k, conf=conf)
    res = analyse_dataset(ds, cfg)

    md = R.write_markdown(res, os.path.join(outdir, "报告.md"))
    html = R.write_html(res, os.path.join(outdir, "报告.html"))
    R.write_json(res, os.path.join(outdir, "结果.json"))
    R.write_excel(res, os.path.join(outdir, "明细.xlsx"))

    print(f"\n[OK] {os.path.basename(path)}")
    print(f"     数据 {len(ds.frame)} 行 | 变量 {len(res.variables)} 个 | 分组 {len(ds.groups)} 组"
          f" | 模式 {ds.schema.mode}")
    if ds.schema.group_col or ds.schema.x_col:
        print(f"     自动识别: 分组列={ds.schema.group_col or '（无，按列分组）'}"
              f"，自变量列={ds.schema.x_col or '（无，按行号）'}")
    n_chart = 0
    for v in res.variables:
        cmp = v.comparison
        ov = cmp.overall if cmp else {}
        name = v.label or v.variable or "测量列"
        line = f"     · {name}: {sum(g.n for g in v.groups)} 个数据点"
        if ov:
            from .uncertainty import format_measurement
            line += (f" | 推荐值 {format_measurement(ov.get('recommended_value'), ov.get('recommended_u'), v.unit)}"
                     f"（k={k:g} 时 U={_fmt((ov.get('recommended_u') or 0) * k, 3)}）"
                     f" | 一致性 χ²/dof={_fmt(ov.get('pooled_chi2_red'), 3)}"
                     f" → {'一致' if ov.get('consistent') else '存在系统偏差'}")
        n_out = sum(len(g.outliers) for g in v.groups)
        if n_out:
            line += f" | 离群值 {n_out} 个（已剔除）"
        print(line)
        for d in v.derived:
            if "error" not in d:
                print(f"       └ 派生量 {d['name']} = {_fmt(d['value'])} ± {_fmt(d['u'])} {d.get('unit','')}")
        n_chart += len(v.chart_files)
    print(f"     图表 {n_chart} 张 | 报告: {os.path.abspath(html)}")
    if open_report:
        try:
            webbrowser.open("file:///" + os.path.abspath(html).replace("\\", "/"))
        except Exception:
            pass
    return {"file": path, "outdir": outdir, "html": html, "md": md, "result": res}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="一键分析",
        description="把测量数据文件丢进来，自动出图表和结论（不需要任何参数）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python 一键分析.py                     # 分析当前文件夹里的全部数据文件\n"
               "  python 一键分析.py 测量记录.csv         # 只分析一个文件\n"
               "  python 一键分析.py D:\\数据 --recursive --open\n")
    p.add_argument("targets", nargs="*", help="数据文件或文件夹（默认：当前文件夹）")
    p.add_argument("--outdir", default=None,
                   help=f"结果输出目录（默认：数据文件旁边的 {OUTDIR_NAME}/ 文件夹）")
    p.add_argument("--recursive", action="store_true", help="递归查找子文件夹")
    p.add_argument("--open", dest="open_report", action="store_true", help="分析完自动用浏览器打开报告")
    p.add_argument("--no-open", dest="open_report", action="store_false", help="不要自动打开（默认）")
    p.set_defaults(open_report=False)
    p.add_argument("--sheet", default=None, help="Excel 工作表名/序号；* 表示合并全部工作表")
    p.add_argument("--k", type=float, default=2.0, help="扩展不确定度包含因子（默认 2）")
    p.add_argument("--conf", type=float, default=0.95, help="置信水平（默认 0.95）")
    p.add_argument("--demo", action="store_true", help="生成示例数据并演示完整流程")
    args = p.parse_args(argv)

    if args.demo:
        from .demo import run_demo
        return run_demo(argparse.Namespace(datadir="sample_data", outdir="output",
                                           charts="both", outlier=None, align=None))

    targets = args.targets or ["."]
    files = find_data_files(targets, args.recursive)
    if not files:
        print("没有找到可分析的数据文件（支持 .csv/.tsv/.txt/.dat/.xlsx/.json）。")
        print("请把数据文件放到本文件夹，或直接把它拖到“一键分析.bat”图标上；")
        print("也可以先看看示例：  python 一键分析.py --demo")
        return 1

    print("=" * 74)
    print(f"一键测量数据分析 —— 共 {len(files)} 个文件"
          + (f"，结果输出到 “{args.outdir}/”" if args.outdir else "，结果放在数据文件旁边的“分析结果/”文件夹"))
    print("=" * 74)
    ok, bad, results, skipped = 0, [], [], []
    for f in files:
        try:
            results.append(analyze_one(f, args.outdir, sheet=args.sheet,
                                       open_report=args.open_report, k=args.k, conf=args.conf))
            ok += 1
        except Exception as exc:  # 单个文件失败不影响其它文件
            msg = str(exc)
            if "没有找到可分析的数值列" in msg or "不是表格数据" in msg:
                skipped.append(os.path.basename(f))   # 不是测量数据表，安静跳过
                continue
            bad.append((f, msg))
            print(f"\n[X] {os.path.basename(f)} 分析失败：{msg}")
    print("\n" + "=" * 74)
    print(f"完成：成功 {ok} 个" + (f"，失败 {len(bad)} 个" if bad else ""))
    if skipped:
        print(f"已跳过 {len(skipped)} 个非数据文件：" + "、".join(skipped[:8])
              + ("…" if len(skipped) > 8 else ""))
    for f, msg in bad:
        print(f"  × {os.path.basename(f)}: {msg}")
    roots = sorted({os.path.abspath(args.outdir) if args.outdir else
                    os.path.join(os.path.dirname(os.path.abspath(r["file"])), OUTDIR_NAME)
                    for r in results})
    print("结果目录：")
    for d in roots:
        print(f"  {d}")
    if ok == 0:
        print("本文件夹里没找到可分析的测量数据。\n"
              "  · 把数据文件（.csv/.xlsx/.txt/.json）放到本文件夹后重新双击；\n"
              "  · 或直接把某个数据文件拖到“一键分析.bat”图标上；\n"
              "  · 想看示例： python 一键分析.py --demo")
    else:
        print("提示：若自动识别不准，可用高级命令指定列，例如\n"
              "  python -m mda run -i 数据.csv --group-col 仪器 --x-col 温度 --value-col 读数")
    return 0 if ok else 1
