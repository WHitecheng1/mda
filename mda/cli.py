"""命令行入口：解析 -> 分析 -> 报告。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

import pandas as pd

from . import report as R
from .analyze import AnalysisConfig, analyse_dataset
from .parse import load_table, read_dataset

__all__ = ["main", "build_parser", "run_files"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mda", description="测量数据处理：解析 / 误差与不确定度 / 多组对齐比较 / 图表与结论")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--config", help="JSON 配置文件（可覆盖其它参数的默认值）")
        sp.add_argument("--outdir", default="output", help="输出目录（默认 output）")
        sp.add_argument("--conf", type=float, default=0.95, help="置信水平，默认 0.95")
        sp.add_argument("--k", type=float, default=2.0, help="扩展不确定度包含因子，默认 2")
        sp.add_argument("--outlier", choices=["grubbs", "mad", "none"], default="grubbs",
                        help="离群值判别方法")
        sp.add_argument("--outlier-alpha", type=float, default=0.05)
        sp.add_argument("--keep-outliers", action="store_true", help="只标记不剔除离群值")
        sp.add_argument("--align", choices=["nearest", "union", "intersection"], default="nearest",
                        help="多组对齐方式")
        sp.add_argument("--align-tol", type=float, default=None, help="x 对齐容差")
        sp.add_argument("--reference", default=None, help="指定参考组名称（默认加权平均）")
        sp.add_argument("--no-fit", action="store_true", help="不做线性拟合")
        sp.add_argument("--unit", default="", help="覆盖单位")
        sp.add_argument("--x-label", default="x", help="横轴名称")
        sp.add_argument("--value-label", default="测量值", help="纵轴名称前缀")
        sp.add_argument("--charts", choices=["both", "png", "svg", "none"], default="both",
                        help="图表输出后端")

    def io(sp):
        sp.add_argument("--input", "-i", required=True, nargs="+", help="输入文件（可多个）")
        sp.add_argument("--sheet", default=None, help="Excel 工作表名或序号；* 表示全部")
        sp.add_argument("--encoding", default=None)
        sp.add_argument("--delimiter", default=None, help="分隔符（默认自动识别）")
        sp.add_argument("--decimal", default=None, help="小数点符号，',' 表示欧式写法")
        sp.add_argument("--header", default="auto", help="表头行号，auto/none/整数")
        sp.add_argument("--value-col", nargs="+", default=None, help="数值列")
        sp.add_argument("--group-col", default=None, help="分组列")
        sp.add_argument("--x-col", default=None, help="自变量列")
        sp.add_argument("--unc-col", default=None, help="不确定度列")
        sp.add_argument("--mode", choices=["wide", "long", "repeat"], default=None)

    sp = sub.add_parser("run", help="完整分析并输出报告")
    io(sp)
    common(sp)
    sp.add_argument("--name", default=None, help="数据集名称")
    sp.add_argument("--merge", action="store_true", help="把多个输入合并为一个数据集比较")

    sp = sub.add_parser("inspect", help="只解析并显示识别到的结构，不分析")
    io(sp)
    sp.add_argument("--rows", type=int, default=12, help="预览行数")

    sp = sub.add_parser("demo", help="生成示例数据并跑通完整流程")
    common(sp)
    sp.add_argument("--datadir", default="sample_data")

    sp = sub.add_parser("selftest", help="运行内置自检测试")
    sp.add_argument("--no-scipy", action="store_true",
                    help="强制关闭 SciPy，用纯 Python 分布实现跑一遍测试")
    return p


def _header_value(v):
    if v in (None, "auto"):
        return "auto"
    if str(v).lower() in ("none", "null", "-1"):
        return None
    return int(v)


def _config_from_args(args) -> AnalysisConfig:
    cfg = AnalysisConfig(
        conf=args.conf, k=args.k, outlier_method=args.outlier,
        outlier_alpha=args.outlier_alpha, exclude_outliers=not args.keep_outliers,
        align_method=args.align, align_tol=args.align_tol, reference=args.reference,
        fit=not args.no_fit, unit=args.unit, x_label=args.x_label,
        value_label=args.value_label, chart_backend=args.charts, outdir=args.outdir,
    )
    if getattr(args, "config", None):
        with open(args.config, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        base = data.get("analysis", data)
        cfg = AnalysisConfig.from_dict({**cfg.__dict__, **base})
        cfg.outdir = args.outdir
        derived = data.get("derived") or base.get("derived")
        if derived:
            cfg.derived = derived
    return cfg


def _read_one(path: str, args, name: str | None = None):
    sheet = args.sheet
    if sheet is not None:
        try:
            sheet = int(sheet)
        except (TypeError, ValueError):
            pass
    return read_dataset(
        path, name=name, sheet=sheet, encoding=args.encoding, delimiter=args.delimiter,
        decimal=args.decimal, comment="#", header=_header_value(args.header),
        value_cols=args.value_col, group_col=args.group_col, x_col=args.x_col,
        unc_col=args.unc_col, mode=args.mode,
    )


def run_files(paths: Sequence[str], args) -> int:
    cfg = _config_from_args(args)
    os.makedirs(cfg.outdir, exist_ok=True)
    datasets = []
    if args.merge and len(paths) > 1:
        frames = []
        for p in paths:
            ds = _read_one(p, args)
            ds.frame = ds.frame.assign(group=ds.name + "|" + ds.frame["group"].astype(str))
            frames.append(ds.frame)
        from .parse import Dataset, Schema
        merged = pd.concat(frames, ignore_index=True)
        ds = Dataset(name=args.name or "合并数据集", frame=merged,
                     schema=Schema(mode="long", value_cols=list(merged["variable"].unique()),
                                   group_col="group", x_col="x"),
                     notes=[f"合并自 {len(paths)} 个文件"])
        datasets = [ds]
    else:
        for p in paths:
            datasets.append(_read_one(p, args, name=args.name if len(paths) == 1 else None))

    rc = 0
    for ds in datasets:
        print(f"\n=== 数据集：{ds.name} ===")
        print(f"数据行数 {len(ds.frame)}，变量 {ds.variables}，组 {ds.groups}")
        for n in ds.notes:
            print(f"  · {n}")
        try:
            result = analyse_dataset(ds, cfg)
        except Exception as exc:
            print(f"[错误] 分析失败：{exc}")
            rc = 1
            continue
        base = os.path.join(cfg.outdir, ds.name)
        md = R.write_markdown(result, base + "_报告.md")
        js = R.write_json(result, base + "_结果.json")
        ht = R.write_html(result, base + "_报告.html")
        xl = R.write_excel(result, base + "_明细.xlsx")
        print(R.summary_dataframe(result).to_string(index=False))
        print(f"已输出：{md} / {ht} / {js} / {xl}")
        for v in result.variables:
            if v.chart_files:
                print(f"  图表 {len(v.chart_files)} 张 -> {os.path.dirname(v.chart_files[0])}")
    return rc


def cmd_inspect(args) -> int:
    for p in args.input:
        raw = load_table(p, sheet=(int(args.sheet) if (args.sheet or "").isdigit() else args.sheet),
                         encoding=args.encoding, delimiter=args.delimiter, decimal=args.decimal,
                         header=_header_value(args.header))
        print(f"\n=== 文件：{p} ===")
        for n in raw.notes:
            print(f"  · {n}")
        print(raw.df.head(args.rows).to_string())
        try:
            ds = _read_one(p, args)
            print(f"\n归一化结果：{ds.describe()}")
            print(f"  分组列={ds.schema.group_col}，x 列={ds.schema.x_col}，"
                  f"数值列={ds.schema.value_cols}，模式={ds.schema.mode}")
            print(ds.frame.head(args.rows).to_string(index=False))
            for v in ds.variables:
                sub = ds.frame[ds.frame["variable"] == v]
                print(f"  变量 {v}: {len(sub)} 点，{sub['group'].nunique()} 组，"
                      f"每组 {sub.groupby('group').size().min()}~{sub.groupby('group').size().max()} 个点")
        except Exception as exc:
            print(f"  [警告] 归一化失败：{exc}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "run":
        return run_files(args.input, args)
    if args.cmd == "inspect":
        return cmd_inspect(args)
    if args.cmd == "demo":
        from .demo import run_demo
        return run_demo(args)
    if args.cmd == "selftest":
        from .tests import run_selftest
        return run_selftest(no_scipy=getattr(args, "no_scipy", False))
    return 2
