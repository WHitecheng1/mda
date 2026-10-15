"""示例数据生成与一键演示。

模拟一次真实的长度计量比对：
4 台仪器在 5 个温度点上各重复测量 6 次，另含宽表、杂记文本与 Excel 两种来源，
最后输出完整报告（Markdown / HTML / JSON / Excel + PNG/SVG 图表）。
"""

from __future__ import annotations

import json
import os
from typing import Sequence

import numpy as np
import pandas as pd

from .cli import main as cli_main

__all__ = ["generate_sample_data", "run_demo"]

# 真值模型：L(T) = L0 * (1 + alpha * (T - 20))，L0 = 1000 mm，alpha = 23.5e-6 /°C
L0_TRUE = 1000.0
ALPHA_TRUE = 23.5e-6
TEMPS = [20.0, 25.0, 30.0, 35.0, 40.0]
INSTRUMENTS = {
    # 名称: (系统偏差 mm, 噪声标准差 mm, 斜率比例, 离群点数)
    "仪器A": (0.000, 0.0014, 1.00, 0),
    "仪器B": (0.008, 0.0030, 1.00, 0),
    "仪器C": (-0.004, 0.0011, 1.00, 0),
    "仪器D": (0.002, 0.0034, 1.06, 1),
}
REPEATS = 6


def _length(t: float, slope_scale: float) -> float:
    return L0_TRUE * (1.0 + ALPHA_TRUE * slope_scale * (t - 20.0))


def generate_sample_data(datadir: str = "sample_data", seed: int = 20261015) -> dict:
    """生成四份示例原始数据和一个分析配置，返回文件路径字典。"""
    rng = np.random.default_rng(seed)
    os.makedirs(datadir, exist_ok=True)
    paths: dict[str, str] = {}

    # ---------- 1) 长表：仪器 / 温度 / 重复测量 ----------
    rows = []
    for inst, (bias, noise, sscale, n_out) in INSTRUMENTS.items():
        base = _length(0, sscale)
        for t in TEMPS:
            true = _length(t, sscale)
            vals = rng.normal(true + bias, noise, REPEATS)
            out_idx = set()
            if n_out:
                out_idx = set(rng.choice(REPEATS, size=n_out, replace=False).tolist())
            for k in range(REPEATS):
                v = vals[k]
                if k in out_idx:
                    v = v + rng.choice([-1, 1]) * (0.02 + 0.01 * rng.random())
                txt = f"{v:.4f}"
                if k == 1 and inst in ("仪器A", "仪器C"):
                    txt = f"{v:.4f} ± 0.0020"      # 现场记录了单次测量的不确定度
                rows.append({"仪器": inst, "温度点℃": t, "长度测量值mm": txt,
                             "备注": "复核" if k in out_idx else ""})
    long_path = os.path.join(datadir, "长度测量_长表.csv")
    pd.DataFrame(rows).to_csv(long_path, index=False, encoding="utf-8-sig")
    paths["long_csv"] = long_path

    # ---------- 2) 宽表：每个仪器一列，带 ± 记法 ----------
    wide = {"温度点℃": TEMPS}
    for inst, (bias, noise, sscale, _n) in INSTRUMENTS.items():
        col = []
        for t in TEMPS:
            m = _length(t, sscale) + bias
            s = noise * 3.0          # 该列给的是几次测量的均值，故不确定度更小
            col.append(f"{m:.4f} ± {s:.4f}")
        # 故意留一个缺失值，演示缺失处理
        col[2] = "NA"
        wide[inst] = col
    wide_path = os.path.join(datadir, "宽度测量_宽表.csv")
    pd.DataFrame(wide).to_csv(wide_path, index=False, encoding="utf-8-sig")
    paths["wide_csv"] = wide_path

    # ---------- 3) 杂记文本：空白分隔 + 注释 + 缺失 + 括号记法 ----------
    lines = ["# 2026-10-12 现场原始记录（手工整理）",
             "# 仪器：千分尺 No.3 / 单位：mm",
             "# 空缺以 -- 或 NA 表示",
             "样品编号    温度点    读数      备注"]
    rng2 = np.random.default_rng(seed + 1)
    for sample, off in (("S-01", 0.0), ("S-02", 0.006), ("S-03", -0.003)):
        for t in (20.0, 30.0, 40.0):
            for k in range(3):
                v = L0_TRUE * (1 + ALPHA_TRUE * (t - 20)) * 0.05 + off + rng2.normal(0, 0.0008)
                if sample == "S-02" and t == 30.0 and k == 2:
                    txt = "--"
                elif k == 2:
                    txt = f"{v:.4f}(3)"            # 括号记法：末位 3 个单位
                else:
                    txt = f"{v:.4f}"
                note = "重测" if note_ok(rng2) else ""
                lines.append(f"{sample:<10} {t:>8.1f} {txt:>12} {note}")
        lines.append("")
    messy_path = os.path.join(datadir, "原始记录_杂记.txt")
    with open(messy_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    paths["messy_txt"] = messy_path

    # ---------- 4) Excel：两个工作表（首次标定 + 复测核对） ----------
    xl_path = os.path.join(datadir, "读数记录.xlsx")
    nominals = [500.0, 750.0, 1000.0, 1250.0]
    insts = ["M1", "M2", "M3"]
    recs1, recs2 = [], []
    for i, inst in enumerate(insts):
        b = (i - 1) * 0.005
        for nom in nominals:
            recs1.append({"仪器": inst, "标称长度mm": nom,
                          "示值mm": round(nom * (1 + ALPHA_TRUE * 10) + b + rng.normal(0, 0.002), 4),
                          "示值不确定度mm": 0.0025})
            recs2.append({"仪器": inst, "标称长度mm": nom,
                          "示值mm": round(nom * (1 + ALPHA_TRUE * 10) + b * 1.2 + rng.normal(0, 0.002), 4),
                          "示值不确定度mm": 0.0030})
    with pd.ExcelWriter(xl_path, engine="openpyxl") as w:
        pd.DataFrame(recs1).to_excel(w, sheet_name="首次标定", index=False)
        pd.DataFrame(recs2).to_excel(w, sheet_name="复测核对", index=False)
    paths["xlsx"] = xl_path

    # ---------- 5) 分析配置（派生量：线膨胀系数） ----------
    cfg = {
        "analysis": {
            "conf": 0.95, "k": 2.0,
            "outlier_method": "grubbs", "outlier_alpha": 0.05, "exclude_outliers": True,
            "align_method": "nearest", "fit": True,
        },
        "derived": [
            {
                "name": "线膨胀系数 alpha",
                "label": "alpha",
                "formula": "slope / L0",
                "unit": "1/℃",
                "fit_source": "reference",
                "vars": {"L0": {"value": 1000.0, "u": 0.05, "dof": 50}},
            }
        ],
    }
    cfg_path = os.path.join(datadir, "分析配置.json")
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=2)
    paths["config"] = cfg_path
    return paths


def note_ok(rng) -> bool:
    return bool(rng.random() < 0.15)


def run_demo(args=None) -> int:
    """生成示例数据并依次跑通四种来源的完整分析。"""
    datadir = getattr(args, "datadir", None) or "sample_data"
    outdir = getattr(args, "outdir", None) or "output"
    extra: list[str] = []
    if getattr(args, "charts", None):
        extra += ["--charts", args.charts]
    if getattr(args, "outlier", None):
        extra += ["--outlier", args.outlier]
    if getattr(args, "align", None):
        extra += ["--align", args.align]

    paths = generate_sample_data(datadir)
    print("示例数据已生成：")
    for k, v in paths.items():
        print(f"  · {k}: {v}")

    jobs = [
        ("长表（4 台仪器 × 5 个温度点 × 6 次重复，含派生量）",
         ["-i", paths["long_csv"], "--config", paths["config"],
          "--x-label", "温度 / ℃", "--value-label", "长度"]),
        ("宽表（每列一台仪器，值带 ± 记法，含缺失值）",
         ["-i", paths["wide_csv"], "--x-label", "温度 / ℃", "--value-label", "宽度"]),
        ("杂记文本（空白分隔 + 注释 + 缺失 + 括号记法）",
         ["-i", paths["messy_txt"], "--x-label", "温度 / ℃", "--value-label", "读数"]),
        ("Excel 多工作表（首次标定 + 复测核对）",
         ["-i", paths["xlsx"], "--sheet", "*", "--x-label", "标称长度 / mm",
          "--value-label", "示值"]),
    ]
    rc = 0
    for title, argv in jobs:
        print("\n" + "=" * 78)
        print("【演示】" + title)
        print("=" * 78)
        code = cli_main(["run", *argv, "--outdir", outdir, *extra])
        rc = rc or code
    print("\n全部演示完成，输出目录：" + os.path.abspath(outdir))
    for f in sorted(os.listdir(outdir)) if os.path.isdir(outdir) else []:
        print("  · " + f)
    chart_dir = os.path.join(outdir, "charts")
    if os.path.isdir(chart_dir):
        n = len([f for f in os.listdir(chart_dir) if f.endswith(".png")])
        print(f"  · charts/ 共 {n} 张 PNG 图")
    return rc
