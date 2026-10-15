"""分析流程编排：数据集 -> 统计与不确定度 -> 对齐比较 -> 图表 -> 结论。"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import align as A
from . import charts as C
from . import dist as D
from . import uncertainty as U
from .parse import Dataset

__all__ = ["AnalysisConfig", "GroupResult", "VariableResult", "AnalysisResult",
           "analyse_variable", "analyse_dataset", "eval_formula"]

SAFE_FUNCS = {k: getattr(math, k) for k in
              ("sqrt", "exp", "log", "log10", "sin", "cos", "tan", "atan", "atan2",
               "pi", "e", "pow", "fabs", "hypot")}
SAFE_FUNCS.update({"abs": abs, "min": min, "max": max, "round": round, "sum": sum})


def eval_formula(formula: str, variables: Mapping[str, float]) -> float:
    """安全地求值一个数学表达式（仅允许 math 函数与给定变量）。"""
    env = dict(SAFE_FUNCS)
    env.update({k: float(v) for k, v in variables.items()})
    return float(eval(formula, {"__builtins__": {}}, env))  # noqa: S307 - 白名单环境


@dataclass
class AnalysisConfig:
    conf: float = 0.95                 # 置信水平
    k: float = 2.0                     # 扩展不确定度包含因子
    outlier_method: str = "grubbs"     # grubbs | mad | none
    outlier_alpha: float = 0.05
    exclude_outliers: bool = True
    align_method: str = "nearest"      # nearest | union | intersection | grid
    align_tol: float | None = None
    reference: str | None = None       # 指定参考组；None 表示加权平均参考值
    fit: bool = True
    unit: str = ""
    x_label: str = "x"
    value_label: str = "测量值"
    derived: list[dict] = field(default_factory=list)
    chart_backend: str = "both"        # both | png | svg | none
    outdir: str | None = None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AnalysisConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in dict(d).items() if k in known})


@dataclass
class GroupResult:
    group: str
    n: int
    summary: dict
    type_a: dict
    outliers: list[dict]
    per_x: pd.DataFrame = field(default_factory=pd.DataFrame)
    used_n: int = 0

    def as_dict(self) -> dict:
        return {"group": self.group, "n": self.n, "used_n": self.used_n,
                "summary": self.summary, "type_a": self.type_a, "outliers": self.outliers}


@dataclass
class VariableResult:
    variable: str
    unit: str
    y_label: str
    groups: list[GroupResult]
    alignment: A.Alignment | None
    comparison: A.Comparison | None
    fits: dict
    derived: list[dict]
    conclusions: list[str]
    charts: list[tuple[str, str]] = field(default_factory=list)   # (标题, 文件或 svg)
    chart_files: list[str] = field(default_factory=list)
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    label: str = ""

    def as_dict(self) -> dict:
        return {
            "variable": self.variable,
            "label": self.label,
            "unit": self.unit,
            "groups": [g.as_dict() for g in self.groups],
            "comparison": self.comparison.as_dict() if self.comparison else None,
            "fits": {k: v.as_dict() for k, v in self.fits.items()},
            "derived": self.derived,
            "conclusions": self.conclusions,
            "chart_files": self.chart_files,
        }


@dataclass
class AnalysisResult:
    dataset: str
    variables: list[VariableResult]
    config: AnalysisConfig
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"dataset": self.dataset, "notes": self.notes,
                "config": {k: v for k, v in self.config.__dict__.items()},
                "variables": [v.as_dict() for v in self.variables]}


def _group_stats(df: pd.DataFrame, cfg: AnalysisConfig) -> tuple[list[GroupResult], pd.DataFrame]:
    """逐组统计 + 离群值判别，返回 (结果, 剔除离群后的数据)。"""
    results: list[GroupResult] = []
    kept_frames = []
    for g, sub in df.groupby("group", sort=False):
        sub = sub.reset_index(drop=True)
        # 有重复测量时，离群判别应在同一 (组, x) 的重复序列内部进行
        cell_sizes = sub.groupby("x").size()
        per_cell = (cfg.outlier_method in ("grubbs", "mad") and sub["x"].nunique(dropna=True) > 1
                    and len(cell_sizes) > 0 and int(cell_sizes.min()) >= 4)
        outl: list[dict] = []
        if per_cell:
            for xv, sx in sub.groupby("x", sort=True):
                vals_c = sx["value"].to_numpy(dtype="float64")
                found = (U.outliers_grubbs(vals_c, cfg.outlier_alpha)
                         if cfg.outlier_method == "grubbs" else U.outliers_mad(vals_c))
                for o in found:
                    o = dict(o)
                    o["index"] = int(sx.index[o["index"]])
                    o["x"] = xv
                    outl.append(o)
        elif cfg.outlier_method == "grubbs":
            outl = U.outliers_grubbs(sub["value"].to_numpy(dtype="float64"), cfg.outlier_alpha)
        elif cfg.outlier_method == "mad":
            outl = U.outliers_mad(sub["value"].to_numpy(dtype="float64"))
        drop = {o["index"] for o in outl} if cfg.exclude_outliers else set()
        keep_mask = np.array([i not in drop for i in range(len(sub))])
        kept = sub[keep_mask]
        kept_frames.append(kept)
        ta = U.type_a(kept["value"].to_numpy(dtype="float64"), cfg.conf)
        per_x = pd.DataFrame()
        if sub["x"].nunique() > 1 and len(sub) > sub["x"].nunique():
            rows = []
            for xv, sx in kept.groupby("x", sort=True):
                if len(sx) >= 2:
                    s = U.stats_summary(sx["value"])
                    rows.append({"x": xv, "n": s.n, "mean": s.mean, "std": s.std,
                                 "u_A": s.sem, "rsd": s.rsd})
                else:
                    rows.append({"x": xv, "n": len(sx), "mean": float(sx["value"].mean()),
                                 "std": np.nan, "u_A": np.nan, "rsd": np.nan})
            per_x = pd.DataFrame(rows)
        results.append(GroupResult(group=str(g), n=len(sub), summary=ta.summary.as_dict(),
                                   type_a=ta.as_dict(), outliers=outl, per_x=per_x,
                                   used_n=len(kept)))
    kept_all = pd.concat(kept_frames, ignore_index=True) if kept_frames else df.iloc[0:0]
    return results, kept_all


def _fit_table(al: A.Alignment, cfg: AnalysisConfig) -> tuple[dict, dict[str, str]]:
    """对每组做线性拟合；返回 (拟合结果, {组名: 失败原因})——失败不再静默丢弃。"""
    fits: dict[str, Any] = {}
    errors: dict[str, str] = {}
    x = np.asarray(al.x, dtype="float64")
    if x.size < 3:
        return fits, errors
    for g in al.groups:
        y = al.mean[g].to_numpy(dtype="float64")
        uy = al.u[g].to_numpy(dtype="float64")
        if np.isfinite(y).sum() >= 3:
            try:
                fits[g] = U.linear_fit(x, y, uy if np.isfinite(uy).any() else None)
            except Exception as exc:
                errors[str(g)] = f"{type(exc).__name__}: {exc}"
    return fits, errors


def _derived_quantities(fits: dict, reference_fit: U.LinearFit | None,
                        cfg: AnalysisConfig, unit: str) -> list[dict]:
    """按配置计算派生量及其不确定度传播。"""
    out = []
    for spec in cfg.derived or []:
        formula = spec.get("formula")
        if not formula:
            continue
        src = spec.get("fit_source", "reference")
        fit = reference_fit if src in ("reference", "加权平均参考值") else fits.get(src)
        if fit is None:
            continue
        variables = {"slope": (fit.slope, fit.u_slope), "intercept": (fit.intercept, fit.u_intercept)}
        dofs = {"slope": float(fit.dof), "intercept": float(fit.dof)}
        for name, v in (spec.get("vars") or {}).items():
            if isinstance(v, dict):
                variables[name] = (float(v.get("value", 0.0)), float(v.get("u", 0.0)))
                if v.get("dof"):
                    dofs[name] = float(v["dof"])
            else:
                variables[name] = (float(v), 0.0)
        try:
            prop = U.propagate(lambda *a, _f=formula, _n=list(variables): eval_formula(_f, dict(zip(_n, a))),
                               variables, dofs=dofs)
        except Exception as exc:  # pragma: no cover
            out.append({"name": spec.get("name", formula), "error": str(exc)})
            continue
        k = D.t_crit(cfg.conf, prop.dof_eff) if np.isfinite(prop.dof_eff) else cfg.k
        out.append({
            "name": spec.get("name", formula),
            "formula": formula,
            "value": prop.value,
            "u": prop.uc,
            "U": k * prop.uc,
            "k": k,
            "unit": spec.get("unit", unit),
            "fit_source": src,
            "contributions": {k2: v2 for k2, v2 in prop.contributions.items()},
            "relative_pct": 100.0 * prop.uc / abs(prop.value) if prop.value else float("nan"),
            "text": (spec.get("label", spec.get("name", formula)) + ": " +
                     U.format_measurement(prop.value, k * prop.uc, spec.get("unit", unit)) +
                     f" (k={k:.3g})"),
        })
    return out


def analyse_variable(long: pd.DataFrame, variable: str, cfg: AnalysisConfig, *,
                     prefix: str = "", backend: str = "both") -> VariableResult:
    """分析单个变量（一列测量值）。"""
    df = long[long["variable"] == variable].copy()
    if df.empty:
        raise ValueError(f"变量 {variable} 没有数据")
    unit = ""
    if "unit" in df and df["unit"].astype(str).str.len().gt(0).any():
        unit = str(df["unit"].mode().iloc[0])
    unit = cfg.unit or unit
    display = variable or cfg.value_label or "测量值"
    slug = variable or "value"
    # 有列名时以列名为准（--value-label 只用于没有列名的宽表场景）
    y_label = display + (f" / {unit}" if unit else "")

    groups, kept = _group_stats(df, cfg)

    x = kept["x"]
    n_x = x.nunique(dropna=True)
    alignment = None
    comparison = None
    fits: dict = {}
    derived: list[dict] = []
    charts: list[tuple[str, str]] = []
    chart_files: list[str] = []

    if n_x >= 1:
        try:
            alignment = A.align_groups(kept, method=cfg.align_method, tol=cfg.align_tol,
                                       x_label=cfg.x_label)
            comparison = A.compare_alignment(alignment, reference=cfg.reference, conf=cfg.conf)
        except Exception as exc:
            alignment = None
            comparison = None
            # 不静默：记进结果（报告会显示），同时给出 warning
            if groups:
                groups[0].summary["align_error"] = str(exc)
            warnings.warn(f"对齐/比较失败，已跳过该变量（{variable}）：{exc}",
                          RuntimeWarning, stacklevel=2)

    if cfg.fit and alignment is not None and len(alignment) >= 3:
        fits, fit_errors = _fit_table(alignment, cfg)
        for gname, msg in fit_errors.items():
            warnings.warn(f"线性拟合失败，已跳过组 {gname}（{variable}）：{msg}",
                          RuntimeWarning, stacklevel=2)
        if fit_errors and groups:
            groups[0].summary["fit_errors"] = "; ".join(
                f"{k}: {v}" for k, v in fit_errors.items())
        ref_fit = None
        if len(alignment) >= 3:
            try:
                ref_fit = U.linear_fit(np.asarray(alignment.x, dtype="float64"),
                                       comparison.reference.to_numpy(dtype="float64")
                                       if comparison else alignment.mean.mean(axis=1).to_numpy(),
                                       comparison.reference_u.to_numpy(dtype="float64")
                                       if comparison and comparison.reference_u.notna().any() else None)
            except Exception as exc:
                ref_fit = None
                warnings.warn(f"参考值线性拟合失败（{variable}）：{exc}",
                              RuntimeWarning, stacklevel=2)
        derived = _derived_quantities(fits, ref_fit, cfg, unit)
        if ref_fit is not None:
            fits["__参考值__"] = ref_fit

    # ---------------- 图表 ----------------
    def emit(title: str, plot: C.Plot, slug: str):
        if backend in ("svg", "both"):
            charts.append((title, plot.svg()))       # 直接渲染 SVG，不落临时文件
        if backend in ("png", "both") and cfg.outdir:
            cdir = os.path.join(cfg.outdir, "charts")
            os.makedirs(cdir, exist_ok=True)
            path = os.path.join(cdir, f"{prefix}{slug}.png")
            plot.render(path, fmt="png")
            chart_files.append(path)

    if alignment is not None:
        # 1) 各组均值趋势 + 误差棒
        p = C.Plot(title=f"{display} — 各组均值随 {cfg.x_label} 的变化",
                   xlabel=cfg.x_label, ylabel=y_label)
        for gi, g in enumerate(alignment.groups):
            p.errorbar(alignment.x, alignment.mean[g].to_numpy(dtype="float64"),
                       alignment.u[g].to_numpy(dtype="float64"), label=g,
                       color=C.PALETTE[gi % len(C.PALETTE)], marker="o", capsize=4)
        if comparison is not None:
            p.errorbar(alignment.x, comparison.reference.to_numpy(dtype="float64"),
                       comparison.reference_u.to_numpy(dtype="float64"),
                       label=comparison.reference_name, color="#111111", marker="D",
                       capsize=5, connect=False)
        emit(f"{display}: 各组均值与不确定度", p, f"{slug}_01_trend")

        # 2) 偏差图
        if comparison is not None:
            p = C.Plot(title=f"{display} — 相对参考值的偏差", xlabel=cfg.x_label,
                       ylabel=f"偏差 / {unit}" if unit else "偏差")
            p.hline(0.0, label="零偏差", color="#111111", dash=(6, 4))
            for gi, g in enumerate(alignment.groups):
                p.errorbar(alignment.x, comparison.deviations[g].to_numpy(dtype="float64"),
                           np.zeros(len(alignment.x)), label=g,
                           color=C.PALETTE[gi % len(C.PALETTE)], marker="o", capsize=0,
                           connect=True)
            emit(f"{display}: 与参考值的偏差", p, f"{slug}_02_deviation")

            # 3) 相对偏差热力图（两两平均相对偏差）
            labels, mat = [], []
            for gi, a in enumerate(alignment.groups):
                row = []
                labels.append(a)
                for gj, b in enumerate(alignment.groups):
                    da = alignment.mean[a].to_numpy(dtype="float64")
                    db = alignment.mean[b].to_numpy(dtype="float64")
                    m = np.isfinite(da) & np.isfinite(db) & (db != 0)
                    row.append(float(np.mean((da[m] - db[m]) / db[m]) * 100) if m.any() else np.nan)
                mat.append(row)
            p = C.Plot(title=f"{display} — 两组间平均相对偏差矩阵 (%)", width=760, height=520,
                       xlabel="作为基准的组", ylabel="被比较的组")
            p.heatmap(np.asarray(mat), labels, labels, cell_fmt="{:+.3f}",
                      cmap="RdBu", annotate=True)
            emit(f"{display}: 偏差矩阵", p, f"{slug}_03_matrix")

            # 4) Bland-Altman（参考值 vs 最差组 或 第一组）
            gs = comparison.group_stats
            worst = gs["rms_dev"].idxmax() if len(gs) else None
            gname = worst if worst in alignment.groups else alignment.groups[0]
            a = alignment.mean[gname].to_numpy(dtype="float64")
            b = comparison.reference.to_numpy(dtype="float64")
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() >= 3:
                ba = U.bland_altman(a[m], b[m], cfg.conf)
                p = C.Plot(title=f"{display} — Bland-Altman（{gname} vs {comparison.reference_name}）",
                           xlabel=f"两者均值 / {unit}" if unit else "两者均值",
                           ylabel=f"差值 ({gname} - 参考) / {unit}" if unit else "差值")
                p.scatter((a[m] + b[m]) / 2, a[m] - b[m], marker="o", markersize=4,
                          color=C.PALETTE[0], label="测量点")
                p.hline(ba["bias"], label=f"平均偏差 {ba['bias']:.4g}", color="#1f77b4", dash=(6, 4))
                p.hline(ba["loa_high"], label=f"+1.96SD {ba['loa_high']:.4g}", color="#d62728", dash=(4, 4))
                p.hline(ba["loa_low"], label=f"-1.96SD {ba['loa_low']:.4g}", color="#d62728", dash=(4, 4))
                emit(f"{display}: Bland-Altman 一致性", p, f"{slug}_04_bland_altman")

    # 5) 箱线图：有参考值时画"相对参考值的偏差分布"，否则画原始值分布
    if len(groups) >= 1:
        use_dev = comparison is not None and alignment is not None
        title = (f"{display} — 各组相对参考值的偏差分布" if use_dev
                 else f"{display} — 各组数据分布")
        ylab = (f"偏差 / {unit}" if unit else "偏差") if use_dev else y_label
        p = C.Plot(title=title, ylabel=ylab, width=820, height=500)
        box_data: dict[str, np.ndarray] = {}
        for g in groups:
            if use_dev and g.group in comparison.deviations.columns:
                vals = comparison.deviations[g.group].to_numpy(dtype="float64")
            else:
                vals = df[df["group"].astype(str) == g.group]["value"].to_numpy(dtype="float64")
            box_data[g.group] = vals
        p.box(box_data)
        if use_dev:
            p.hline(0.0, label="零偏差", color="#111111", dash=(6, 4))
        for gi, g in enumerate(groups):
            vals = box_data[g.group]
            vals = vals[np.isfinite(vals)]
            jitter = (np.random.default_rng(0).random(vals.size) - 0.5) * 0.24
            p.scatter(np.full(vals.size, gi, dtype="float64") + jitter, vals,
                      marker="o", markersize=2.6, color="#344054", opacity=0.7)
        emit(f"{display}: 偏差分布箱线图" if use_dev else f"{display}: 分布箱线图",
             p, f"{slug}_05_box")

    # 6) 直方图（全体数据）
    allv = kept["value"].to_numpy(dtype="float64")
    if allv.size >= 3:
        p = C.Plot(title=f"{display} — 全体数据直方图", xlabel=y_label, ylabel="概率密度")
        p.hist(allv, bins=min(15, max(4, int(math.sqrt(allv.size)) + 1)), density=True,
               normal_curve=True, label="频数分布")
        emit(f"{display}: 直方图", p, f"{slug}_06_hist")

    # 7) 拟合线（以参考值为准）
    if fits:
        p = C.Plot(title=f"{display} — 线性拟合", xlabel=cfg.x_label, ylabel=y_label)
        xr = np.asarray(alignment.x, dtype="float64")
        for gi, g in enumerate(alignment.groups):
            p.errorbar(xr, alignment.mean[g].to_numpy(dtype="float64"),
                       alignment.u[g].to_numpy(dtype="float64"), label=g,
                       color=C.PALETTE[gi % len(C.PALETTE)], marker="o", connect=False, capsize=3)
        if "__参考值__" in fits:
            f = fits["__参考值__"]
            xs = np.linspace(float(xr.min()), float(xr.max()), 50)
            p.line(xs, f.predict(xs), label=f"参考拟合 y={f.slope:.5g}x+{f.intercept:.5g}",
                   color="#111111", width=2.2, dash=(5, 3))
        emit(f"{display}: 线性拟合", p, f"{slug}_07_fit")

    # 8) 不确定度贡献（派生量）
    for di, dv in enumerate(derived):
        contrib = dv.get("contributions") or {}
        if not contrib:
            continue
        names = list(contrib.keys())
        vals = [abs(contrib[n]) for n in names]
        p = C.Plot(title=f"{dv['name']} 的不确定度来源", xlabel="标准不确定度贡献", width=780,
                   height=460)
        p.bars(names, vals, colors=[C.PALETTE[i % len(C.PALETTE)] for i in range(len(names))],
               horizontal=True)
        emit(f"{dv['name']}: 不确定度构成", p, f"{slug}_08_contrib_{di}")

    conclusions = _conclusions(display, unit, groups, alignment, comparison, fits, derived, cfg)
    tbl = _summary_table(groups, comparison, unit)

    return VariableResult(variable=variable, unit=unit, y_label=y_label, groups=groups,
                          alignment=alignment, comparison=comparison, fits=fits,
                          derived=derived, conclusions=conclusions, charts=charts,
                          chart_files=chart_files, table=tbl, label=display)


def _summary_table(groups: list[GroupResult], comparison: A.Comparison | None, unit: str) -> pd.DataFrame:
    rows = []
    for g in groups:
        s = g.summary
        rows.append({
            "组": g.group, "n": s["n"], "剔除后n": g.used_n,
            "均值": s["mean"], "标准差": s["std"], "u_A": s["sem"],
            "相对标准差%": (s["rsd"] * 100 if s["rsd"] is not None else np.nan),
            "最小值": s["minimum"], "最大值": s["maximum"],
            "离群数": len(g.outliers),
        })
    df = pd.DataFrame(rows)
    if comparison is not None and not comparison.group_stats.empty:
        gs = comparison.group_stats
        df = df.merge(gs[["mean_dev", "rms_dev", "max_abs_dev", "mean_rel_dev_pct"]],
                      left_on="组", right_index=True, how="left")
    if unit:
        df.attrs["unit"] = unit
    return df


def _conclusions(variable: str, unit: str, groups: list[GroupResult], al: A.Alignment | None,
                 cmp: A.Comparison | None, fits: dict, derived: list[dict],
                 cfg: AnalysisConfig) -> list[str]:
    out: list[str] = []
    unit_s = f" {unit}" if unit else ""
    for g in groups:
        s, ta = g.summary, g.type_a
        out.append(
            f"【{g.group}】n={s['n']}（剔除离群后 {g.used_n}），均值 {s['mean']:.6g}{unit_s}，"
            f"样本标准差 s={s['std']:.4g}，A 类标准不确定度 u={ta['u_A']:.4g}，"
            f"扩展不确定度 U={ta['U']:.4g}（k={ta['k']:.3f}, {int(cfg.conf*100)}%，自由度 {s['dof']}），"
            f"相对标准差 {s['rsd']*100:.3f}%。")
        if g.outliers:
            vals = ", ".join(f"{o['value']:.6g}（G={o['G']:.3f}>{o['G_crit']:.3f}）" for o in g.outliers)
            out.append(f"【{g.group}】检出 {len(g.outliers)} 个离群值：{vals}；"
                       f"{'已剔除' if cfg.exclude_outliers else '本轮保留，建议复核'}。")
    if al is not None and cmp is not None:
        ov = cmp.overall
        out.append(
            f"对齐：{ov['n_groups']} 组、{ov['n_grid_points']} 个公共网格点（方法 {al.method}）；"
            f"最大偏差 {ov['max_abs_dev']:.4g}{unit_s}，"
            f"RMS 偏差 {ov['rms_dev']:.4g}{unit_s}，"
            f"平均绝对偏差 {ov['mean_abs_dev']:.4g}{unit_s}。")
        verdict = "各测量系统之间一致性良好" if ov.get("consistent") else "各测量系统之间存在系统偏差"
        out.append(
            f"一致性：合并 chi2={ov['pooled_chi2']:.3f}（自由度 {ov['pooled_dof']}），"
            f"chi2/dof={ov['pooled_chi2_red']:.3f}，Birge 比 {ov['birge']:.3f}，"
            f"p={ov['pooled_p_value']:.4g} -> {verdict}；"
            f"有 {ov['n_inconsistent_rows']} 个网格点的组间离散超出各自不确定度。")
        if not cmp.group_stats.empty:
            gs = cmp.group_stats
            wi = gs["rms_dev"].idxmax()
            bi = gs["mean_rel_dev_pct"].abs().idxmax()
            out.append(
                f"偏差最大分组：{wi}（RMS 偏差 {gs.loc[wi, 'rms_dev']:.4g}{unit_s}）；"
                f"平均相对偏差绝对值最大分组：{bi}（{gs.loc[bi, 'mean_rel_dev_pct']:+.4f}%）。")
        out.append(
            f"推荐值（各网格点加权平均的合成）：{U.format_measurement(ov['recommended_value'], ov['recommended_u'], unit)}"
            f"；U(k={cfg.k:g})={cfg.k*ov['recommended_u']:.4g}{unit_s}。")
        if not cmp.pairwise.empty:
            pw = cmp.pairwise
            sig = pw[pw["significant"]]
            if len(sig):
                desc = "；".join(f"{r.group_a} vs {r.group_b} 平均差 {r.mean_diff:+.4g}（p={r.p_value:.3g}）"
                                 for r in sig.itertuples())
                out.append(f"两两配对 t 检验显示显著性差异：{desc}。")
            else:
                out.append("两两配对 t 检验：所有组对之间均无显著性差异（α=0.05）。")
            best = pw.loc[pw["pct_within_loa"].idxmax()] if pw["pct_within_loa"].notna().any() else None
            if best is not None:
                out.append(
                    f"Bland-Altman：一致性最好的一对是 {best.group_a} vs {best.group_b}，"
                    f"偏差 {best.bland_bias:+.4g}，一致性界限 [{best.loa_low:.4g}, {best.loa_high:.4g}]，"
                    f"{best.pct_within_loa:.1f}% 的点落在界限内。")
    for name, f in fits.items():
        if name == "__参考值__":
            continue
        out.append(
            f"拟合（{name}）：斜率 {f.slope:.6g} ± {f.u_slope:.3g}"
            f"（相对 {abs(f.slope_rel)*100:.3f}%），截距 {f.intercept:.6g} ± {f.u_intercept:.3g}，"
            f"R2={f.r2:.6f}，残差标准差 {f.s_resid:.4g}，chi2/dof={f.chi2_red:.3f}。")
    if "__参考值__" in fits:
        f = fits["__参考值__"]
        out.append(f"参考值线性模型：y = {f.slope:.6g}·x {f.intercept:+.6g}，"
                   f"R2={f.r2:.6f}，斜率标准不确定度 {f.u_slope:.3g}。")
    for d in derived:
        if "error" in d:
            out.append(f"派生量 {d.get('name')} 计算失败：{d['error']}")
            continue
        top = sorted((d.get("contributions") or {}).items(), key=lambda kv: -abs(kv[1]))[:3]
        tp = "，".join(f"{k} 贡献 {v:.4g}" for k, v in top)
        out.append(f"派生量【{d['name']}】= {d['value']:.6g} ± {d['u']:.4g} "
                   f"(U={d['U']:.4g}, k={d['k']:.3g}) {d.get('unit','')}；"
                   f"相对不确定度 {d['relative_pct']:.3f}%；主要来源：{tp}。")
    return out


def analyse_dataset(ds: Dataset, cfg: AnalysisConfig, *, prefix: str = "") -> AnalysisResult:
    """分析整个数据集（可能含多个测量变量）。"""
    out_vars: list[VariableResult] = []
    for v in ds.variables:
        unit_hint = ""
        sub = ds.frame[ds.frame["variable"] == v]
        if "unit" in sub and sub["unit"].astype(str).str.len().gt(0).any():
            unit_hint = str(sub["unit"].mode().iloc[0])
        sub_cfg = AnalysisConfig(**{**cfg.__dict__, "unit": cfg.unit or unit_hint,
                                    "value_label": cfg.value_label})
        res = analyse_variable(ds.frame, v, sub_cfg, prefix=prefix,
                               backend=cfg.chart_backend)
        out_vars.append(res)
    return AnalysisResult(dataset=ds.name, variables=out_vars, config=cfg,
                          notes=list(ds.notes))
