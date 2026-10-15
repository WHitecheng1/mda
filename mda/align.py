"""多组测量的对齐与比较。

把"不同仪器 / 不同实验室 / 不同批次"在同一自变量（温度、时间、标称值…）上的
测量结果对齐到同一网格，然后做偏差、相关性、一致性(Bland-Altman)、
两两系统偏差与线性回归比较，并给出推荐值（加权平均）与一致性判据。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import dist as D
from . import uncertainty as U

__all__ = ["Alignment", "Comparison", "align_groups", "resample_time",
           "weighted_reference", "compare_alignment"]


# --------------------------------------------------------------------------
# 对齐
# --------------------------------------------------------------------------
@dataclass
class Alignment:
    """对齐后的多组数据。"""

    x: np.ndarray
    mean: pd.DataFrame          # index=x, 列=组，值=该 x 上的均值
    u: pd.DataFrame             # 标准不确定度（重复测量时为 s/sqrt(n)）
    n: pd.DataFrame             # 参与平均的点数
    method: str = "nearest"
    x_label: str = "x"
    is_time: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def groups(self) -> list[str]:
        return list(self.mean.columns)

    def __len__(self) -> int:
        return int(self.mean.shape[0])

    def coverage(self) -> pd.Series:
        return self.mean.notna().sum(axis=0)

    def as_arrays(self) -> tuple[np.ndarray, dict]:
        return np.asarray(self.x, dtype="float64"), {
            g: (self.mean[g].to_numpy(dtype="float64"),
                self.u[g].to_numpy(dtype="float64"),
                self.n[g].to_numpy(dtype="float64"))
            for g in self.groups
        }


def _as_x(series: pd.Series) -> tuple[np.ndarray, bool]:
    """把 x 列转成数值数组；若能解析为时间则返回 ns 时间戳。"""
    s = series.copy()
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.astype("int64").to_numpy(dtype="float64"), True
    if pd.api.types.is_numeric_dtype(s):
        return s.to_numpy(dtype="float64"), False
    parsed = pd.to_datetime(s, errors="coerce", format="mixed")
    if parsed.notna().mean() >= 0.8:
        return parsed.astype("int64").to_numpy(dtype="float64"), True
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64"), False


def _cluster_members(values: np.ndarray, tol: float) -> tuple[np.ndarray, list[list[int]]]:
    order = np.argsort(values, kind="stable")
    centers: list[float] = []
    members: list[list[int]] = []
    for i in order:
        if centers and abs(values[i] - centers[-1]) <= tol:
            members[-1].append(int(i))
            centers[-1] = float(np.mean([values[j] for j in members[-1]]))
        else:
            centers.append(float(values[i]))
            members.append([int(i)])
    return np.asarray(centers, dtype="float64"), members


def align_groups(long: pd.DataFrame, *, method: str = "nearest", tol: float | None = None,
                 grid: Sequence[float] | None = None, x_label: str = "x",
                 min_coverage: int | None = None) -> Alignment:
    """把长表对齐到公共网格。

    method:
      * union        取所有组 x 的并集（相近点在 tol 内合并）
      * intersection 只保留所有组都有的 x
      * nearest      以并集为网格，每组就近匹配（超过 tol 记为缺失），
                     并丢弃覆盖组数不足的网格点
      * grid         使用显式给定的网格
    tol: 数值容差（数值型 x 默认取跨度的 1e-6；时间型默认 0）
    """
    notes: list[str] = []
    if long.empty:
        raise ValueError("空的测量表")
    groups = list(dict.fromkeys(long["group"].astype(str)))
    xs, is_time = _as_x(long["x"])
    long = long.assign(__x__=xs)
    span = float(np.nanmax(xs) - np.nanmin(xs)) if xs.size else 0.0
    if tol is None:
        if is_time:
            tol = 0.0 if method in ("union", "intersection") else 1.0  # 1 ns
        else:
            tol = span * 1e-6 if method in ("union", "intersection") else max(span * 1e-6, 1e-12)

    finite = np.isfinite(xs)
    all_x = np.unique(xs[finite])
    centers, _members = _cluster_members(all_x, tol)
    groups_arr = long["group"].astype(str).to_numpy()
    cover: dict[float, set] = {}
    if centers.size:
        d = np.abs(xs[:, None] - centers[None, :])
        d[~finite, :] = np.inf
        assign = np.argmin(d, axis=1)
        for i, c in enumerate(centers):
            cover[float(c)] = set(groups_arr[(assign == i) & finite])

    if grid is not None:
        target = np.asarray(grid, dtype="float64")
        notes.append("使用显式网格")
    elif method == "intersection":
        need = set(groups)
        target = np.asarray([c for c in centers if cover.get(float(c), set()) >= need], dtype="float64")
        notes.append(f"交集网格: {target.size} 点")
    else:
        target = centers
        notes.append(f"并集网格: {target.size} 点")

    if target.size == 0:
        raise ValueError("对齐后没有任何公共网格点，请放宽容差或改用 union")

    rows: dict[str, list[float]] = {}
    urows: dict[str, list[float]] = {}
    nrows: dict[str, list[float]] = {}
    for g in groups:
        sub = long[long["group"].astype(str) == g]
        sx = sub["__x__"].to_numpy(dtype="float64")
        sv = sub["value"].to_numpy(dtype="float64")
        su = sub["u"].to_numpy(dtype="float64") if "u" in sub else np.full(sx.size, np.nan)
        means, us, ns = [], [], []
        for t in target:
            d = np.abs(sx - t)
            if d.size and np.nanmin(d) <= tol:
                sel = d <= tol
                vals = sv[sel]
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    means.append(np.nan); us.append(np.nan); ns.append(0.0); continue
                m = float(np.mean(vals))
                if vals.size > 1:
                    sd = float(np.std(vals, ddof=1))
                    uu = sd / math.sqrt(vals.size)
                else:
                    uu = np.nan
                if np.any(np.isfinite(su[sel])):
                    given = su[sel][np.isfinite(su[sel])]
                    uu = float(math.sqrt(np.nanmean(given ** 2) + (uu ** 2 if np.isfinite(uu) else 0.0)))
                means.append(m); us.append(uu); ns.append(float(vals.size))
            else:
                means.append(np.nan); us.append(np.nan); ns.append(0.0)
        rows[g], urows[g], nrows[g] = means, us, ns

    mean_df = pd.DataFrame(rows, index=pd.Index(target, name=x_label))
    u_df = pd.DataFrame(urows, index=mean_df.index)
    n_df = pd.DataFrame(nrows, index=mean_df.index)

    if method == "nearest":
        need = min_coverage if min_coverage is not None else max(2, len(groups) - 1)
        keep = mean_df.notna().sum(axis=1) >= need
        if not keep.all():
            notes.append(f"按最小覆盖组数 {need} 过滤掉 {int((~keep).sum())} 个网格点")
            mean_df, u_df, n_df = mean_df[keep], u_df[keep], n_df[keep]
            target = target[keep.to_numpy()]

    if is_time:
        idx = pd.to_datetime(target.astype("int64"))
        mean_df.index = idx
        u_df.index = idx
        n_df.index = idx
        notes.append("x 轴按时间解析")

    return Alignment(x=target, mean=mean_df, u=u_df, n=n_df, method=method,
                     x_label=x_label, is_time=is_time, notes=notes)


def resample_time(long: pd.DataFrame, rule: str = "1min", *, how: str = "mean") -> pd.DataFrame:
    """时间序列重采样对齐（每组独立聚合到固定时间间隔）。"""
    df = long.copy()
    t = pd.to_datetime(df["x"], errors="coerce", format="mixed")
    df["x"] = t
    df = df.dropna(subset=["x"])
    out = []
    for g, sub in df.groupby("group", sort=False):
        s = sub.set_index("x")["value"].resample(rule)
        agg = s.mean() if how == "mean" else s.median()
        cnt = s.count()
        sd = s.std()
        part = pd.DataFrame({"group": g, "value": agg, "n": cnt,
                             "u": sd / np.sqrt(cnt.replace(0, np.nan))})
        out.append(part.reset_index())
    return pd.concat(out, ignore_index=True)


def weighted_reference(mean_df: pd.DataFrame, u_df: pd.DataFrame) -> tuple[pd.Series, pd.Series, dict]:
    """按行（每个 x）做反方差加权平均，得到参考值与一致性统计。"""
    ref, refu, stats = [], [], {"rows": []}
    for i, xv in enumerate(mean_df.index):
        vals, sig = [], []
        for g in mean_df.columns:
            v = mean_df.iloc[i][g]
            if pd.notna(v):
                vals.append(float(v))
                uu = u_df.iloc[i][g]
                sig.append(float(uu) if pd.notna(uu) and uu > 0 else np.nan)
        if not vals:
            ref.append(np.nan); refu.append(np.nan)
            continue
        wm = U.weighted_mean(vals, sig)
        ref.append(wm.mean); refu.append(wm.u)
        stats["rows"].append({"x": str(xv), "n": len(vals), "mean": wm.mean, "u": wm.u,
                              "chi2_red": wm.chi2_red, "birge": wm.birge,
                              "consistent": wm.consistent})
    return pd.Series(ref, index=mean_df.index), pd.Series(refu, index=mean_df.index), stats


# --------------------------------------------------------------------------
# 比较
# --------------------------------------------------------------------------
@dataclass
class Comparison:
    reference_name: str
    reference: pd.Series
    reference_u: pd.Series
    deviations: pd.DataFrame         # 各组 - 参考值
    deviations_rel: pd.DataFrame     # 相对偏差 (%)
    group_stats: pd.DataFrame        # 每组偏差汇总
    pairwise: pd.DataFrame           # 两两比较汇总
    row_consistency: pd.DataFrame    # 每个 x 上的加权一致性与 chi2
    overall: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "reference": self.reference_name,
            "group_stats": self.group_stats.reset_index().to_dict("records"),
            "pairwise": self.pairwise.reset_index().to_dict("records"),
            "overall": self.overall,
        }


def compare_alignment(al: Alignment, *, reference: str | None = None,
                      conf: float = 0.95) -> Comparison:
    """对齐之后：与参考值比较 + 两两比较 + 一致性判据。

    reference: None 表示"加权平均参考值(model)"，也可指定某一组作为参考。
    """
    if reference and reference in al.mean.columns:
        ref = al.mean[reference].astype("float64").copy()
        refu = al.u[reference].astype("float64").copy()
        ref_name = f"组 {reference}"
        row_stats = pd.DataFrame({"x": [str(x) for x in al.mean.index]})
    else:
        ref, refu, rst = weighted_reference(al.mean, al.u)
        ref_name = "加权平均参考值"
        row_stats = pd.DataFrame(rst["rows"]) if rst["rows"] else pd.DataFrame()
    ref = ref.astype("float64")

    dev = al.mean.astype("float64").sub(ref, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dev_rel = dev.div(ref.abs().replace(0, np.nan), axis=0) * 100.0

    gstats = []
    for g in al.mean.columns:
        d = dev[g].dropna()
        dr = dev_rel[g].dropna()
        vals = al.mean[g].dropna()
        rv = ref.reindex(vals.index).dropna()
        pair = pd.concat([vals, rv], axis=1).dropna()
        lin = None
        if pair.shape[0] >= 3:
            try:
                fit = U.linear_fit(pair.iloc[:, 1].to_numpy(), pair.iloc[:, 0].to_numpy())
                lin = fit
            except Exception:
                lin = None
        gstats.append({
            "group": g,
            "n_points": int(len(vals)),
            "coverage": round(100.0 * len(vals) / max(len(al.mean), 1), 1),
            "mean_value": float(vals.mean()) if len(vals) else np.nan,
            "mean_dev": float(d.mean()) if len(d) else np.nan,
            "rms_dev": float(math.sqrt((d ** 2).mean())) if len(d) else np.nan,
            "max_abs_dev": float(d.abs().max()) if len(d) else np.nan,
            "mean_rel_dev_pct": float(dr.mean()) if len(dr) else np.nan,
            "max_rel_dev_pct": float(dr.abs().max()) if len(dr) else np.nan,
            "slope_vs_ref": float(lin.slope) if lin else np.nan,
            "u_slope": float(lin.u_slope) if lin else np.nan,
            "intercept_vs_ref": float(lin.intercept) if lin else np.nan,
            "r2_vs_ref": float(lin.r2) if lin else np.nan,
        })
    group_stats = pd.DataFrame(gstats).set_index("group")

    # 两两比较
    pairs = []
    cols = list(al.mean.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a = al.mean[cols[i]].astype("float64")
            b = al.mean[cols[j]].astype("float64")
            m = a.notna() & b.notna()
            if int(m.sum()) < 2:
                continue
            av, bv = a[m].to_numpy(), b[m].to_numpy()
            tt = U.t_test_paired(av, bv, conf)
            ba = U.bland_altman(av, bv, conf)
            fit = None
            if m.sum() >= 3:
                try:
                    fit = U.linear_fit(bv, av)
                except Exception:
                    fit = None
            pairs.append({
                "group_a": cols[i], "group_b": cols[j], "n_pairs": int(m.sum()),
                "mean_a": float(av.mean()), "mean_b": float(bv.mean()),
                "mean_diff": float(av.mean() - bv.mean()),
                "rms_diff": float(math.sqrt(np.mean((av - bv) ** 2))),
                "mean_rel_diff_pct": float(np.mean((av - bv) / bv) * 100) if np.all(bv != 0) else np.nan,
                "p_value": float(tt.get("p", np.nan)),
                "significant": bool(tt.get("reject", False)),
                "bland_bias": float(ba.get("bias", np.nan)),
                "loa_low": float(ba.get("loa_low", np.nan)),
                "loa_high": float(ba.get("loa_high", np.nan)),
                "pct_within_loa": float(ba.get("pct_within", np.nan)),
                "slope_a_on_b": float(fit.slope) if fit else np.nan,
                "u_slope": float(fit.u_slope) if fit else np.nan,
                "r2": float(fit.r2) if fit else np.nan,
            })
    pairwise = pd.DataFrame(pairs)

    # 每个 x 上的一致性（用参考值口径不同：这里重新做加权 chi2）
    row_stats2 = []
    for i, xv in enumerate(al.mean.index):
        vals, sig = [], []
        for g in al.mean.columns:
            v = al.mean.iloc[i][g]
            if pd.notna(v):
                vals.append(float(v))
                uu = al.u.iloc[i][g]
                sig.append(float(uu) if pd.notna(uu) and uu > 0 else np.nan)
        if len(vals) >= 2:
            wm = U.weighted_mean(vals, sig, conf=conf)
            row_stats2.append({"x": xv, "n": len(vals), "value": wm.mean, "u": wm.u,
                               "chi2": wm.chi2, "chi2_red": wm.chi2_red, "birge": wm.birge,
                               "p_value": wm.p_value, "consistent": wm.consistent})
    row_consistency = pd.DataFrame(row_stats2)
    if not row_consistency.empty:
        row_consistency = row_consistency.set_index("x")

    # 总体统计
    all_dev = dev.stack().dropna() if hasattr(dev, "stack") else pd.Series(dtype=float)
    pooled_chi2 = float(row_consistency["chi2"].sum()) if not row_consistency.empty else float("nan")
    pooled_dof = int(row_consistency["n"].sub(1).clip(lower=0).sum()) if not row_consistency.empty else 0
    overall = {
        "reference": ref_name,
        "n_groups": len(al.mean.columns),
        "n_grid_points": int(len(al.mean)),
        "mean_abs_dev": float(all_dev.abs().mean()) if len(all_dev) else float("nan"),
        "rms_dev": float(math.sqrt((all_dev ** 2).mean())) if len(all_dev) else float("nan"),
        "max_abs_dev": float(all_dev.abs().max()) if len(all_dev) else float("nan"),
        "pooled_chi2": pooled_chi2,
        "pooled_dof": pooled_dof,
        "pooled_chi2_red": (pooled_chi2 / pooled_dof) if pooled_dof > 0 else float("nan"),
        "pooled_p_value": float(D.chi2_sf(pooled_chi2, pooled_dof)) if pooled_dof > 0 else float("nan"),
        "n_inconsistent_rows": int((~row_consistency["consistent"]).sum()) if not row_consistency.empty else 0,
        "recommended_value": float(ref.mean()) if len(ref.dropna()) else float("nan"),
        "recommended_u": float(np.sqrt(np.nanmean(refu.to_numpy(dtype="float64") ** 2)))
        if refu.notna().any() else float("nan"),
    }
    if np.isfinite(overall["pooled_chi2_red"]):
        overall["birge"] = math.sqrt(max(overall["pooled_chi2_red"], 0.0))
        overall["consistent"] = bool(overall["pooled_chi2_red"] <= 2.0)
    else:
        overall["birge"] = float("nan")
        overall["consistent"] = None

    return Comparison(reference_name=ref_name, reference=ref, reference_u=refu,
                      deviations=dev, deviations_rel=dev_rel, group_stats=group_stats,
                      pairwise=pairwise, row_consistency=row_consistency, overall=overall)
