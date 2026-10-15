"""误差、不确定度与统计检验。

覆盖 GUM 体系下最常用的一套计算：
* A 类不确定度（重复测量统计：均值、样本标准差、均值的标准不确定度、t 区间）
* B 类不确定度（均匀/三角/正态/分辨力 等规格换算为标准不确定度）
* 合成标准不确定度、Welch–Satterthwaite 有效自由度、扩展不确定度
* 加权平均与一致性检验（chi2、Birge 比）
* 最小二乘线性拟合（含斜率/截距的不确定度与预测区间）
* 离群值判别（Grubbs 迭代、MAD 稳健判据）
* 间接测量量的不确定度传播（数值偏导）
* t 检验、方差分析、Bland-Altman、相关系数
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import dist as D

__all__ = [
    "stats_summary", "type_a", "type_b", "combine_uncertainties", "expanded_uncertainty",
    "welch_satterthwaite", "weighted_mean", "linear_fit", "propagate",
    "outliers_grubbs", "outliers_mad", "grubbs_critical",
    "t_test_1sample", "t_test_paired", "t_test_2sample", "anova_oneway",
    "bland_altman", "pearson", "spearman", "format_measurement", "sig_round",
    "Summary", "TypeAResult", "WeightedMean", "LinearFit", "Propagation",
]


def _arr(values) -> np.ndarray:
    a = np.asarray(pd.Series(list(values) if not isinstance(values, (pd.Series, np.ndarray)) else values,
                             dtype="float64"), dtype="float64")
    return a[np.isfinite(a)]


def sig_round(value: float, uncertainty: float, sig: int = 2) -> tuple[float, float, int]:
    """按测量不确定度的有效位数对测量结果修约（GUM 惯例）。

    不确定度保留 sig 位有效数字，测量值修约到同一位。
    返回 (修约后的值, 修约后的不确定度, 小数位数)。
    """
    if not np.isfinite(value):
        return value, uncertainty, 0
    if not np.isfinite(uncertainty) or uncertainty == 0:
        return value, uncertainty, 0
    exp = math.floor(math.log10(abs(uncertainty)))
    decimals = sig - 1 - exp
    decimals = int(np.clip(decimals, -12, 12))
    u_r = round(uncertainty, decimals)
    v_r = round(value, decimals)
    return v_r, u_r, max(decimals, 0)


def format_measurement(value: float, uncertainty: float | None = None, unit: str | None = None,
                       sig: int = 2) -> str:
    """把"值±不确定度"格式化为规范字符串，如 12.345 ± 0.012 mm。"""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if uncertainty is None or not np.isfinite(uncertainty) or uncertainty == 0:
        s = f"{value:.6g}"
    else:
        v_r, u_r, dec = sig_round(value, uncertainty, sig)
        s = f"{v_r:.{dec}f} ± {u_r:.{dec}f}"
    if unit:
        s += f" {unit}"
    return s


# --------------------------------------------------------------------------
# 基本统计
# --------------------------------------------------------------------------
@dataclass
class Summary:
    n: int
    mean: float
    std: float          # 样本标准差 (ddof=1)
    sem: float          # 均值的标准不确定度 (A 类)
    dof: int
    minimum: float
    maximum: float
    median: float
    q1: float
    q3: float
    iqr: float
    range_: float
    rsd: float          # 相对标准差
    skewness: float = float("nan")
    kurtosis: float = float("nan")

    def as_dict(self) -> dict:
        return {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                for k, v in self.__dict__.items()}


def stats_summary(values) -> Summary:
    a = _arr(values)
    n = a.size
    if n == 0:
        raise ValueError("空数据无法统计")
    mean = float(np.mean(a))
    std = float(np.std(a, ddof=1)) if n > 1 else float("nan")
    sem = std / math.sqrt(n) if n > 1 else float("nan")
    q1, med, q3 = (float(x) for x in np.percentile(a, [25, 50, 75]))
    skew = float(pd.Series(a).skew()) if n > 2 else float("nan")
    kurt = float(pd.Series(a).kurtosis()) if n > 3 else float("nan")
    return Summary(
        n=n, mean=mean, std=std, sem=sem, dof=n - 1,
        minimum=float(np.min(a)), maximum=float(np.max(a)), median=med,
        q1=q1, q3=q3, iqr=q3 - q1, range_=float(np.max(a) - np.min(a)),
        rsd=(std / abs(mean) if n > 1 and mean != 0 else float("nan")),
        skewness=skew, kurtosis=kurt,
    )


@dataclass
class TypeAResult:
    """A 类（统计）不确定度评定结果。"""

    summary: Summary
    conf: float
    k: float            # t 分位数
    u: float            # 均值标准不确定度 = s/sqrt(n)
    U: float            # 扩展不确定度 = k*u
    interval: tuple[float, float]

    def as_dict(self) -> dict:
        d = self.summary.as_dict()
        d.update({"conf": self.conf, "k": self.k, "u_A": self.u, "U": self.U,
                  "ci_low": self.interval[0], "ci_high": self.interval[1]})
        return d


def type_a(values, conf: float = 0.95) -> TypeAResult:
    """A 类不确定度：由重复测量列直接评定。"""
    s = stats_summary(values)
    k = D.t_crit(conf, s.dof) if s.dof > 0 else float("nan")
    u = s.sem
    U = k * u if np.isfinite(k) and np.isfinite(u) else float("nan")
    return TypeAResult(s, conf, k, u, U, (s.mean - U, s.mean + U))


# --------------------------------------------------------------------------
# B 类不确定度
# --------------------------------------------------------------------------
B_KINDS = {
    "uniform": ("均匀(矩形)", "a/sqrt(3)"),
    "rect": ("均匀(矩形)", "a/sqrt(3)"),
    "均匀": ("均匀(矩形)", "a/sqrt(3)"),
    "矩形": ("均匀(矩形)", "a/sqrt(3)"),
    "triangular": ("三角", "a/sqrt(6)"),
    "三角": ("三角", "a/sqrt(6)"),
    "normal": ("正态", "a/k"),
    "正态": ("正态", "a/k"),
    "resolution": ("分辨力", "d/(2*sqrt(3))"),
    "分辨力": ("分辨力", "d/(2*sqrt(3))"),
    "半宽": ("正态", "a/k"),
}


def type_b(spec: str, a: float, k: float = 2.0) -> float:
    """B 类不确定度：把规格/误差限换算为标准不确定度。

    spec: uniform/矩形(半宽 a)、triangular/三角(半宽 a)、
          normal/正态(置信区间半宽 a, 包含因子 k)、resolution/分辨力(分辨力 d)
    """
    key = str(spec).strip().lower()
    if key not in B_KINDS:
        raise ValueError(f"不支持的 B 类类型: {spec}（可选 {sorted(set(B_KINDS))}）")
    if key in ("uniform", "rect", "均匀", "矩形"):
        return abs(a) / math.sqrt(3.0)
    if key in ("triangular", "三角"):
        return abs(a) / math.sqrt(6.0)
    if key in ("normal", "正态", "半宽"):
        return abs(a) / abs(k) if k else float("nan")
    return abs(a) / (2.0 * math.sqrt(3.0))  # 分辨力


def welch_satterthwaite(components: Mapping[str, tuple[float, float | None]]) -> float:
    """有效自由度 veff = uc^4 / sum(u_i^4 / v_i)。"""
    num = 0.0
    den = 0.0
    for u, v in components.values():
        if u is None or not np.isfinite(u) or u == 0:
            continue
        num += u ** 2
        if v is not None and np.isfinite(v) and v > 0:
            den += u ** 4 / v
    if den <= 0:
        return float("inf")
    return num ** 2 / den


@dataclass
class Propagation:
    """不确定度传播结果。"""

    value: float
    uc: float
    sensitivities: dict
    contributions: dict
    dof_eff: float = float("inf")

    def top(self, n: int = 5) -> list[tuple[str, float, float]]:
        """返回贡献最大的若干项 (名称, 贡献 u_i, 占比)。

        占比按方差份额 (u_i/uc)² 计算——合成不确定度是方和根，用线性比会得到
        总和大于 100% 的结果。
        """
        items = sorted(self.contributions.items(), key=lambda kv: -abs(kv[1]))
        return [(k, v, (v / self.uc) ** 2 if self.uc else float("nan")) for k, v in items[:n]]

    def as_dict(self) -> dict:
        return {"value": self.value, "uc": self.uc, "dof_eff": self.dof_eff,
                "contributions": self.contributions, "sensitivities": self.sensitivities}


def combine_uncertainties(components: Mapping[str, float],
                          dofs: Mapping[str, float] | None = None) -> Propagation:
    """合成标准不确定度（互不相关，方和根）。"""
    comps = {k: float(v) for k, v in components.items() if v is not None and np.isfinite(v)}
    uc = math.sqrt(sum(v * v for v in comps.values()))
    dofs = dict(dofs or {})
    veff = welch_satterthwaite({k: (v, dofs.get(k)) for k, v in comps.items()}) if dofs else float("inf")
    return Propagation(value=float("nan"), uc=uc, sensitivities={k: 1.0 for k in comps},
                       contributions=comps, dof_eff=veff)


def expanded_uncertainty(uc: float, k: float = 2.0, conf: float | None = None,
                         dof: float | None = None) -> float:
    """扩展不确定度 U = k*uc；给出 conf 与有效自由度时用 t 分位数作 k。"""
    if conf is not None and dof is not None and np.isfinite(dof):
        kk = D.t_crit(conf, dof)
    else:
        kk = k
    return kk * uc


def propagate(func: Callable[..., float], variables: Mapping[str, tuple[float, float]],
              relative: bool = False, dofs: Mapping[str, float] | None = None) -> Propagation:
    """间接测量量的不确定度传播（中心差分求灵敏度系数）。

    variables: {名称: (最佳估计值, 标准不确定度)}
    relative : True 时输入不确定度按相对值解释（u/x）
    """
    names = list(variables.keys())
    x0 = np.array([float(variables[n][0]) for n in names], dtype="float64")
    u = np.array([float(variables[n][1]) for n in names], dtype="float64")
    if relative:
        u = np.abs(u * x0)
    value = float(func(*x0))

    sens = {}
    contrib = {}
    for i, n in enumerate(names):
        h = max(abs(x0[i]) * 1e-6, 1e-9)
        xp = x0.copy(); xp[i] += h
        xm = x0.copy(); xm[i] -= h
        c = (float(func(*xp)) - float(func(*xm))) / (2 * h)
        sens[n] = c
        contrib[n] = abs(c) * u[i]
    uc = math.sqrt(sum(v * v for v in contrib.values()))
    veff = welch_satterthwaite({k: (v, (dofs or {}).get(k)) for k, v in contrib.items()}) \
        if dofs else float("inf")
    return Propagation(value=value, uc=uc, sensitivities=sens, contributions=contrib, dof_eff=veff)


# --------------------------------------------------------------------------
# 加权平均与一致性
# --------------------------------------------------------------------------
@dataclass
class WeightedMean:
    mean: float
    u: float                 # 加权平均值的标准不确定度
    chi2: float
    dof: int
    chi2_red: float
    p_value: float
    birge: float             # Birge 比 = chi2_red 的平方根
    weights: list
    consistent: bool
    k: float = 2.0
    U: float = float("nan")
    n_missing_u: int = 0     # 传入的 sigmas 中缺失/非正、被按最大 sigma 填充的个数

    def as_dict(self) -> dict:
        return {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                for k, v in self.__dict__.items()}


def weighted_mean(values, sigmas=None, conf: float = 0.95) -> WeightedMean:
    """反方差加权平均 + chi2 一致性检验。

    sigmas 为各值标准不确定度；缺省或全为 0 时退化为等权平均。
    """
    y = _arr(values)
    n = y.size
    if n == 0:
        raise ValueError("空数据")
    if sigmas is None:
        s = np.full(n, np.nan)
    else:
        s = np.asarray(pd.Series(list(sigmas), dtype="float64"), dtype="float64")
        s = s[np.isfinite(np.asarray(values, dtype="float64"))]
    if s.size != n or not np.any(np.isfinite(s)) or np.nanmax(np.abs(s)) == 0:
        w = np.full(n, 1.0 / n)
    else:
        s = np.where(np.isfinite(s) & (np.abs(s) > 0), np.abs(s), np.nan)
        if not np.any(np.isfinite(s)):
            w = np.full(n, 1.0 / n)
        else:
            s_fill = np.where(np.isfinite(s), s, np.nanmax(s))
            ws = 1.0 / s_fill ** 2
            w = ws / ws.sum()
    mean = float(np.sum(w * y))
    var = float(np.sum(w ** 2 * (y - mean) ** 2)) if n > 1 else float("nan")
    # 反方差加权时 u = sqrt(sum w_i^2 sigma_i^2)；等权时按样本标准差估计
    n_missing_u = 0
    if sigmas is not None and np.any(np.isfinite(s)):
        s_fill = np.where(np.isfinite(s), s, np.nanmax(s))
        n_missing_u = int(np.sum(~np.isfinite(s)))
        if n_missing_u:
            # 不静默处理：缺失 u 的点按最大 sigma 给最小权重，并明确告知调用方
            warnings.warn(
                f"{n_missing_u} 个数据点缺少有效的不确定度，已按最大 sigma "
                f"({float(np.nanmax(s)):.6g}) 参与加权（等价于最小权重）；"
                f"如需忽略这些点请先自行过滤。",
                RuntimeWarning, stacklevel=2)
        u = math.sqrt(float(np.sum((w * s_fill) ** 2)))
    else:
        u = float(np.std(y, ddof=1) / math.sqrt(n)) if n > 1 else float("nan")

    if n > 1 and np.any(np.isfinite(s)):
        s_fill = np.where(np.isfinite(s), s, np.nanmax(s))
        chi2 = float(np.sum(((y - mean) / s_fill) ** 2))
        dof = n - 1
        chi2_red = chi2 / dof
        p = D.chi2_sf(chi2, dof)
        birge = math.sqrt(chi2_red)
    else:
        chi2, dof, chi2_red, p, birge = float("nan"), n - 1, float("nan"), float("nan"), float("nan")
    consistent = bool(np.isfinite(chi2_red) and chi2_red <= D.chi2_crit(0.95, dof) / max(dof, 1))
    k = D.t_crit(conf, max(n - 1, 1))
    returned = WeightedMean(mean=mean, u=u, chi2=chi2, dof=dof, chi2_red=chi2_red,
                            p_value=p, birge=birge, weights=w.tolist(),
                            consistent=consistent, k=k, U=k * u,
                            n_missing_u=n_missing_u)
    return returned


# --------------------------------------------------------------------------
# 离群值
# --------------------------------------------------------------------------
def grubbs_critical(n: int, alpha: float = 0.05) -> float:
    """Grubbs 单侧临界值 G(alpha, n)。"""
    if n < 3:
        return float("inf")
    t = D.t_ppf(1 - alpha / (2 * n), n - 2)
    return (n - 1) / math.sqrt(n) * math.sqrt(t * t / (n - 2 + t * t))


def outliers_grubbs(values, alpha: float = 0.05, max_iter: int = 10) -> list[dict]:
    """迭代 Grubbs 检验，返回被判为离群的点（保留原始下标）。"""
    a = np.asarray(pd.Series(list(values), dtype="float64"), dtype="float64")
    idx = list(range(a.size))
    work = list(a)
    found: list[dict] = []
    for _ in range(max_iter):
        n = len(work)
        if n < 3:
            break
        arr = np.asarray(work, dtype="float64")
        s = float(np.std(arr, ddof=1))
        if s == 0 or not np.isfinite(s):
            break
        g = np.abs(arr - arr.mean()) / s
        i = int(np.argmax(g))
        gcrit = grubbs_critical(n, alpha)
        if g[i] > gcrit:
            found.append({"index": idx[i], "value": float(arr[i]), "G": float(g[i]),
                          "G_crit": float(gcrit), "n": n,
                          "mean": float(arr.mean()), "std": s})
            work.pop(i)
            idx.pop(i)
        else:
            break
    return found


def outliers_mad(values, threshold: float = 3.5) -> list[dict]:
    """稳健离群判据：|x - median| / (1.4826*MAD) > threshold。"""
    a = np.asarray(pd.Series(list(values), dtype="float64"), dtype="float64")
    if a.size < 4:
        return []
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med))) * 1.4826
    if mad == 0:
        return []
    z = np.abs(a - med) / mad
    return [{"index": int(i), "value": float(a[i]), "z": float(z[i]),
             "threshold": threshold, "median": med, "mad": mad}
            for i in np.where(z > threshold)[0]]


# --------------------------------------------------------------------------
# 线性拟合
# --------------------------------------------------------------------------
@dataclass
class LinearFit:
    slope: float
    intercept: float
    u_slope: float
    u_intercept: float
    r2: float
    r: float
    s_resid: float
    dof: int
    chi2_red: float
    n: int
    cov: float = float("nan")
    slope_rel: float = float("nan")

    def predict(self, x) -> np.ndarray:
        return self.slope * np.asarray(x, dtype="float64") + self.intercept

    def as_dict(self) -> dict:
        return {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                for k, v in self.__dict__.items()}


def linear_fit(x, y, uy=None, conf: float = 0.95) -> LinearFit:
    """最小二乘线性拟合 y = a + b x（可给 y 的不确定度做加权拟合）。

    不确定度按经典误差传播计算（自由度 n-2）。
    """
    xv = np.asarray(pd.Series(list(x), dtype="float64"), dtype="float64")
    yv = np.asarray(pd.Series(list(y), dtype="float64"), dtype="float64")
    m = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[m], yv[m]
    n = xv.size
    if n < 3:
        raise ValueError("线性拟合至少需要 3 个有效点")
    if uy is not None:
        w = np.asarray(pd.Series(list(uy), dtype="float64"), dtype="float64")[m]
        w = np.where(np.isfinite(w) & (w > 0), 1.0 / w ** 2, np.nan)
        if not np.all(np.isfinite(w)):
            w = np.ones(n)
    else:
        w = np.ones(n)
    sw = w.sum()
    mx = float(np.sum(w * xv) / sw)
    my = float(np.sum(w * yv) / sw)
    sxx = float(np.sum(w * (xv - mx) ** 2))
    sxy = float(np.sum(w * (xv - mx) * (yv - my)))
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = yv - (slope * xv + intercept)
    dof = n - 2
    chi2 = float(np.sum(w * resid ** 2))
    # 单位权方差（若为等权，等价于残差方差）
    s2 = chi2 / dof
    u_slope = math.sqrt(s2 / sxx)
    u_intercept = math.sqrt(s2 * (1.0 / sw + mx ** 2 / sxx))
    cov = -mx * s2 / sxx
    ss_tot = float(np.sum(w * (yv - my) ** 2))
    r2 = 1 - chi2 / ss_tot if ss_tot > 0 else float("nan")
    return LinearFit(slope=float(slope), intercept=float(intercept),
                     u_slope=float(u_slope), u_intercept=float(u_intercept),
                     r2=float(r2), r=float(math.copysign(math.sqrt(abs(r2)), slope)),
                     s_resid=float(math.sqrt(s2) if s2 >= 0 else float("nan")), dof=dof,
                     chi2_red=float(chi2 / dof), n=n, cov=float(cov),
                     slope_rel=float(u_slope / abs(slope)) if slope else float("nan"))


# --------------------------------------------------------------------------
# 假设检验
# --------------------------------------------------------------------------
def t_test_1sample(values, mu0: float = 0.0, conf: float = 0.95) -> dict:
    s = stats_summary(values)
    if s.n < 2 or s.std == 0:
        return {"n": s.n, "t": float("nan"), "p": float("nan"), "dof": s.dof,
                "mean": s.mean, "mu0": mu0, "reject": False}
    t = (s.mean - mu0) / (s.std / math.sqrt(s.n))
    p = 2 * D.t_sf(abs(t), s.dof)
    return {"n": s.n, "mean": s.mean, "std": s.std, "mu0": mu0, "t": float(t),
            "p": float(p), "dof": s.dof, "t_crit": float(D.t_crit(conf, s.dof)),
            "ci": (s.mean - D.t_crit(conf, s.dof) * s.sem,
                   s.mean + D.t_crit(conf, s.dof) * s.sem),
            "reject": bool(p < 1 - conf)}


def t_test_paired(a, b, conf: float = 0.95) -> dict:
    av, bv = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
    m = np.isfinite(av) & np.isfinite(bv)
    av, bv = av[m], bv[m]
    d = av - bv
    res = t_test_1sample(d, 0.0, conf)
    res.update({"n_pairs": int(d.size), "mean_diff": float(np.mean(d)) if d.size else float("nan"),
                "std_diff": float(np.std(d, ddof=1)) if d.size > 1 else float("nan")})
    return res


def t_test_2sample(a, b, conf: float = 0.95) -> dict:
    """Welch 双样本 t 检验（不假定方差相等）。"""
    av, bv = _arr(a), _arr(b)
    n1, n2 = av.size, bv.size
    if n1 < 2 or n2 < 2:
        return {"n1": n1, "n2": n2, "t": float("nan"), "p": float("nan"), "reject": False}
    s1, s2 = float(np.var(av, ddof=1)), float(np.var(bv, ddof=1))
    se = math.sqrt(s1 / n1 + s2 / n2)
    t = (float(np.mean(av)) - float(np.mean(bv))) / se if se else float("nan")
    df = (s1 / n1 + s2 / n2) ** 2 / ((s1 / n1) ** 2 / (n1 - 1) + (s2 / n2) ** 2 / (n2 - 1)) \
        if (s1 or s2) else float("nan")
    p = 2 * D.t_sf(abs(t), df) if np.isfinite(t) else float("nan")
    return {"n1": n1, "n2": n2, "mean1": float(np.mean(av)), "mean2": float(np.mean(bv)),
            "mean_diff": float(np.mean(av)) - float(np.mean(bv)), "se": se, "t": float(t),
            "dof": float(df), "p": float(p), "reject": bool(np.isfinite(p) and p < 1 - conf)}


def anova_oneway(groups: Sequence, conf: float = 0.95) -> dict:
    """单因素方差分析：检验多组均值是否一致。"""
    arrs = [_arr(g) for g in groups]
    arrs = [a for a in arrs if a.size > 0]
    k = len(arrs)
    N = int(sum(a.size for a in arrs))
    if k < 2 or N <= k:
        return {"k": k, "N": N, "F": float("nan"), "p": float("nan"), "reject": False}
    gm = float(np.concatenate(arrs).mean())
    ssb = float(sum(a.size * (a.mean() - gm) ** 2 for a in arrs))
    ssw = float(sum(((a - a.mean()) ** 2).sum() for a in arrs))
    dfb, dfw = k - 1, N - k
    msb, msw = ssb / dfb, ssw / dfw
    F = msb / msw if msw > 0 else float("inf")
    p = D.f_sf(F, dfb, dfw) if np.isfinite(F) else 0.0
    return {"k": k, "N": N, "F": float(F), "p": float(p), "dof_between": dfb,
            "dof_within": dfw, "ms_between": msb, "ms_within": msw,
            "F_crit": float(D.f_crit(conf, dfb, dfw)), "reject": bool(p < 1 - conf)}


def bland_altman(a, b, conf: float = 0.95, loa_k: float | None = None) -> dict:
    """Bland-Altman 一致性分析：偏差、一致性界限(LoA)与超出比例。

    一致性界限 LoA = bias ± k·s_d。默认 k = 1.96（Bland & Altman 1986 的正态近似，
    与文献口径一致）；样本量小时可显式传 ``loa_k=D.t_crit(conf, n-1)`` 使用 t 分位数
    （n = 5 时界线约宽 29%），结果里始终同时给出两种口径，避免与偏倚 CI 的 t 口径混淆。

    返回的 ``u_loa`` 取 sqrt(3·s_d²/n)，即 LoA 的标准误（Bland & Altman 1986 给出的近似式）。
    """
    av, bv = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
    m = np.isfinite(av) & np.isfinite(bv)
    av, bv = av[m], bv[m]
    n = av.size
    if n < 2:
        return {"n": n}
    d = av - bv
    bias = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    t = D.t_crit(conf, n - 1)
    k = float(1.96 if loa_k is None else loa_k)
    loa_low, loa_high = bias - k * sd, bias + k * sd
    within = int(np.sum((d >= loa_low) & (d <= loa_high)))
    se_bias = sd / math.sqrt(n)
    rel = float(np.mean(np.abs(d)) / abs(np.mean(np.concatenate([av, bv])))) \
        if np.mean(np.concatenate([av, bv])) != 0 else float("nan")
    return {"n": n, "bias": bias, "sd_diff": sd, "u_bias": se_bias,
            "ci_bias": (bias - t * se_bias, bias + t * se_bias),
            "loa_k": k,
            "loa_low": float(loa_low), "loa_high": float(loa_high),
            "loa_low_t": float(bias - t * sd), "loa_high_t": float(bias + t * sd),
            "u_loa": float(math.sqrt(3 * sd ** 2 / n)),
            "n_within": within, "pct_within": 100.0 * within / n,
            "mean_abs_diff": float(np.mean(np.abs(d))), "rel_diff": rel,
            "differences": d.tolist(), "means": ((av + bv) / 2).tolist()}


def pearson(x, y) -> dict:
    xv, yv = np.asarray(x, dtype="float64"), np.asarray(y, dtype="float64")
    m = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[m], yv[m]
    n = xv.size
    if n < 3:
        return {"n": n, "r": float("nan"), "p": float("nan")}
    r = float(np.corrcoef(xv, yv)[0, 1])
    t = r * math.sqrt((n - 2) / max(1e-300, 1 - r * r))
    p = 2 * D.t_sf(abs(t), n - 2)
    return {"n": n, "r": r, "r2": r * r, "t": float(t), "p": float(p)}


def spearman(x, y) -> dict:
    xv = pd.Series(np.asarray(x, dtype="float64"))
    yv = pd.Series(np.asarray(y, dtype="float64"))
    m = xv.notna() & yv.notna()
    xv, yv = xv[m].rank(), yv[m].rank()
    return pearson(xv.to_numpy(), yv.to_numpy())
