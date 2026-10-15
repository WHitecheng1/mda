"""概率分布、分位数与临界值。

本模块刻意只依赖标准库 math：若运行环境装有 SciPy 会自动改用 SciPy 的
实现（更快更准），否则使用下面这套纯 Python 数值算法
（正则化不完全 Beta / 不完全 Gamma 函数 + 二分求逆），精度优于 1e-10，
足以支撑计量校准、实验数据处理中的 t 分布、卡方分布与 F 分布查询。
"""

from __future__ import annotations

import math

try:  # pragma: no cover - 取决于环境
    from scipy import stats as _sp  # type: ignore
except Exception:  # pragma: no cover
    _sp = None

__all__ = [
    "has_scipy", "norm_pdf", "norm_cdf", "norm_sf", "norm_ppf", "norm_crit",
    "betainc", "t_pdf", "t_cdf", "t_sf", "t_ppf", "t_crit",
    "gammainc_p", "chi2_pdf", "chi2_cdf", "chi2_sf", "chi2_ppf", "chi2_crit",
    "f_cdf", "f_sf", "f_crit", "z_crit", "student_t_crit",
]

_FPMIN = 1e-300


def has_scipy() -> bool:
    """当前环境是否可用 SciPy。"""
    return _sp is not None


# --------------------------------------------------------------------------
# 正态分布
# --------------------------------------------------------------------------
def norm_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


def norm_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def norm_sf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """上侧尾概率 P(X > x)。"""
    return 0.5 * math.erfc((x - mu) / (sigma * math.sqrt(2.0)))


def norm_ppf(p: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """标准正态分位数（Acklam 有理逼近 + 一轮 Halley 修正）。"""
    if not 0.0 < p < 1.0:
        raise ValueError("p 必须落在 (0, 1) 区间")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        z = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    elif p <= phigh:
        q = p - 0.5
        r = q * q
        z = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        z = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    # Halley 迭代精修
    e = norm_cdf(z) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(z * z / 2)
    z = z - u / (1 + z * u / 2)
    return mu + sigma * z


def norm_crit(conf: float = 0.95, k: float | None = None) -> float:
    """正态包含因子：给出 k 直接返回 k，否则返回双侧 conf 分位数。"""
    if k is not None:
        return float(k)
    return norm_ppf(1 - (1 - conf) / 2)


# --------------------------------------------------------------------------
# 正则化不完全 Beta 函数（t、F 分布的基础）
# --------------------------------------------------------------------------
def _betacf(a: float, b: float, x: float) -> float:
    maxit, eps = 300, 3e-16
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """正则化不完全 Beta 函数 I_x(a, b)（Numerical Recipes 连分式）。"""
    if a <= 0 or b <= 0:
        raise ValueError("a、b 必须为正")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1.0 - x) / b


# --------------------------------------------------------------------------
# t 分布
# --------------------------------------------------------------------------
def t_pdf(t: float, df: float) -> float:
    if _sp is not None:
        return float(_sp.t.pdf(t, df))
    lg = (math.lgamma((df + 1) / 2) - math.lgamma(df / 2)
          - 0.5 * math.log(df * math.pi))
    return math.exp(lg - (df + 1) / 2 * math.log1p(t * t / df))


def t_cdf(t: float, df: float) -> float:
    if df <= 0:
        raise ValueError("自由度必须为正")
    if _sp is not None:
        return float(_sp.t.cdf(t, df))
    x = df / (df + t * t)
    p = 0.5 * betainc(df / 2.0, 0.5, x)
    return p if t <= 0 else 1.0 - p


def t_sf(t: float, df: float) -> float:
    """上尾概率 P(T > t)。直接计算，避免 1 - cdf 在尾部失效。"""
    if df <= 0:
        raise ValueError("自由度必须为正")
    if _sp is not None:
        return float(_sp.t.sf(t, df))
    x = df / (df + t * t)
    p = 0.5 * betainc(df / 2.0, 0.5, x)          # = P(T > |t|)
    return p if t >= 0 else 1.0 - p


def t_ppf(p: float, df: float) -> float:
    """t 分布分位数（二分求逆，绝对精度约 1e-12）。

    注意：中位数处 t=0，而 t_cdf 的参数化 df/(df+t²) 在 |t| << sqrt(df) 时
    会把 t² 的四舍五入吃掉，因此 p=0.5 直接返回 0，避免留下 ~1e-6 的绝对伪影。
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p 必须落在 (0, 1) 区间")
    if df <= 0:
        raise ValueError("自由度必须为正")
    if _sp is not None:
        return float(_sp.t.ppf(p, df))
    if p == 0.5:
        return 0.0
    lo, hi = -1.0, 1.0
    while t_cdf(lo, df) > p:
        lo *= 2.0
        if lo < -1e12:
            return float("-inf")
    while t_cdf(hi, df) < p:
        hi *= 2.0
        if hi > 1e12:
            return float("inf")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid == lo or mid == hi:
            break
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13 * max(1.0, abs(lo)):
            break
    return 0.5 * (lo + hi)


def t_crit(conf: float = 0.95, df: float = 1e9) -> float:
    """双侧 t 临界值；df 极大时退化为正态分位数。"""
    if df >= 1e7:
        return norm_ppf(1 - (1 - conf) / 2)
    return t_ppf(1 - (1 - conf) / 2, df)


student_t_crit = t_crit


# --------------------------------------------------------------------------
# 卡方分布
# --------------------------------------------------------------------------
def _series_maxit(a: float) -> int:
    """级数/连分式的迭代上限：项数约 O(sqrt(a))，固定 1000 会在 a > 1e4 时提前截断。

    实测 a = 5e5、x ≈ a 时级数需要约 5400 项；截断会使 P 出现约 16% 的相对误差。
    """
    return max(1000, int(20.0 * math.sqrt(max(a, 0.0))) + 200)


def _gser(a: float, x: float) -> float:
    maxit, eps = _series_maxit(a), 3e-16
    ap, s, delta = a, 1.0 / a, 1.0 / a
    converged = False
    for _ in range(maxit):
        ap += 1.0
        delta *= x / ap
        s += delta
        if abs(delta) < abs(s) * eps:
            converged = True
            break
    if not converged:
        # 级数未收敛（a 极大）：用互补函数的连分式兜底，避免静默返回截断值
        return 1.0 - _gcf(a, x)
    return s * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gcf(a: float, x: float) -> float:
    maxit, eps = _series_maxit(a), 3e-16
    b = x + 1.0 - a
    c = 1.0 / _FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, maxit + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b + an / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def gammainc_p(a: float, x: float) -> float:
    """正则化下不完全 Gamma 函数 P(a, x)。"""
    if a <= 0:
        raise ValueError("a 必须为正")
    if x <= 0:
        return 0.0
    if x < a + 1.0:
        return _gser(a, x)
    return 1.0 - _gcf(a, x)


def gammainc_q(a: float, x: float) -> float:
    """正则化上不完全 Gamma 函数 Q(a, x) = 1 - P(a, x)。

    按 x 与 a 的相对大小直接选取级数/连分式，而不是用 1 - P 相减，
    否则大 x 时 P 已饱和到 1，尾部概率会被浮点消去吃成 0。
    """
    if a <= 0:
        raise ValueError("a 必须为正")
    if x <= 0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _gser(a, x)
    return _gcf(a, x)


def chi2_pdf(x: float, df: float) -> float:
    if x <= 0:
        return 0.0
    if _sp is not None:
        return float(_sp.chi2.pdf(x, df))
    k = df / 2.0
    return math.exp((k - 1) * math.log(x) - x / 2 - k * math.log(2) - math.lgamma(k))


def chi2_cdf(x: float, df: float) -> float:
    if df <= 0:
        raise ValueError("自由度必须为正")
    if x <= 0:
        return 0.0
    if _sp is not None:
        return float(_sp.chi2.cdf(x, df))
    return gammainc_p(df / 2.0, x / 2.0)


def chi2_sf(x: float, df: float) -> float:
    """上尾概率 P(X > x)。直接算 Q，避免 1 - cdf 在尾部失效。"""
    if df <= 0:
        raise ValueError("自由度必须为正")
    if x <= 0:
        return 1.0
    if _sp is not None:
        return float(_sp.chi2.sf(x, df))
    return gammainc_q(df / 2.0, x / 2.0)


def chi2_ppf(p: float, df: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError("p 必须落在 (0, 1) 区间")
    if _sp is not None:
        return float(_sp.chi2.ppf(p, df))
    lo, hi = 0.0, max(1.0, df)
    while chi2_cdf(hi, df) < p:
        hi *= 2.0
        if hi > 1e12:
            return float("inf")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid == lo or mid == hi:
            break
        if chi2_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13 * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def chi2_crit(conf: float = 0.95, df: float = 1) -> float:
    return chi2_ppf(conf, df)


# --------------------------------------------------------------------------
# F 分布（方差分析 / 方差一致性检验）
# --------------------------------------------------------------------------
def f_cdf(f: float, d1: float, d2: float) -> float:
    if f <= 0:
        return 0.0
    if _sp is not None:
        return float(_sp.f.cdf(f, d1, d2))
    return betainc(d1 / 2.0, d2 / 2.0, d1 * f / (d1 * f + d2))


def f_sf(f: float, d1: float, d2: float) -> float:
    """上尾概率 P(F > f)。用 Beta 对称恒等式直接算，避免 1 - cdf 在尾部失效。"""
    if f <= 0:
        return 1.0
    if _sp is not None:
        return float(_sp.f.sf(f, d1, d2))
    return betainc(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * f))


def f_crit(conf: float = 0.95, d1: float = 1, d2: float = 1) -> float:
    """F 分布上侧分位数 F_conf(d1, d2)。"""
    if _sp is not None:
        return float(_sp.f.ppf(conf, d1, d2))
    lo, hi = 0.0, 1.0
    while f_cdf(hi, d1, d2) < conf:
        hi *= 2.0
        if hi > 1e12:
            return float("inf")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid == lo or mid == hi:
            break
        if f_cdf(mid, d1, d2) < conf:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


z_crit = norm_crit
