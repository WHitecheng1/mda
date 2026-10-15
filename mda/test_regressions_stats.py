# -*- coding: utf-8 -*-
"""回归测试：本次修复的统计/报告缺陷（dist、uncertainty、report）。

每项测试都对应一个已被复现的真实缺陷，断言即修复的固化：

* test_gammainc_large_a        —— _gser 迭代上限写死 1000，a=5e5 时 P 误差 15.7%
* test_gamma_tail_not_zero     —— 上尾用 1-cdf 相减，尾部概率被吃成 0
* test_share_is_variance_ratio —— 不确定度占比用线性比，合计 >100%
* test_bland_altman_loa        —— LoA 固定 1.96，与 t 口径混用且未标注
* test_weighted_mean_missing_u —— 缺失 u 被静默按最大 sigma 填充
* test_excel_sheet_name_illegal—— 组名含 []:*?/\\ 时工作表被静默丢弃
* test_html_report_escaped     —— 报告 HTML 不转义，数据里的标签原样进文件
* test_render_rejects_format   —— render('x.jpg','jpg') 静默写出 PNG

运行：python -m unittest mda.test_regressions_stats -v
"""
from __future__ import annotations

import os
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

import mda
from mda import dist as D
from mda import report as R
from mda import uncertainty as U
from mda.charts import Plot


class TestDist(unittest.TestCase):
    """纯 Python 分布路径（SciPy 存在时也要强制走纯 Python 分支）。"""

    def setUp(self):
        self._saved = D._sp
        D._sp = None

    def tearDown(self):
        D._sp = self._saved

    def test_gammainc_large_a(self):
        """a ≈ 5e5 时级数需要约 5400 项；固定 1000 上限会造成 ~16% 误差。"""
        for a in (1e4, 1e5, 5e5):
            for x in (a - 1, a):
                p = D.gammainc_p(a, x)
                self.assertLess(abs(p - 0.5), 0.01,
                                f"gammainc_p({a:g},{x:g})={p!r} 明显偏离 0.5")
        # 权威参考值（SciPy 1.18）
        self.assertAlmostEqual(D.gammainc_p(500000, 499999), 0.4996238733309627, places=8)
        self.assertAlmostEqual(D.gammainc_q(500000, 499999), 0.5003761266690374, places=8)

    def test_gamma_tail_not_zero(self):
        """上尾直接计算，不再被 1-cdf 消成 0（旧实现 chi2_sf(100,1) 返回 0.0）。"""
        self.assertGreater(D.chi2_sf(100.0, 1), 0.0)
        self.assertAlmostEqual(D.chi2_sf(100.0, 1), 1.523971e-23, delta=1e-26)
        self.assertGreater(D.t_sf(100.0, 30), 0.0)
        self.assertGreater(D.f_sf(100.0, 3, 20), 0.0)

    def test_cdf_sf_complementary(self):
        """cdf 与 sf 必须互补（同一参数化下两边都算对）。"""
        for args in ((2.0, 5), (30.0, 1), (1e3, 1e4)):
            self.assertAlmostEqual(D.chi2_cdf(*args) + D.chi2_sf(*args), 1.0, places=12)
        self.assertAlmostEqual(D.t_cdf(2.5, 9) + D.t_sf(2.5, 9), 1.0, places=12)


class TestUncertainty(unittest.TestCase):

    def test_share_is_variance_ratio(self):
        """占比 = (u_i/uc)²；线性比会让合计超过 100%。"""
        p = U.propagate(lambda s, l0: s / l0, {"s": (2.0, 2.0e-3), "l0": (1.0, 1.0e-8)})
        share = sum(frac for _, _, frac in p.top())
        self.assertAlmostEqual(share, 1.0, places=12)
        name, contrib, frac = p.top()[0]
        self.assertAlmostEqual(frac, (contrib / p.uc) ** 2, places=15)
        # 线性比（旧实现）会明显偏大，这里确保没有回退
        self.assertLess(frac, abs(contrib) / p.uc)

    def test_bland_altman_loa(self):
        """LoA 用 k·s_d（默认 1.96）并同时给出 t 口径，二者都在结果里。"""
        a = [10.0, 11.0, 12.0, 13.0, 14.0]
        b = [10.5, 11.2, 11.8, 13.4, 14.1]
        r = U.bland_altman(a, b)
        self.assertEqual(r["loa_k"], 1.96)
        sd = r["sd_diff"]
        self.assertAlmostEqual(r["loa_low"], r["bias"] - 1.96 * sd, places=12)
        # n=5 时 t 口径明显更宽（约 29%）
        self.assertLess(r["loa_low_t"], r["loa_low"])
        self.assertGreater(r["loa_high_t"], r["loa_high"])
        self.assertAlmostEqual(r["u_loa"], float(np.sqrt(3 * sd ** 2 / len(a))), places=12)

    def test_weighted_mean_missing_u(self):
        """缺失 u 时给出 RuntimeWarning 并在结果里计数，而不是静默填充。"""
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            r = U.weighted_mean([1.0, 2.0, 3.0], [0.1, float("nan"), 0.1])
        self.assertEqual(r.n_missing_u, 1)
        self.assertTrue(any(issubclass(w.category, RuntimeWarning) for w in rec))
        self.assertIn("n_missing_u", r.as_dict())


class TestReport(unittest.TestCase):

    def test_excel_sheet_name_illegal(self):
        """非法字符必须被替换而不是抛错/丢表。"""
        used: set = set()
        name = R._sheet_name("A[1]:B/2*?", used)
        self.assertNotRegex(name, r"[\[\]:*?/\\]")
        self.assertLessEqual(len(name), 31)
        # 去重
        self.assertNotEqual(name, R._sheet_name("A[1]:B/2*?", used))
        # 超长名截断后仍唯一
        long1 = R._sheet_name("x" * 40, used)
        long2 = R._sheet_name("x" * 40, used)
        self.assertLessEqual(len(long1), 31)
        self.assertNotEqual(long1, long2)

    def test_md_table_escapes_pipe(self):
        """列名/单元格里的竖线必须转义，否则 Markdown 表格结构被破坏。"""
        df = pd.DataFrame({"a|b": ["x|y"], "c": [1]})
        txt = R._md_table(df)
        header = txt.splitlines()[0]
        self.assertIn(r"a\|b", header)                                  # 转义后的竖线在
        self.assertEqual(header.count("|") - header.count(r"\|"), 3)     # 2 列 → 3 个结构竖线
        self.assertIn(r"x\|y", txt)

    def test_html_escaping(self):
        """数据里的 HTML 标签不能原样进入报告（之前 <img onerror> 会落地）。"""
        df = pd.DataFrame({"组": ["<img src=x onerror=alert(1)>"]})
        out = R._html_table(df)
        self.assertNotIn("<img", out)
        self.assertIn("&lt;img", out)
        self.assertEqual(R._esc("<b>&"), "&lt;b&gt;&amp;")

    def test_render_rejects_format(self):
        """非 png/svg 的格式直接报错，不再静默写出 PNG。"""
        p = Plot(title="t")
        p.bars(["a"], [1.0])
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                p.render(os.path.join(d, "a.jpg"), fmt="jpg")
            # 正常路径仍然可用
            self.assertTrue(os.path.exists(p.render(os.path.join(d, "a.png"), fmt="png")))
            self.assertTrue(os.path.exists(p.render(os.path.join(d, "a.svg"), fmt="svg")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
