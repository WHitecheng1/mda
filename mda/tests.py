"""内置自检测试：python -m mda selftest。"""

from __future__ import annotations

import math
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from . import align as A
from . import charts as C
from . import dist as D
from . import report as R
from . import uncertainty as U
from .analyze import AnalysisConfig, analyse_dataset
from .parse import coerce_columns, infer_schema, load_table, parse_quantity, read_dataset, to_long


class TestDist(unittest.TestCase):
    def test_quantiles(self):
        self.assertAlmostEqual(D.norm_ppf(0.975), 1.959963984540054, places=8)
        self.assertAlmostEqual(D.t_ppf(0.975, 10), 2.2281388519649385, places=6)
        self.assertAlmostEqual(D.t_ppf(0.995, 30), 2.749995653182527, places=6)
        self.assertAlmostEqual(D.chi2_ppf(0.95, 1), 3.841458820694124, places=6)
        self.assertAlmostEqual(D.f_crit(0.95, 3, 20), 3.0983912121404056, places=5)

    def test_cdf(self):
        self.assertAlmostEqual(D.t_cdf(2.0, 5), 0.9490302605850709, places=8)
        self.assertAlmostEqual(D.chi2_cdf(5.0, 3), 0.8282028557032669, places=8)
        self.assertAlmostEqual(D.norm_cdf(1.0), 0.8413447460685429, places=10)


class TestParse(unittest.TestCase):
    def test_parse_quantity(self):
        self.assertAlmostEqual(parse_quantity("12.34").value, 12.34)
        q = parse_quantity("12.34 ± 0.05")
        self.assertAlmostEqual(q.value, 12.34)
        self.assertAlmostEqual(q.uncertainty, 0.05)
        q = parse_quantity("12.3(4)")
        self.assertAlmostEqual(q.value, 12.3)
        self.assertAlmostEqual(q.uncertainty, 0.4)
        q = parse_quantity("123(4)")
        self.assertAlmostEqual(q.uncertainty, 4.0)
        q = parse_quantity("1.2345(67) mm")
        self.assertAlmostEqual(q.value, 1.2345, places=6)
        self.assertAlmostEqual(q.uncertainty, 0.0067, places=6)
        self.assertEqual(q.unit, "mm")
        self.assertAlmostEqual(parse_quantity("1.2e-3").value, 0.0012)
        self.assertAlmostEqual(parse_quantity("23.5 °C").value, 23.5)
        self.assertEqual(parse_quantity("23.5 °C").unit, "°C")
        self.assertFalse(parse_quantity("NA").ok)
        self.assertFalse(parse_quantity("--").ok)
        self.assertAlmostEqual(parse_quantity("1,234.5").value, 1234.5)

    def test_load_text_table(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "t.txt")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("# 注释行\n仪器  温度  读数\nA  20  1.001\nA  20  1.003\nB  20  --\n")
            raw = load_table(p)
            self.assertEqual(raw.df.shape[0], 3)
            self.assertIn("仪器", list(raw.df.columns))

    def test_schema_wide_and_long(self):
        df = pd.DataFrame({"温度": [1, 2, 3], "A": [1.0, 2.0, 3.0], "B": [1.1, 2.1, 3.1]})
        sch = infer_schema(df)
        self.assertEqual(sch.mode, "wide")
        self.assertEqual(sch.x_col, "温度")
        self.assertEqual(sorted(sch.value_cols), ["A", "B"])
        ds = to_long(df, sch, name="t")
        self.assertEqual(set(ds.groups), {"A", "B"})

        df2 = pd.DataFrame({"仪器": ["A", "A", "B", "B"], "温度": [1, 1, 1, 1], "值": [1.0, 1.2, 1.1, 1.3]})
        sch2 = infer_schema(df2)
        self.assertEqual(sch2.group_col, "仪器")
        self.assertEqual(sch2.mode, "repeat")


class TestUncertainty(unittest.TestCase):
    def test_type_a(self):
        vals = [10.02, 10.01, 10.03, 10.00, 10.02]
        res = U.type_a(vals, 0.95)
        self.assertEqual(res.summary.n, 5)
        self.assertAlmostEqual(res.summary.mean, 10.016, places=6)
        self.assertAlmostEqual(res.summary.std, float(np.std(vals, ddof=1)), places=10)
        self.assertAlmostEqual(res.u, res.summary.std / math.sqrt(5), places=10)
        self.assertAlmostEqual(res.k, D.t_crit(0.95, 4), places=10)

    def test_type_b(self):
        self.assertAlmostEqual(U.type_b("uniform", 0.03), 0.03 / math.sqrt(3), places=12)
        self.assertAlmostEqual(U.type_b("三角", 0.03), 0.03 / math.sqrt(6), places=12)
        self.assertAlmostEqual(U.type_b("normal", 0.06, k=2), 0.03, places=12)
        self.assertAlmostEqual(U.type_b("分辨力", 0.01), 0.01 / (2 * math.sqrt(3)), places=12)

    def test_propagate_analytic(self):
        # y = a*b，相对不确定度应为 sqrt((ua/a)^2+(ub/b)^2)
        prop = U.propagate(lambda a, b: a * b, {"a": (100.0, 1.0), "b": (2.0, 0.04)})
        self.assertAlmostEqual(prop.value, 200.0, places=10)
        expect = 200.0 * math.sqrt((1 / 100) ** 2 + (0.04 / 2) ** 2)
        self.assertAlmostEqual(prop.uc, expect, places=8)
        self.assertAlmostEqual(prop.contributions["a"], 2.0, places=8)

    def test_weighted_mean(self):
        wm = U.weighted_mean([10.0, 10.2, 10.1], [0.1, 0.2, 0.1])
        self.assertTrue(9.9 < wm.mean < 10.2)
        self.assertGreater(wm.chi2, 0)
        self.assertAlmostEqual(wm.chi2_red, wm.chi2 / wm.dof, places=12)
        wm2 = U.weighted_mean([1.0, 1.0, 1.0], [0.01, 0.01, 0.01])
        self.assertAlmostEqual(wm2.mean, 1.0, places=12)
        self.assertAlmostEqual(wm2.u, 0.01 / math.sqrt(3), places=12)

    def test_linear_fit_exact(self):
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = 3.0 * x + 2.0
        f = U.linear_fit(x, y)
        self.assertAlmostEqual(f.slope, 3.0, places=9)
        self.assertAlmostEqual(f.intercept, 2.0, places=9)
        self.assertAlmostEqual(f.r2, 1.0, places=9)
        self.assertLess(f.s_resid, 1e-9)

    def test_linear_fit_uncertainty(self):
        rng = np.random.default_rng(7)
        x = np.linspace(0, 10, 40)
        y = 1.5 * x + 0.5 + rng.normal(0, 0.1, x.size)
        f = U.linear_fit(x, y)
        self.assertAlmostEqual(f.slope, 1.5, delta=0.05)
        self.assertGreater(f.u_slope, 0)
        self.assertLess(f.u_slope, 0.05)

    def test_outliers_grubbs(self):
        vals = [10.0, 10.1, 9.9, 10.05, 10.02, 20.0]
        found = U.outliers_grubbs(vals, 0.05)
        self.assertEqual(len(found), 1)
        self.assertAlmostEqual(found[0]["value"], 20.0)

    def test_bland_altman(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = a + 0.1
        ba = U.bland_altman(a, b)
        self.assertAlmostEqual(ba["bias"], -0.1, places=9)
        self.assertAlmostEqual(ba["pct_within"], 100.0, places=6)

    def test_anova(self):
        g1 = np.array([1.0, 1.1, 0.9, 1.0])
        g2 = np.array([1.0, 1.05, 0.95, 1.02])
        g3 = np.array([5.0, 5.1, 4.9, 5.0])
        res = U.anova_oneway([g1, g2, g3])
        self.assertTrue(res["reject"])
        res2 = U.anova_oneway([g1, g2])
        self.assertFalse(res2["reject"])

    def test_format_measurement(self):
        self.assertEqual(U.format_measurement(12.3456, 0.0123, "mm"), "12.346 ± 0.012 mm")
        self.assertEqual(U.format_measurement(1234.5, 12.0), "1234 ± 12")


class TestAlign(unittest.TestCase):
    def _long(self):
        rows = []
        for g, off in (("A", 0.0), ("B", 0.5), ("C", 1.0)):
            for x in (1.0, 2.0, 3.0):
                rows.append({"group": g, "x": x, "value": x * 10 + off, "u": 0.1})
        return pd.DataFrame(rows)

    def test_align_union(self):
        al = A.align_groups(self._long(), method="union", tol=1e-9)
        self.assertEqual(len(al.mean), 3)
        self.assertEqual(list(al.mean.columns), ["A", "B", "C"])
        self.assertAlmostEqual(al.mean.loc[1.0, "B"], 10.5, places=9)

    def test_align_intersection_nearest(self):
        df = self._long()
        df = df[~((df["group"] == "C") & (df["x"] == 2.0))]
        al = A.align_groups(df, method="intersection", tol=1e-9)
        self.assertEqual(len(al.mean), 2)
        al2 = A.align_groups(df, method="nearest", tol=1e-9, min_coverage=2)
        self.assertEqual(len(al2.mean), 3)
        self.assertTrue(np.isnan(al2.mean.loc[2.0, "C"]))

    def test_compare(self):
        al = A.align_groups(self._long(), method="union", tol=1e-9)
        cmp = A.compare_alignment(al)
        self.assertIn("A", cmp.group_stats.index)
        self.assertAlmostEqual(cmp.deviations.loc[1.0, "B"], 0.5 - cmp.reference.loc[1.0] + 0.0,
                               places=6) if False else None
        self.assertAlmostEqual(cmp.deviations.loc[1.0, "B"],
                               al.mean.loc[1.0, "B"] - cmp.reference.loc[1.0], places=9)
        self.assertEqual(len(cmp.pairwise), 3)

    def test_reference_group(self):
        al = A.align_groups(self._long(), method="union", tol=1e-9)
        cmp = A.compare_alignment(al, reference="A")
        self.assertAlmostEqual(cmp.group_stats.loc["B", "mean_dev"], 0.5, places=9)


class TestCharts(unittest.TestCase):
    def test_render(self):
        with tempfile.TemporaryDirectory() as td:
            p = C.Plot(title="测试图", xlabel="x", ylabel="y")
            x = np.linspace(0, 10, 11)
            p.errorbar(x, np.sin(x), np.full_like(x, 0.1), label="sin", marker="o")
            p.line(x, np.cos(x), label="cos", dash=(5, 3))
            p.hline(0.5, label="阈值")
            png = os.path.join(td, "a.png")
            svg = os.path.join(td, "a.svg")
            p.render(png, "png")
            p.render(svg, "svg")
            self.assertTrue(os.path.getsize(png) > 2000)
            self.assertIn("<svg", open(svg, encoding="utf-8").read())
            p2 = C.Plot(title="柱状", width=600, height=400)
            p2.bars(["A", "B", "C"], [1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
            p2.render(os.path.join(td, "b.png"), "png")
            p3 = C.Plot(title="热力图")
            p3.heatmap(np.array([[0.0, 1.0], [2.0, 3.0]]), ["r1", "r2"], ["c1", "c2"])
            p3.render(os.path.join(td, "c.png"), "png")
            self.assertTrue(os.path.getsize(os.path.join(td, "c.png")) > 1000)


class TestEndToEnd(unittest.TestCase):
    def test_pipeline(self):
        from .demo import generate_sample_data
        with tempfile.TemporaryDirectory() as td:
            paths = generate_sample_data(td)
            for key in ("long_csv", "wide_csv", "messy_txt", "xlsx"):
                self.assertTrue(os.path.exists(paths[key]))
            ds = read_dataset(paths["long_csv"])
            self.assertGreater(len(ds.frame), 100)
            self.assertGreaterEqual(len(ds.groups), 3)
            cfg = AnalysisConfig(outdir=os.path.join(td, "out"), chart_backend="both")
            cfg.derived = [{"name": "alpha", "formula": "slope / L0", "unit": "1/℃",
                            "fit_source": "reference",
                            "vars": {"L0": {"value": 1000.0, "u": 0.05, "dof": 50}}}]
            res = analyse_dataset(ds, cfg)
            self.assertEqual(len(res.variables), 1)
            v = res.variables[0]
            self.assertIsNotNone(v.comparison)
            self.assertTrue(v.chart_files)
            # 真值 alpha ≈ 23.5e-6，检查派生量在合理范围
            self.assertAlmostEqual(v.derived[0]["value"], 23.5e-6, delta=2.0e-6)
            md = R.write_markdown(res, os.path.join(td, "r.md"))
            js = R.write_json(res, os.path.join(td, "r.json"))
            ht = R.write_html(res, os.path.join(td, "r.html"))
            xl = R.write_excel(res, os.path.join(td, "r.xlsx"))
            for f in (md, js, ht, xl):
                self.assertTrue(os.path.getsize(f) > 500, f)
            self.assertGreater(len(res.variables[0].conclusions), 3)

    def test_wide_and_messy(self):
        from .demo import generate_sample_data
        with tempfile.TemporaryDirectory() as td:
            paths = generate_sample_data(td)
            a = read_dataset(paths["wide_csv"])
            self.assertEqual(len(a.groups), 4)
            b = read_dataset(paths["messy_txt"])
            self.assertGreaterEqual(len(b.groups), 3)
            xl = read_dataset(paths["xlsx"], sheet="*")
            self.assertGreater(len(xl.groups), 2)


def run_selftest(no_scipy: bool = False, extra_modules: tuple[str, ...] = (
        "mda.test_regressions_stats", "mda.test_regressions_parse")) -> int:
    """运行内置自检。

    no_scipy=True 时强制关闭 SciPy，用纯 Python 分布实现跑一遍——
    否则装了 SciPy 的机器永远测不到 dist.py 的兜底路径（README 声称的能力）。
    extra_modules 里存在的回归测试模块会被一起加载（缺失则静默跳过，便于单独分发）。
    """
    import importlib

    if no_scipy:
        from . import dist as _dist
        _dist._sp = None
        print("[selftest] 已强制关闭 SciPy，测试纯 Python 分布实现")

    loader = unittest.defaultTestLoader
    suite = loader.loadTestsFromModule(__import__(__name__, fromlist=["x"]))
    for name in extra_modules:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        suite.addTests(loader.loadTestsFromModule(mod))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run_selftest())
