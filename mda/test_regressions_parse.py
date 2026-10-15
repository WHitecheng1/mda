"""mda.parse 的回归测试（只覆盖 parse 模块的三个已确认缺陷）。

运行方式::

    python -m unittest mda.test_regressions_parse -v

每个测试都只使用公开 API：``mda.read_dataset`` / ``mda.load_table`` /
``mda.parse_quantity``（均从包顶层或 ``mda.parse`` 导出），不依赖私有函数。
测试内容对应三处修复：

* P1  表头 + 纯单位行（含"表头前有标题行"、"只有部分列有单位"）必须被识别，
      单位行要被剥离并记录到 ``RawTable.units``；第二行是真实数据时不得被吃掉。
* P2  ``_extract_unit`` 必须整体匹配复合单位（m/s、m/s^2、kg/m3、Ω·m …），
      不能截断成前缀（m、kg、Ω）；普通单位与无量纲数值保持原行为。
* P3  JSON 列字典 ``{"x": [1,2,3], "y": [4,5,6]}`` 必须解析成 3 行 2 列，
      不得抛 ``TypeError: unhashable type: 'list'``；记录数组与单条记录保持可用。
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import mda
from mda.parse import load_table, parse_quantity, read_dataset


class _TempFiles(unittest.TestCase):
    """给每个测试一个临时目录，写文本文件后交给被测代码。"""

    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)

    def write(self, name: str, text: str) -> str:
        path = os.path.join(self._td.name, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path


# --------------------------------------------------------------------------
# P1 —— 表头 + 单位行的识别
# --------------------------------------------------------------------------
class TestHeaderWithUnitRow(_TempFiles):
    """P1: "表头行 / 单位行 / 数据行" 结构（以及其变体）必须正确解析。"""

    HEADER = "温度点℃,长度测量值mm"
    UNIT = "℃,mm"
    DATA = "20,1000.00\n25,1000.12\n30,1000.23\n"

    def test_header_then_pure_unit_row(self):
        """缺陷复现用例：修复前 rows[0]/rows[1] 被当成数据行（5 行、列名 col0/col1）。"""
        path = self.write("unit_probe.csv", f"{self.HEADER}\n{self.UNIT}\n{self.DATA}")

        raw = load_table(path)
        # 编码修复的核心断言：表头行与单位行不再进入数据
        self.assertEqual(raw.df.shape, (3, 2))
        self.assertEqual(list(raw.df.columns), ["温度点℃", "长度测量值mm"])
        # 单位行被剥离并记录为列单位
        self.assertEqual(raw.units, {"温度点℃": "℃", "长度测量值mm": "mm"})
        self.assertTrue(any("单位行" in n for n in raw.notes), raw.notes)

        ds = read_dataset(path)
        self.assertEqual(len(ds.frame), 3)
        self.assertEqual(sorted(ds.frame["unit"].unique()), ["mm"])
        self.assertEqual(ds.frame["x"].tolist(), [20.0, 25.0, 30.0])
        self.assertEqual(ds.frame["value"].tolist(), [1000.00, 1000.12, 1000.23])

    def test_header_without_unit_row_still_works(self):
        """没有单位行的普通表头文件必须保持原样（3 行数据、无单位）。"""
        path = self.write("plain.csv", f"{self.HEADER}\n{self.DATA}")
        raw = load_table(path)
        self.assertEqual(raw.df.shape, (3, 2))
        self.assertEqual(list(raw.df.columns), ["温度点℃", "长度测量值mm"])
        self.assertEqual(raw.units, {})
        self.assertEqual(len(read_dataset(path).frame), 3)

    def test_title_line_before_header(self):
        """表头前有标题/说明行时，表头行与单位行仍要被识别（而不是拿标题当表头）。"""
        path = self.write("titled.csv", f"实验记录表\n{self.HEADER}\n{self.UNIT}\n{self.DATA}")
        raw = load_table(path)
        self.assertEqual(raw.df.shape, (3, 2))
        self.assertEqual(list(raw.df.columns), ["温度点℃", "长度测量值mm"])
        self.assertEqual(raw.units.get("长度测量值mm"), "mm")
        self.assertEqual(len(read_dataset(path).frame), 3)

    def test_unit_row_with_some_columns_blank(self):
        """单位行只有部分列写了单位（其余为空）也要识别，空列不产生单位。"""
        path = self.write("partial.csv",
                          f"温度点℃,长度测量值mm,备注\n℃,mm,\n"
                          f"20,1000.00,ok\n25,1000.12,\n30,1000.23,复核\n")
        raw = load_table(path)
        self.assertEqual(raw.df.shape, (3, 3))
        self.assertEqual(list(raw.df.columns), ["温度点℃", "长度测量值mm", "备注"])
        self.assertEqual(raw.units, {"温度点℃": "℃", "长度测量值mm": "mm"})
        self.assertNotIn("备注", raw.units)
        ds = read_dataset(path)
        self.assertEqual(len(ds.frame), 3)
        self.assertEqual(sorted(ds.frame["unit"].unique()), ["mm"])

    def test_numeric_second_row_is_not_eaten_as_units(self):
        """不得过度触发：第二行（含带单位的数值单元格）是真实数据，不能被当单位行剥离。"""
        path = self.write("numeric.csv", f"{self.HEADER}\n{self.DATA}")
        raw = load_table(path)
        self.assertEqual(raw.df.shape, (3, 2))          # 3 行数据一行不少
        self.assertEqual(raw.units, {})

        path2 = self.write("numeric_unit_cells.csv",
                           "温度点,读数\n20,1000.00 mm\n25,1000.12 mm\n")
        raw2 = load_table(path2)
        self.assertEqual(raw2.df.shape, (2, 2))          # 带单位的数值行仍是数据
        self.assertEqual(raw2.units, {})
        self.assertEqual(raw2.df["读数"].tolist(), ["1000.00 mm", "1000.12 mm"])

    def test_headerless_file_unchanged(self):
        """没有表头时列名仍自动生成（不被 P1 的改动波及）。"""
        path = self.write("noheader.csv", self.DATA)
        raw = load_table(path)
        self.assertEqual(raw.df.shape, (3, 2))
        self.assertEqual(list(raw.df.columns), ["col0", "col1"])
        self.assertEqual(raw.units, {})


# --------------------------------------------------------------------------
# P2 —— 复合单位
# --------------------------------------------------------------------------
class TestCompositeUnits(unittest.TestCase):
    """P2: 复合单位必须整体匹配；修复前会被截断成前缀（m/s -> m、kg/m3 -> kg）。"""

    def test_composite_units_are_matched_whole(self):
        for text, value, unit in [
            ("1.5 m/s", 1.5, "m/s"),
            ("9.81 m/s^2", 9.81, "m/s^2"),
            ("9.81 m/s2", 9.81, "m/s2"),
            ("0.5 mm/s", 0.5, "mm/s"),
            ("12 kg/m3", 12.0, "kg/m3"),
            ("12 kg/m^3", 12.0, "kg/m^3"),
            ("3 rad/s", 3.0, "rad/s"),
            ("5 Ω·m", 5.0, "Ω·m"),
        ]:
            q = parse_quantity(text)
            self.assertAlmostEqual(q.value, value, msg=text)
            self.assertEqual(q.unit, unit, msg=text)

    def test_composite_unit_with_exponent_and_uncertainty(self):
        q = parse_quantity("1.2e-3 m/s^2")
        self.assertAlmostEqual(q.value, 0.0012, places=12)
        self.assertEqual(q.unit, "m/s^2")
        q2 = parse_quantity("12.34 ± 0.05 m/s")
        self.assertAlmostEqual(q2.value, 12.34)
        self.assertAlmostEqual(q2.uncertainty, 0.05)
        self.assertEqual(q2.unit, "m/s")

    def test_plain_units_and_plain_numbers_unchanged(self):
        self.assertEqual(parse_quantity("20 mm").unit, "mm")
        self.assertEqual(parse_quantity("23.5 °C").unit, "°C")
        self.assertEqual(mda.parse_quantity("1.2345(67) mm").unit, "mm")   # 包顶层再导出
        self.assertEqual(parse_quantity("12.34 ± 0.05 mm").unit, "mm")
        self.assertIsNone(parse_quantity("1,234.5").unit)
        self.assertIsNone(parse_quantity("1000.00").unit)
        # 非数值串仍然是 NaN（残余文字不得被误当单位）
        self.assertFalse(parse_quantity("S-01").ok)
        self.assertFalse(parse_quantity("xxx").ok)

    def test_composite_units_survive_a_column_read(self):
        """整列读取时单位行里的复合单位也要按列记录。"""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "speed.csv")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("时间s,速度\ns,m/s\n1,1.5\n2,2.5\n")
            raw = load_table(path)
            self.assertEqual(raw.units, {"时间s": "s", "速度": "m/s"})
            ds = read_dataset(path)
            self.assertEqual(sorted(ds.frame["unit"].unique()), ["m/s"])


# --------------------------------------------------------------------------
# P3 —— JSON 列字典
# --------------------------------------------------------------------------
class TestJsonColumnDict(_TempFiles):
    """P3: JSON 列字典；修复前 load_table 把每列 list 塞进一个单元格，read_dataset 崩。"""

    def write_json(self, name: str, payload) -> str:
        return self.write(name, json.dumps(payload, ensure_ascii=False))

    def test_column_dict_loads_as_table(self):
        path = self.write_json("cols.json", {"x": [1, 2, 3], "y": [4, 5, 6]})

        raw = load_table(path)                     # 修复前：列值是 list，下游崩溃
        self.assertEqual(raw.df.shape, (3, 2))
        self.assertEqual(list(raw.df.columns), ["x", "y"])
        self.assertEqual(raw.df["x"].tolist(), [1, 2, 3])
        self.assertEqual(raw.df["y"].tolist(), [4, 5, 6])
        self.assertFalse(any(isinstance(v, list) for v in raw.df.to_numpy().ravel()))

        ds = read_dataset(path)                    # 修复前：TypeError: unhashable type: 'list'
        self.assertEqual(len(ds.frame), 3)
        self.assertEqual(ds.frame["x"].tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(ds.frame["value"].tolist(), [4.0, 5.0, 6.0])

    def test_data_key_wrapped_records_still_work(self):
        path = self.write_json("wrapped.json", {"data": [{"x": 0, "v": 10}, {"x": 1, "v": 20}]})
        raw = load_table(path)
        self.assertEqual(raw.df.shape, (2, 2))
        self.assertEqual(list(raw.df.columns), ["x", "v"])
        self.assertEqual(read_dataset(path).frame["x"].tolist(), [0.0, 1.0])

    def test_nested_dict_column_dict_is_rejected_not_crashed(self):
        """边界（本次未改）：{"data": {...}} 这类嵌套对象仍按"非表格"拒绝，
        而不是被当成配置误收；报错路径不变，且不会抛出 unhashable 之类的内部异常。"""
        path = self.write_json("nested.json", {"data": {"t": [0, 1], "v": [10, 20]}})
        with self.assertRaises(ValueError):
            load_table(path)

    def test_list_of_records_still_works(self):
        path = self.write_json("records.json", [{"a": 1}, {"a": 2}])
        raw = load_table(path)
        self.assertEqual(raw.df.shape, (2, 1))
        self.assertEqual(raw.df["a"].tolist(), [1, 2])
        self.assertEqual(read_dataset(path).frame["value"].tolist(), [1.0, 2.0])

    def test_single_scalar_dict_is_one_row(self):
        """单条记录 {"a":1,"b":2} 解析为 1 行 2 列（宽表），而不是 2 行数据。"""
        path = self.write_json("scalar.json", {"a": 1, "b": 2})
        raw = load_table(path)
        self.assertEqual(raw.df.shape, (1, 2))
        self.assertEqual(list(raw.df.columns), ["a", "b"])
        self.assertEqual(raw.df.iloc[0].tolist(), [1, 2])
        ds = read_dataset(path)                    # 宽表模式：两列各是一组，共 2 个长表行
        self.assertEqual(set(ds.groups), {"a", "b"})
        self.assertEqual(len(ds.frame), 2)
        self.assertEqual(ds.frame["x"].tolist(), [0, 0])

    def test_uneven_column_dict_reports_clear_error(self):
        path = self.write_json("uneven.json", {"x": [1, 2, 3], "y": [4, 5]})
        with self.assertRaises(ValueError) as ctx:
            load_table(path)
        self.assertIn("列字典", str(ctx.exception))


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
