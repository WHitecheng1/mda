"""数据解析：把各种"脏"的原始记录读成规整的长表。

支持格式
--------
* 文本表格：.csv/.tsv/.txt/.dat/.log（自动识别分隔符、编码、注释行、单位行）
* Excel：.xlsx/.xlsm/.xls（可指定工作表，或 sheet="*" 读取全部）
* JSON：记录数组 / 字典 / {data: [...]} 包裹
* 混合记法：12.34、12.34+-0.05、12.34(5)、1.2e-3 mm、23.5 °C、NA/--/无

解析结果统一为长表（long format）：
    dataset | variable | group | x | value | u | unit
"""

from __future__ import annotations

import io
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "Quantity", "parse_quantity", "load_table", "read_raw",
    "coerce_columns", "infer_schema", "to_long", "read_dataset", "Schema", "Dataset",
    "json_is_tabular",
]


def json_is_tabular(obj: Any) -> bool:
    """粗略判断一个 JSON 对象是否是"表格数据"（记录数组 / 列字典）。

    用于避免把配置文件（如 mda 的分析配置 JSON）当成测量数据。
    """
    if isinstance(obj, list):
        if not obj:
            return False
        if all(isinstance(x, dict) for x in obj):
            return not any(isinstance(v, (dict, list)) for x in obj for v in x.values())
        return all(isinstance(x, (list, int, float, str, bool)) or x is None for x in obj)
    if isinstance(obj, dict):
        for key in ("data", "rows", "records", "items", "results"):
            if isinstance(obj.get(key), list):
                return json_is_tabular(obj[key])
        vals = list(obj.values())
        if not vals:
            return False
        if all(isinstance(v, (int, float, str, bool)) or v is None for v in vals):
            return len(vals) > 1
        return all(isinstance(v, list) for v in vals)
    return False

# --------------------------------------------------------------------------
# 基础：数值 / 不确定度 / 单位 的混合记法解析
# --------------------------------------------------------------------------
MISSING_TOKENS = {
    "", "-", "--", "---", "na", "n/a", "nan", "null", "none", "nil",
    "无", "缺", "缺失", "缺测", "/", "\\", "?", "待测", "未测",
}

_NUM = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
_NUM_RE = re.compile(_NUM)
_BRACKET_RE = re.compile(r"^\s*(" + _NUM + r")\s*\(\s*(\d+)\s*\)\s*(.*)$")
_PM_SPLIT_RE = re.compile(r"\s*(?:±|\+/-|\+-|\u00b1|\bpm\b|正负)\s*")

# 常见单位（长单位在前，避免 mm 抢走 mmm 之类的前缀误判）
_UNITS = [
    "m/s^2", "m/s2", "m/s", "mm/s", "rad/s", "kg/m3", "kg/m^3", "g/cm3", "g/cm^3",
    "MPa", "kPa", "GPa", "hPa", "mbar", "Pa", "bar", "N·m", "N.m", "Nm", "kN",
    "µV", "uV", "mV", "kV", "V", "µA", "uA", "mA", "kA", "A",
    "µm", "um", "nm", "pm", "mm", "cm", "dm", "km", "m",
    "µs", "us", "ns", "ms", "min", "hr", "h", "s",
    "GHz", "MHz", "kHz", "Hz", "rpm",
    "kg", "mg", "µg", "ug", "g", "t",
    "°C", "℃", "°F", "K", "°",
    "mL", "ml", "L", "uL", "µL", "cm3", "mm3", "m3",
    "dBm", "dB", "ppm", "ppb", "Ω·m", "Ω", "ohm", "kΩ", "MΩ",
    "kN·m", "N", "W", "kW", "mW", "J", "kJ", "Wh", "kWh",
    "rad", "deg", "arcsec", "′", "″", "%vol", "%",
]
# 候选单位都按“长度降序”排列，保证复合单位（m/s^2、kg/m^3）先于其前缀（m、kg）匹配
_UNIT_ALT = "|".join(re.escape(u) for u in sorted(_UNITS, key=len, reverse=True))
_UNIT_RE = re.compile("^(?:" + _UNIT_ALT + ")")          # 前缀匹配（写法松散时用）
_UNIT_FULL_RE = re.compile("^(?:" + _UNIT_ALT + ")$")    # 整词匹配：复合单位必须整体命中

_FULLWIDTH = str.maketrans({
    "＋": "+", "－": "-", "．": ".", "，": ",", "（": "(", "）": ")",
    "　": " ", "％": "%", "／": "/", "０": "0", "１": "1", "２": "2", "３": "3",
    "４": "4", "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
})


@dataclass
class Quantity:
    """一个带不确定度与单位的测量量。"""

    value: float = float("nan")
    uncertainty: float | None = None
    unit: str | None = None
    raw: str = ""

    @property
    def ok(self) -> bool:
        return not (isinstance(self.value, float) and math.isnan(self.value))

    def __str__(self) -> str:  # pragma: no cover - 展示用
        if not self.ok:
            return "NaN"
        s = f"{self.value:g}"
        if self.uncertainty is not None:
            s += f" ± {self.uncertainty:g}"
        if self.unit:
            s += f" {self.unit}"
        return s


def _clean(text: Any) -> str:
    return str(text).translate(_FULLWIDTH).strip()


def parse_quantity(text: Any, decimal: str | None = None) -> Quantity:
    """解析一个单元格：支持 12.34 / 12.34±0.05 / 12.34(5) / 1.2e-3 mm / NA。

    参数
    ----
    text    : 原始文本或数值
    decimal : 小数点符号，',' 表示欧式写法（12,34）
    """
    if text is None:
        return Quantity(raw="")
    if isinstance(text, (int, float, np.integer, np.floating)) and not isinstance(text, bool):
        v = float(text)
        return Quantity(value=v, raw=str(text))
    raw = str(text)
    s = _clean(raw)
    if decimal == ",":
        s = s.replace(".", "").replace(",", ".")
    if s.lower() in MISSING_TOKENS:
        return Quantity(raw=raw)

    # 括号式（有效位数写法）：12.34(5) -> 12.34 ± 0.05
    m = _BRACKET_RE.match(s)
    if m:
        numtxt, digits, rest = m.group(1), m.group(2), m.group(3)
        value = float(numtxt)
        mantissa = numtxt.split("e")[0].split("E")[0]
        decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
        exp = 0
        if "e" in numtxt.lower():
            exp = int(re.split("[eE]", numtxt)[1])
        u = int(digits) * (10.0 ** (exp - decimals))
        unit = _extract_unit(rest)
        return Quantity(value=value, uncertainty=u, unit=unit, raw=raw)

    # ± 写法
    parts = _PM_SPLIT_RE.split(s, maxsplit=1)
    if len(parts) == 2:
        left, right = parts
        mv = _NUM_RE.search(left)
        mu = _NUM_RE.search(right)
        if mv:
            value = float(mv.group(0))
            u = abs(float(mu.group(0))) if mu else None
            unit = _extract_unit(right.replace(mu.group(0), "") if mu else right, extra=left)
            return Quantity(value=value, uncertainty=u, unit=unit, raw=raw)
        return Quantity(raw=raw)

    # 普通数值（可能带单位、百分号、千分位）
    work = s
    if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+(\.\d+)?", work):  # 1,234.5
        work = work.replace(",", "")
    m = _NUM_RE.search(work)
    if not m:
        return Quantity(raw=raw)
    value = float(m.group(0))
    rest = work[: m.start()] + " " + work[m.end():]
    unit = _extract_unit(rest, strict=True)
    if rest.strip(" \t,;:=()[]") and unit is None:
        return Quantity(raw=raw)   # 残余文字无法识别为单位 -> 整串不是数值（如 S-01）
    return Quantity(value=value, unit=unit, raw=raw)


def _extract_unit(text: str, extra: str = "", strict: bool = False) -> str | None:
    """从数值周围的残余文本中提取单位。

    必须在剔除数字「之前」先做整词单位匹配：``m/s^2``、``kg/m3``、``m3`` 这类
    复合单位本身含数字（幂次），先跑 ``re.sub(_NUM, ...)`` 会把尾巴剪掉，只剩
    ``m``/``kg``。匹配一律按长度降序（``_UNIT_ALT``），所以 ``m/s`` 不会被 ``m``
    抢先命中。
    """
    cand = _clean(text)
    if extra:
        cand += " " + _clean(extra)
    cand = cand.replace("±", " ").replace("(", " ").replace(")", " ")

    # 1) 整串 / 逐词做「完整单位」匹配（含幂次的复合单位也能命中）
    for piece in [cand.strip(" \t,;:=")] + cand.split():
        piece = piece.strip(" \t,;:=")
        if piece and _UNIT_FULL_RE.match(piece):
            return _UNIT_FULL_RE.match(piece).group(0)  # type: ignore[union-attr]

    # 2) 再剔除数字（数值常出现在 extra 一侧），做宽松的前缀匹配
    cleaned = re.sub(_NUM, " ", cand).strip(" \t,;:=")
    if not cleaned:
        return None
    m = _UNIT_RE.match(cleaned)
    if m:
        return m.group(0)
    for tok in cleaned.split():
        m2 = _UNIT_RE.match(tok)
        if m2:
            return m2.group(0)
    if strict:
        return None
    # 纯字母短串也视为单位（如 mVrms 之类）
    tok = cleaned.split()[0]
    if re.fullmatch(r"[A-Za-zµΩ°%/]{1,10}", tok):
        return tok
    return None


def parse_series(series: pd.Series, decimal: str | None = None) -> tuple[pd.Series, pd.Series, str | None]:
    """把一列解析为 (数值, 不确定度, 单位)。"""
    vals, uncs, units = [], [], {}
    for item in series.tolist():
        q = parse_quantity(item, decimal=decimal)
        vals.append(q.value)
        uncs.append(np.nan if q.uncertainty is None else q.uncertainty)
        if q.unit:
            units[q.unit] = units.get(q.unit, 0) + 1
    unit = max(units, key=units.get) if units else None
    return pd.Series(vals, index=series.index, dtype="float64"), \
        pd.Series(uncs, index=series.index, dtype="float64"), unit


def _numeric_ratio(values: Sequence[Any]) -> float:
    """一列中可解析为数字的比例。"""
    vals = [v for v in values if _clean(v).lower() not in MISSING_TOKENS and _clean(v) != ""]
    if not vals:
        return 0.0
    good = sum(1 for v in vals if parse_quantity(v).ok)
    return good / len(vals)


# --------------------------------------------------------------------------
# 文件读取
# --------------------------------------------------------------------------
@dataclass
class RawTable:
    df: pd.DataFrame
    source: str
    sheet: str | None = None
    units: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _decode(raw: bytes, encoding: str | None = None) -> tuple[str, str]:
    encs = [encoding] if encoding else ["utf-8-sig", "utf-8", "gb18030", "gbk", "big5", "latin-1"]
    for enc in encs:
        try:
            return raw.decode(enc), enc  # type: ignore[arg-type]
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8(replace)"


def _split_line(line: str, delim: str) -> list[str]:
    if delim == "ws":
        return re.split(r"\s{2,}|\t", line.strip())
    return [c.strip().strip('"').strip("'") for c in line.split(delim)]


def _guess_delimiter(lines: list[str]) -> str:
    best, best_score = "ws", -1.0
    for d in [",", ";", "\t", "|"]:
        counts = [len(_split_line(ln, d)) for ln in lines[:50]]
        if not counts:
            continue
        modal = max(set(counts), key=counts.count)
        if modal < 2:
            continue
        score = counts.count(modal) / len(counts) + min(modal, 20) / 100.0
        if score > best_score:
            best, best_score = d, score
    return best


def _rows_from_text(text: str, delimiter: str | None, comment: str | None,
                    decimal: str | None) -> tuple[list[list[str]], list[str]]:
    notes: list[str] = []
    lines, skipped = [], 0
    for ln in text.splitlines():
        st = ln.strip()
        if not st or st in {"\ufeff"}:
            continue
        if comment and st.startswith(comment):
            skipped += 1
            continue
        lines.append(st)
    if skipped:
        notes.append(f"跳过 {skipped} 行注释")
    if not lines:
        return [], notes
    delim = delimiter or _guess_delimiter(lines)
    notes.append(f"字段分隔符: {_delim_name(delim)}")
    rows = [_split_line(ln, delim) for ln in lines]
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    return rows, notes


def _delim_name(d: str) -> str:
    return {"\t": "Tab", ",": "逗号", ";": "分号", "|": "竖线", "ws": "空白"}.get(d, repr(d))


def _looks_like_unit_row(row: Sequence[str]) -> bool:
    """判断一行是否为「纯单位行」（如 ``℃  mm``）。

    含数字/非单位文本的单元格一律判否——这正是「第二行是真实数据时不能当单位行
    吃掉」的保证。允许部分列为空（有些列本来就没写单位）；整词匹配保证 ``m/s^2``、
    ``kg/m3`` 这类复合单位能被认出，``mm 尺寸`` 之类的前缀噪音则不会误判。
    """
    if not row:
        return False
    hits = 0
    nonempty = 0
    for tok in row:
        t = _clean(tok)
        if not t:
            continue
        nonempty += 1
        if len(t) <= 8 and _UNIT_FULL_RE.match(t):
            hits += 1
            continue
        return False          # 出现数字或非单位文本 -> 不是单位行
    return nonempty > 0 and hits == nonempty


def _detect_header(rows: list[list[str]]) -> tuple[bool, int]:
    """返回 (是否有表头, 数据起始行)。

    除「非数值行 + 下一行是数值数据」这一基本判据外，还识别两种常见记录式：
    * 表头 + 纯单位行（``温度点℃,长度测量值mm`` 后跟 ``℃,mm``）——否则第二行不是
      数值，会被误判成「没有表头」，表头与单位行双双沦为数据行；
    * 表头前有标题/说明行（最多向前看 5 行，遇到数值行即停止）。
    数据起始行 = 表头行号 + 1，load_table 用 ``rows[start-1]`` 取名，用 ``rows[start:]``
    当数据。
    """
    if not rows:
        return False, 0
    n = len(rows)
    for i in range(min(n, 5)):
        r = [_clean(x) for x in rows[i]]
        if _numeric_ratio(r) >= 0.4:
            break                       # 已经进入数据区：前面没有表头
        nxt = rows[i + 1] if i + 1 < n else None
        if nxt is None:
            # 只有一行且全为非数值 -> 视为表头（无数据）
            return (True, i + 1) if i == 0 else (False, 0)
        if _numeric_ratio(nxt) > 0.6 or _looks_like_unit_row(nxt):
            return True, i + 1
    return False, 0


def load_table(path: str, *, sheet: str | int | None = None, encoding: str | None = None,
               delimiter: str | None = None, decimal: str | None = None,
               comment: str | None = "#", header: str | int | None = "auto") -> RawTable:
    """读取一个表格文件，返回未做数值化的 DataFrame。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    ext = os.path.splitext(path)[1].lower()
    notes: list[str] = []

    if ext in {".xlsx", ".xlsm", ".xls", ".xlsb"}:
        all_sheets = isinstance(sheet, str) and sheet.strip().lower() in ("*", "all")
        book = pd.read_excel(path, sheet_name=None if all_sheets else (sheet if sheet is not None else 0),
                             dtype=object)
        if isinstance(book, dict):
            frames = []
            for name, part in book.items():
                part = part.copy()
                part.insert(0, "__sheet__", name)
                frames.append(part)
            df = pd.concat(frames, ignore_index=True)
            notes.append(f"合并工作表: {', '.join(map(str, book.keys()))}")
            return RawTable(df, path, None, {}, notes)
        notes.append(f"工作表: {sheet if sheet is not None else 0}")
        return RawTable(book, path, str(sheet if sheet is not None else 0), {}, notes)

    if ext == ".json":
        with open(path, "rb") as fh:
            payload = json.loads(_decode(fh.read(), encoding)[0])
        if not json_is_tabular(payload):
            raise ValueError(f"{os.path.basename(path)} 不是表格数据（顶层是配置/嵌套对象）；"
                             f"若确实要分析，请先转成 CSV 或 JSON 记录数组")
        if isinstance(payload, dict):
            for key in ("data", "rows", "records", "items", "results"):
                if isinstance(payload.get(key), list):
                    notes.append(f"使用 JSON 键 {key}")
                    payload = payload[key]
                    break
            else:
                # 列字典 {"x": [1,2,3], "y": [4,5,6]}：每个键是一列，必须按列构造；
                # 若按记录包成 [payload]，单元格会变成 list，下游 nunique 抛
                # TypeError: unhashable type: 'list'。
                if payload and all(isinstance(v, (list, tuple)) for v in payload.values()):
                    try:
                        df = pd.DataFrame({str(k): list(v) for k, v in payload.items()})
                    except ValueError as exc:  # 各列长度不一致
                        raise ValueError(
                            f"{os.path.basename(path)} 的 JSON 列字典各列长度不一致: {exc}") from exc
                    notes.append("按列字典解析（每个键为一列）")
                    return RawTable(df, path, None, {}, notes)
                # 单条记录 {"a": 1, "b": 2} -> 1 行 N 列（而不是被当成 N 行）
                if payload:
                    notes.append("单条记录（1 行 N 列）")
                payload = [payload]
        if payload and isinstance(payload[0], dict):
            df = pd.DataFrame(payload)
        elif payload and isinstance(payload[0], (list, tuple)):
            first = payload[0]
            if all(isinstance(x, str) for x in first):
                df = pd.DataFrame(payload[1:], columns=list(first))
                notes.append("首行作为表头")
            else:
                df = pd.DataFrame(payload)
        else:
            df = pd.DataFrame({"value": payload})
        return RawTable(df, path, None, {}, notes)

    with open(path, "rb") as fh:
        text, enc = _decode(fh.read(), encoding)
    notes.append(f"编码: {enc}")
    rows, n2 = _rows_from_text(text, delimiter, comment, decimal)
    notes.extend(n2)
    if header == "auto":
        has_header, start = _detect_header(rows)
    elif header is None:
        has_header, start = False, 0
    else:
        has_header, start = True, int(header) + 1

    units: dict[str, str] = {}
    if has_header:
        # 表头行 = 数据起始行的前一行（允许前面有标题/说明行，因此不能死用 rows[0]）
        head = rows[start - 1] if 0 < start <= len(rows) else rows[0]
        cols = [_clean(c) or f"col{i}" for i, c in enumerate(head)]
        if start > 1:
            notes.append(f"表头位于第 {start} 行，前面 {start - 1} 行作为标题/说明")
        data = rows[start:]
        if data and _looks_like_unit_row(data[0]):
            units = {cols[i]: _clean(v) for i, v in enumerate(data[0]) if _clean(v)}
            data = data[1:]
            notes.append("识别到单位行并已剥离")
        notes.append("表头: " + ", ".join(cols))
    else:
        cols = [f"col{i}" for i in range(len(rows[0]) if rows else 0)]
        data = rows
        notes.append("未检测到表头，列名自动生成")

    df = pd.DataFrame(data, columns=cols)
    df = df.replace({"": np.nan})
    return RawTable(df, path, None, units, notes)


def read_raw(path: str, **kw) -> RawTable:
    """load_table 的别名。"""
    return load_table(path, **kw)


def coerce_columns(df: pd.DataFrame, *, decimal: str | None = None,
                   force: Iterable[str] | None = None,
                   min_ratio: float = 0.6) -> pd.DataFrame:
    """把"看起来是数值"的列转成 float（同时提取 ± 记法中的不确定度）。

    返回的 DataFrame 会附加属性 df.attrs['uncertainties'] = {列名: Series}。
    """
    out = df.copy()
    unc_map: dict[str, pd.Series] = {}
    force = set(force or [])
    for col in df.columns:
        if col in force:
            vals, us, _ = parse_series(df[col], decimal=decimal)
            out[col] = vals
            unc_map[col] = us
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        ratio = _numeric_ratio(df[col].tolist())
        if ratio >= min_ratio:
            vals, us, _ = parse_series(df[col], decimal=decimal)
            out[col] = vals
            if us.notna().any():
                unc_map[col] = us
    out.attrs["uncertainties"] = unc_map
    return out


# --------------------------------------------------------------------------
# 结构推断：宽表 / 长表 / 重复测量
# --------------------------------------------------------------------------
GROUP_HINTS = ["group", "组", "组别", "类别", "样品", "样品号", "试样", "仪器", "设备",
               "实验室", "机构", "人员", "操作者", "批次", "编号", "id", "name",
               "名称", "来源", "方法", "型号", "station", "lab", "operator", "batch",
               "sample", "tag", "label", "category", "mode", "状态"]
X_HINTS = ["x", "温度", "temp", "时间", "time", "date", "日期", "时刻", "频率", "freq",
           "标称", "nominal", "载荷", "load", "电压", "voltage", "电流", "current",
           "位置", "position", "序号", "index", "index_no", "点", "point", "压力",
           "pressure", "湿度", "humidity", "浓度", "concentration", "波长", "wavelength"]
UNC_HINTS = ["u", "u95", "unc", "uncertainty", "不确定度", "误差", "标准差", "sd",
             "std", "stdev", "sigma", "扩展不确定度", "标准偏差", "波动"]
VALUE_HINTS = ["value", "值", "测量值", "读数", "reading", "measurement", "meas",
               "y", "result", "结果", "数据", "data", "观测值", "输出", "output"]


def _match_hint(col: str, hints: Sequence[str]) -> bool:
    c = str(col).strip().lower()
    for h in hints:
        if c == h or c.startswith(h) or h in c:
            return True
    return False


@dataclass
class Schema:
    mode: str                      # wide | long | repeat
    value_cols: list[str]
    group_col: str | None = None
    x_col: str | None = None
    unc_cols: dict[str, str] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "value_cols": self.value_cols,
            "group_col": self.group_col,
            "x_col": self.x_col,
            "unc_cols": self.unc_cols,
            "units": self.units,
            "notes": self.notes,
        }


def infer_schema(df: pd.DataFrame, *, value_cols: Sequence[str] | None = None,
                 group_col: str | None = None, x_col: str | None = None,
                 unc_col: str | None = None, mode: str | None = None,
                 units: dict[str, str] | None = None) -> Schema:
    """在没有（或只有部分）显式指定的情况下推断数据结构。"""
    notes: list[str] = []
    units = dict(units or {})
    cols = [str(c) for c in df.columns]
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    nonnum = [c for c in cols if c not in numeric]

    # 1) 分组列
    if group_col is None:
        # 优先非数值且名字像分组；其次数值型但取值少（真正的分组列）；最后取低基数列
        cands = [c for c in nonnum if _match_hint(c, GROUP_HINTS)]
        if not cands:
            cands = [c for c in cols if _match_hint(c, GROUP_HINTS)
                     and pd.api.types.is_numeric_dtype(df[c])
                     and 1 < df[c].nunique(dropna=True) <= max(2, len(df) // 3)]
        if not cands:
            cands = [c for c in nonnum
                     if 1 < df[c].nunique(dropna=True) <= max(2, min(20, len(df) // 3))]
        if len(cands) >= 1:
            group_col = cands[0]
            notes.append(f"自动识别分组列: {group_col}")

    # 2) 不确定度列
    unc_cols: dict[str, str] = {}
    unc_cands = [c for c in cols if _match_hint(c, UNC_HINTS) and c != group_col]
    if unc_col is not None:
        unc_cands = [unc_col]

    # 3) x 轴列
    if x_col is None:
        xc = [c for c in cols if _match_hint(c, X_HINTS) and c != group_col and c not in unc_cands]
        if xc:
            x_col = xc[0]
            notes.append(f"自动识别自变量列: {x_col}")

    # 4) 数值列（排除分组列、自变量列与不确定度列）
    if value_cols:
        vcols = list(value_cols)
    else:
        vcols = [c for c in numeric if c not in (group_col, x_col) and c not in unc_cands]

    if unc_col is not None and vcols:
        unc_cols[vcols[0]] = unc_col
    elif unc_cands and vcols:
        # 依次配对：v1,u1,v2,u2...；只有一个 u 且只有一个 v 时直接配对
        if len(unc_cands) == 1 and len(vcols) == 1:
            unc_cols[vcols[0]] = unc_cands[0]
        else:
            for i, v in enumerate(vcols):
                if i < len(unc_cands):
                    unc_cols[v] = unc_cands[i]

    if not vcols:
        raise ValueError("没有找到可分析的数值列，请用 --value-col 指定")

    # 5) 模式
    if mode is None:
        if group_col is None:
            mode = "wide" if len(vcols) >= 1 else "long"
            notes.append("无分组列 -> 宽表模式（每个数值列为一组）")
        else:
            nun = df[group_col].nunique(dropna=True)
            # 同一 group 在同一 x 上重复出现 -> 重复测量
            if x_col is not None and nun > 0:
                dup = df.duplicated(subset=[group_col, x_col]).any()
                mode = "repeat" if dup else "long"
            else:
                mode = "long"
            notes.append(f"分组列 {group_col} 共 {nun} 组 -> {mode} 模式")

    for v in vcols:
        if v in units and units[v]:
            continue
        units.setdefault(v, "")
    return Schema(mode=mode, value_cols=vcols, group_col=group_col, x_col=x_col,
                  unc_cols=unc_cols, units=units, notes=notes)


@dataclass
class Dataset:
    """归一化后的数据集（长表）。"""

    name: str
    frame: pd.DataFrame            # dataset, variable, group, x, value, u, unit
    schema: Schema
    source: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def variables(self) -> list[str]:
        return [v for v in self.frame["variable"].dropna().unique().tolist()]

    @property
    def groups(self) -> list[str]:
        return self.frame["group"].dropna().unique().tolist()

    def subset(self, variable: str | None = None) -> pd.DataFrame:
        if variable is None:
            return self.frame
        return self.frame[self.frame["variable"] == variable]

    def describe(self) -> str:  # pragma: no cover - 展示用
        return (f"{self.name}: {len(self.frame)} 行, 变量={self.variables}, "
                f"组={self.groups}, 模式={self.schema.mode}")


def to_long(df: pd.DataFrame, schema: Schema, *, name: str = "data") -> Dataset:
    """把表格按 schema 归一化为长表。"""
    frames = []
    unc_map: dict[str, pd.Series] = df.attrs.get("uncertainties", {})
    x = df[schema.x_col] if schema.x_col else None
    group = df[schema.group_col] if schema.group_col else None

    for vi, vcol in enumerate(schema.value_cols):
        if vcol not in df.columns:
            raise KeyError(f"数值列不存在: {vcol}")
        ucol = schema.unc_cols.get(vcol)
        if ucol and ucol in df.columns:
            u = pd.to_numeric(df[ucol], errors="coerce")
        elif vcol in unc_map:
            u = unc_map[vcol]
        else:
            u = pd.Series(np.nan, index=df.index)
        # 宽表模式：每个数值列是一"组"，它们测的是同一个量 -> 共用一个变量名
        is_wide = group is None
        part = pd.DataFrame({
            "dataset": name,
            "variable": "" if is_wide else vcol,
            "group": vcol if group is None else group.astype(str).values,
            "x": x.values if x is not None else np.arange(len(df)),
            "value": pd.to_numeric(df[vcol], errors="coerce").values,
            "u": u.values,
        })
        frames.append(part)
    long = pd.concat(frames, ignore_index=True)
    long = long.dropna(subset=["value"]).reset_index(drop=True)
    if schema.mode == "repeat" and schema.group_col is not None and schema.x_col is None:
        long["x"] = long.groupby(["variable", "group"]).cumcount() + 1
    if schema.mode == "wide":
        long["unit"] = long["group"].map(lambda g: schema.units.get(g) or "")
    else:
        long["unit"] = long["variable"].map(lambda v: schema.units.get(v) or "")
    return Dataset(name=name, frame=long, schema=schema, notes=list(schema.notes))


def read_dataset(path: str, *, name: str | None = None, sheet=None, encoding=None,
                 delimiter=None, decimal=None, comment="#", header="auto",
                 value_cols: Sequence[str] | None = None, group_col: str | None = None,
                 x_col: str | None = None, unc_col: str | None = None,
                 mode: str | None = None) -> Dataset:
    """一步到位：读文件 -> 解析数值 -> 推断结构 -> 归一化长表。"""
    raw = load_table(path, sheet=sheet, encoding=encoding, delimiter=delimiter,
                     decimal=decimal, comment=comment, header=header)
    df = coerce_columns(raw.df, decimal=decimal)
    schema = infer_schema(df, value_cols=value_cols, group_col=group_col, x_col=x_col,
                          unc_col=unc_col, mode=mode, units=raw.units)
    ds = to_long(df, schema, name=name or os.path.splitext(os.path.basename(path))[0])
    ds.source = path
    ds.notes = list(raw.notes) + list(schema.notes)
    return ds
