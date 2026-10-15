"""Python 数据读取 / 清洗 / 预处理 / 格式转换 —— 可运行的基础教程。

运行方式::

    python 教程/数据基础操作.py

脚本会在同目录下生成 示例数据/ 与 示例输出/，每一步都打印前后对比。
环境：Python 3.12 + numpy 2.3.5 + pandas 3.0.1 + openpyxl（本机实测）。
"""

from __future__ import annotations

import io
import json
import os
import re
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "示例数据")
OUT_DIR = os.path.join(HERE, "示例输出")


def sec(title: str) -> None:
    print("\n" + "=" * 78)
    print(" " + title)
    print("=" * 78)


# 一份"很脏但很真实"的原始数据：中文表头、单位后缀、千分位、全角字符、
# 时间格式不统一、缺失标记五花八门、重复行、大小写不一。
MESSY_CSV = """日期,仪器,温度(℃),读数(mm),备注
2026-10-01,仪器A,20,1000.0001 mm,
2026-10-01 ,仪器A,20℃,"1,000.0003",复核
2026-10-02,仪器A,25,1000.1185mm,
2026-10-02,仪器A,25,1000.1185mm,
2026/10/02,仪器B,25, 1000.1263 ,
2026-10-03,仪器B,30,NA,探头未接
2026-10-03,仪器B,30,--,
2026-10-04,仪器 C ,30,1000.2312,偶发
2026-10-04,仪器C,35,1000.3489,
2026-10-05,仪器C,35,1000.3489,重复录入
2026-10-05,仪器D,40,１０００.４９０２,全角数字
"""


def make_example_files() -> dict:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = {}
    p = os.path.join(DATA_DIR, "原始记录.csv")
    with open(p, "w", encoding="utf-8-sig", newline="") as fh:
        fh.write(MESSY_CSV)
    paths["csv"] = p

    # 宽表（每列一台仪器）
    wide = pd.DataFrame({
        "温度℃": [20, 25, 30, 35],
        "仪器A": [1000.0002, 1000.1185, 1000.2339, 1000.3525],
        "仪器B": [1000.0091, 1000.1263, 1000.2428, 1000.3577],
        "仪器C": [999.9956, 1000.1132, 1000.2317, 1000.3480],
    })
    paths["wide"] = os.path.join(DATA_DIR, "宽表.xlsx")
    wide.to_excel(paths["wide"], index=False, sheet_name="宽度")

    # 长表（同一 (仪器,温度) 有多次重复测量）
    rng = np.random.default_rng(0)
    rows = []
    for inst, base in (("仪器A", 1000.0), ("仪器B", 1000.008)):
        for t in (20, 25):
            for k in range(3):
                rows.append({"仪器": inst, "温度℃": t, "读数mm": round(base + t * 0.0235 + rng.normal(0, 0.002), 5)})
    paths["long"] = os.path.join(DATA_DIR, "长表.csv")
    pd.DataFrame(rows).to_csv(paths["long"], index=False, encoding="utf-8-sig")
    return paths


# ==========================================================================
def demo_read(paths: dict) -> None:
    sec("1. 读取：从文件到 DataFrame")

    # 1.1 CSV：编码 + 让缺失标记统一 + 不做类型猜测
    df = pd.read_csv(paths["csv"], encoding="utf-8-sig", dtype=str)
    print("[1.1] 原样读入（dtype=str，什么都不猜）")
    print(df.to_string(index=False))

    # 1.2 只用几列 / 只读前几行 / 分块读大文件
    head = pd.read_csv(paths["csv"], encoding="utf-8-sig", usecols=["仪器", "读数(mm)"], nrows=3)
    print("\n[1.2] usecols + nrows：", head.shape, list(head.columns))
    total = 0
    for chunk in pd.read_csv(paths["csv"], encoding="utf-8-sig", chunksize=4):
        total += len(chunk)
    print("      chunksize=4 分块累加行数：", total)

    # 1.3 从字符串/内存读（爬虫、接口返回常用）
    df_mem = pd.read_csv(io.StringIO(MESSY_CSV))
    print("[1.3] 从字符串读入：", df_mem.shape)

    # 1.4 Excel：单表 / 全部表 / 指定表
    book = pd.read_excel(paths["wide"], sheet_name=None)      # 返回 {表名: DataFrame}
    print("[1.4] Excel 全部工作表：", list(book.keys()), "| 表形状", book["宽度"].shape)

    # 1.5 JSON：记录数组 / 列字典
    records = [{"仪器": "A", "值": 1.0}, {"仪器": "B", "值": 2.0}]
    print("[1.5] JSON 记录数组：\n", pd.DataFrame(records).to_string(index=False))

    # 1.6 分隔符/表头/注释各不相同的文本，用参数显式说明
    raw = "仪器  温度  读数\nA  20  1.0001\nA  20  1.0003\n"
    d = pd.read_csv(io.StringIO(raw), sep=r"\s+", engine="python")
    print("[1.6] 空白分隔文本：", d.shape, list(d.columns))


# ==========================================================================
def demo_inspect(df: pd.DataFrame) -> None:
    sec("2. 体检：先看清数据长什么样，再动手")
    print("[2.1] info()：类型与缺失一目了然")
    df.info()
    print("\n[2.2] 每列缺失与取值数")
    rep = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "缺失数": df.isna().sum(),
        "缺失率%": (df.isna().mean() * 100).round(1),
        "唯一值": df.nunique(dropna=True),
        "样例": [df[c].dropna().iloc[0] if df[c].notna().any() else None for c in df.columns],
    })
    print(rep.to_string())


# ==========================================================================
def demo_clean(paths: dict) -> pd.DataFrame:
    sec("3. 清洗：列名 / 缺失 / 类型 / 文本 / 重复 / 异常")
    df = pd.read_csv(paths["csv"], encoding="utf-8-sig", dtype=str)

    # 3.1 列名规范化：去空白、统一括号、加可读后缀
    df.columns = [re.sub(r"\s+", "", str(c)) for c in df.columns]
    df = df.rename(columns={"读数(mm)": "读数_mm", "温度(℃)": "温度_C"})
    print("[3.1] 规范列名：", list(df.columns))

    # 3.2 缺失值：先统一各种"缺失写法"，再统计
    MISSING = {"", "-", "--", "na", "n/a", "nan", "null", "无", "缺测", "待测"}
    df = df.map(lambda v: np.nan if isinstance(v, str) and v.strip().lower() in MISSING else v)
    print("[3.2] 统一缺失标记后每列缺失数：", df.isna().sum().to_dict())

    # 3.3 文本清洗：去空格、全角转半角、去千分位、剥离单位
    #     注意：单位剥离必须"分列做"。如果对仪器列也套用温度单位的正则，
    #     "仪器C" 会被当成以 C 结尾而削成 "仪器" —— 这是真实会犯的错。
    def clean_base(v):
        if not isinstance(v, str):
            return v
        s = v.strip().translate(str.maketrans("０１２３４５６７８９．", "0123456789."))
        return s.replace(",", "")

    def strip_tail(v, pattern):
        return re.sub(pattern, "", v).strip() if isinstance(v, str) else v

    df["仪器"] = df["仪器"].map(clean_base).map(lambda s: re.sub(r"\s+", "", s) if isinstance(s, str) else s)
    df["温度_C"] = df["温度_C"].map(clean_base).map(lambda s: strip_tail(s, r"\s*(℃|°C|C)\s*$"))
    df["读数_mm"] = df["读数_mm"].map(clean_base).map(lambda s: strip_tail(s, r"\s*(mm|MM|毫米)\s*$"))
    df = df.replace({"": np.nan})
    print("[3.3] 文本清洗后样例：")
    print(df.head(4).to_string(index=False))

    # 3.4 类型转换：数字用 to_numeric(errors="coerce")，日期用 to_datetime
    df["读数_mm"] = pd.to_numeric(df["读数_mm"], errors="coerce")   # 转不了的变 NaN，不炸
    df["温度_C"] = pd.to_numeric(df["温度_C"], errors="coerce")
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce", format="mixed")  # 混合格式必须 format="mixed"
    print("[3.4] dtypes：\n", df.dtypes.to_string())

    # 3.5 重复行：整行完全一致 → 删；同一业务键但数值不同 → 是重复测量，绝不能删
    before = len(df)
    df = df.drop_duplicates()
    print(f"[3.5] drop_duplicates（整行完全一致才删）：{before} -> {len(df)} 行")
    key_dup = df.duplicated(subset=["仪器", "温度_C"], keep=False)
    if key_dup.any():
        print(f"      同一 (仪器, 温度) 出现多行的有 {int(key_dup.sum())} 条 —— 这是**重复测量**，"
              f"正确做法是聚合而不是删除：")
        print(df[key_dup].groupby(["仪器", "温度_C"])["读数_mm"]
                .agg(["count", "mean", "std"]).round(5).to_string())

    # 3.6 分类字段归一：去空白后统一写法
    df["仪器"] = df["仪器"].str.replace(r"\s+", "", regex=True)
    print("[3.6] 仪器取值：", sorted(df["仪器"].dropna().unique().tolist()))

    # 3.7 异常值：IQR 箱线判据（稳健）与 z-score（要求近似正态）
    v = df["读数_mm"].dropna()
    q1, q3 = v.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    flag_iqr = (df["读数_mm"] < lo) | (df["读数_mm"] > hi)
    z = (v - v.mean()) / v.std(ddof=1)
    print(f"[3.7] IQR 界限 [{lo:.4f}, {hi:.4f}] -> 异常 {int(flag_iqr.sum())} 个；"
          f"|z|>3 的 {int((z.abs() > 3).sum())} 个")

    # 3.8 缺失值的三种处理选择（这里只演示，不真删）
    print("[3.8] 缺失处理示例：")
    print("      前向填充 :", df["读数_mm"].ffill().tolist()[:5])
    print("      插值     :", df["读数_mm"].interpolate().round(4).tolist()[:5])
    print("      中位数填充:", round(df["读数_mm"].median(), 4))
    return df


# ==========================================================================
def demo_preprocess(clean: pd.DataFrame, paths: dict) -> None:
    sec("4. 预处理：分组 / 宽长互转 / 合并 / 对齐 / 分箱 / 标准化")

    # 4.1 分组聚合：一次算出多个统计量
    g = clean.groupby(["仪器", "温度_C"])["读数_mm"].agg(
        次数="count", 均值="mean", 标准差="std", 最小值="min", 最大值="max")
    print("[4.1] groupby.agg 结果：")
    print(g.round(5).to_string())

    # 4.2 宽表 -> 长表（melt）与长表 -> 宽表（pivot_table）
    wide = pd.read_excel(paths["wide"], sheet_name="宽度")
    long = wide.melt(id_vars="温度℃", var_name="仪器", value_name="读数_mm")
    print("\n[4.2] melt 前", wide.shape, "-> melt 后", long.shape)
    back = long.pivot_table(index="温度℃", columns="仪器", values="读数_mm")
    print("      pivot_table 还原：", back.shape)

    # 4.3 合并：横向 merge 与纵向 concat
    meta = pd.DataFrame({"仪器": ["仪器A", "仪器B", "仪器C"], "实验室": ["一室", "二室", "三室"]})
    merged = long.merge(meta, on="仪器", how="left")
    print("[4.3] merge 后列：", list(merged.columns))

    # 4.4 按最近时间/最近点对齐（merge_asof）：多组测量对齐的关键工具
    a = pd.DataFrame({"t": [20.00, 25.00, 30.00], "A": [1.0, 2.0, 3.0]}).sort_values("t")
    b = pd.DataFrame({"t": [20.01, 24.98, 30.20], "B": [1.1, 2.2, 3.3]}).sort_values("t")
    aligned = pd.merge_asof(a, b, on="t", direction="nearest", tolerance=0.05)
    print("[4.4] merge_asof（容差 0.05，最近邻对齐）：")
    print(aligned.to_string(index=False))
    print("      30.20 与 30.00 相差 0.2 > 容差 0.05，所以 B 对齐失败留空 —— 容差决定了'多近才算同一个点'")

    # 4.5 分箱与分位数分组
    clean = clean.copy()
    clean["温度段"] = pd.cut(clean["温度_C"], bins=[0, 22, 32, 50], labels=["低", "中", "高"])
    clean["读数等级"] = pd.qcut(clean["读数_mm"], q=3, labels=["小", "中", "大"], duplicates="drop")
    print("[4.5] 分箱结果：")
    print(clean[["温度_C", "温度段", "读数_mm", "读数等级"]].dropna().to_string(index=False))

    # 4.6 标准化：z-score / min-max / 稳健标准化（中位数 + MAD）
    x = clean["读数_mm"].dropna()
    z = (x - x.mean()) / x.std(ddof=1)
    mm = (x - x.min()) / (x.max() - x.min())
    med, mad = x.median(), (x - x.median()).abs().median()
    robust = (x - med) / (1.4826 * mad)
    print(f"[4.6] z-score 均值/标准差 {z.mean():.2e}/{z.std(ddof=1):.3f}；"
          f"min-max 范围 [{mm.min():.2f}, {mm.max():.2f}]；稳健标准化范围 "
          f"[{robust.min():.2f}, {robust.max():.2f}]")

    # 4.7 时间序列：重采样与滚动统计
    ts = pd.DataFrame({"t": pd.date_range("2026-10-01", periods=10, freq="h"),
                       "值": np.arange(10, dtype="float64")})
    ts = ts.set_index("t")
    print("[4.7] 2 小时均值重采样：", ts.resample("2h")["值"].mean().tolist())
    print("      3 点滑动均值    ：", ts["值"].rolling(3).mean().round(2).tolist()[:6])


# ==========================================================================
def demo_convert(clean: pd.DataFrame, paths: dict) -> None:
    sec("5. 格式转换与输出：CSV / Excel / JSON / Markdown")
    os.makedirs(OUT_DIR, exist_ok=True)

    # 5.1 CSV：中文给 Excel 看要用 utf-8-sig（否则 Excel 打开是乱码）
    out_csv = os.path.join(OUT_DIR, "清洗后.csv")
    clean.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 5.2 Excel：一个文件多个工作表 + 格式化数字
    out_xlsx = os.path.join(OUT_DIR, "清洗后.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        clean.to_excel(w, sheet_name="明细", index=False)
        clean.groupby("仪器")["读数_mm"].agg(["count", "mean", "std"]).round(6) \
             .to_excel(w, sheet_name="按仪器汇总")

    # 5.3 JSON：常用 orient；日期一定要给 date_format="iso"，
    #     否则 pandas 会把时间写成"毫秒时间戳"（3.0 起还会给 Pandas4Warning）
    out_json = os.path.join(OUT_DIR, "清洗后.json")
    clean.head(3).to_json(out_json, orient="records", force_ascii=False, indent=2,
                          date_format="iso")
    print("[5.3] to_json(orient='records')：")
    print(json.dumps(json.load(open(out_json, encoding="utf-8")), ensure_ascii=False, indent=2)[:320] + " ...")

    # 5.4 Markdown 表格（可直接贴进文档/PR）——需要可选依赖 tabulate
    try:
        # 注意：Markdown 表格默认会四舍五入，关键数值要显式给 floatfmt
        md = clean.head(3).to_markdown(index=False, floatfmt=".5f")
        print("[5.4] to_markdown：\n" + md)
    except ImportError:
        print("[5.4] to_markdown 需要 tabulate：pip install tabulate；这里用简易降级输出：")
        cols = list(clean.columns)
        print("| " + " | ".join(cols) + " |")
        print("| " + " | ".join("---" for _ in cols) + " |")
        for _, r in clean.head(3).iterrows():
            print("| " + " | ".join(str(r[c]) for c in cols) + " |")

    # 5.5 其它常用出口
    print("[5.5] to_dict('records') 第一条：", clean.head(1).to_dict("records")[0])
    print("      to_numpy() 形状：", clean.select_dtypes("number").to_numpy().shape)

    # 5.6 Parquet / Feather：列式存储，比 CSV 小且保留类型（需要 pyarrow）
    try:
        import pyarrow  # noqa: F401
        out_pq = os.path.join(OUT_DIR, "清洗后.parquet")
        clean.to_parquet(out_pq, index=False)
        back = pd.read_parquet(out_pq)
        print(f"[5.6] Parquet 往返：{back.shape}，dtypes 保持 {dict(back.dtypes.astype(str))}")
    except ImportError:
        print("[5.6] 未安装 pyarrow，跳过 Parquet（pip install pyarrow 后可 to_parquet/read_parquet）")
    for f in (out_csv, out_xlsx, out_json):
        print(f"      已写出 {os.path.relpath(f, HERE)}（{os.path.getsize(f)} 字节）")


# ==========================================================================
def demo_pandas3() -> None:
    sec("6. pandas 3.0 的四个变化（照抄老教程会踩坑）")
    # 6.1 字符串默认 dtype 是 str，不再是 object
    s = pd.Series(["a", "b", None])
    print(f"[6.1] pd.Series(['a','b',None]).dtype = {s.dtype}（旧版是 object）")

    # 6.2 元素级映射：DataFrame.applymap 已移除，用 DataFrame.map
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    print("[6.2] df.map(...) 可用，applymap 已删除：", hasattr(df, "map"), hasattr(df, "applymap"))

    # 6.3 fillna(method=) 已移除，改成 ffill()/bfill()
    try:
        pd.Series([1.0, None]).fillna(method="ffill")
    except TypeError as e:
        print("[6.3] fillna(method='ffill') 报错 ->", "改用 .ffill() / .bfill()")

    # 6.4 Copy-on-Write 生效：链式赋值会静默失效（并给出 ChainedAssignmentError 告警）
    d = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        d[d["a"] > 1]["b"] = 99                     # 链式赋值：看似改了，其实没改
        kinds = [x.category.__name__ for x in w]
    print(f"[6.4] 链式赋值后 b = {d['b'].tolist()}（没生效），告警 {kinds}")
    d.loc[d["a"] > 1, "b"] = 99                     # 正确写法：用 .loc 定位
    print(f"      d.loc[d['a'] > 1, 'b'] = 99 之后 b = {d['b'].tolist()}")


# ==========================================================================
def robust_read(path: str, **kw) -> pd.DataFrame:
    """一个"什么文件都能读"的健壮封装（mda/parse.py 的精简版思路）。"""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(path, sheet_name=kw.pop("sheet", 0), dtype=object)
    if ext == ".json":
        return pd.DataFrame(json.load(open(path, encoding="utf-8")))
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, **kw)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法解码：{path}")


def main() -> int:
    paths = make_example_files()
    print("示例数据已生成：", {k: os.path.relpath(v, HERE) for k, v in paths.items()})

    demo_read(paths)
    df_raw = pd.read_csv(paths["csv"], encoding="utf-8-sig", dtype=str)
    demo_inspect(df_raw)
    clean = demo_clean(paths)
    demo_preprocess(clean, paths)
    demo_convert(clean, paths)
    demo_pandas3()

    sec("7. 健壮读取封装（可直接抄走）")
    print(robust_read(paths["csv"], dtype=str).head(3).to_string(index=False))
    print("\n教程结束。输出目录：", os.path.relpath(OUT_DIR, HERE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
