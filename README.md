# mda —— 测量数据处理与不确定度分析工具包

把实验室/产线的原始测量记录，一步处理成**图表 + 数据结论**：

```
原始数据（CSV / TXT / Excel / JSON）
   │  ① 解析：自动识别分隔符、编码、表头、单位行、± 与括号记法、缺失值
   ▼
规整长表（dataset, variable, group, x, value, u）
   │  ② 误差与不确定度：A 类(t 区间)、B 类、合成、传播、加权平均、离群、拟合、检验
   ▼
各组统计 + 不确定度预算
   │  ③ 对齐比较：多组对齐到同一网格 → 参考值 → 偏差 / Bland-Altman / 配对检验 / 热力图
   ▼
④ 图表（PNG + SVG）＋ 结论（Markdown / HTML / JSON / Excel）
```

## 最简用法（推荐先看这个）

**完全不用记参数**，程序自己认表头、认分组列、认自变量列，然后出图表和结论。

| 你想做的事 | 怎么做 |
| --- | --- |
| 分析一个数据文件 | 把文件**拖到 [一键分析.bat](一键分析.bat) 图标上**（或 `python analyze.py 我的数据.csv`） |
| 分析一整个文件夹 | **双击 [一键分析.bat](一键分析.bat)**（或 `python analyze.py`） |
| 看示例 | `python 一键分析.py --demo` |
| 命令行等价写法 | `python 一键分析.py 数据.csv` / `python 一键分析.py D:\数据 --recursive` |

双击即用，结果会生成在**数据文件旁边的 `分析结果/<文件名>/` 文件夹**里：

```
分析结果/长度测量_长表/
  报告.html   ← 双击就能看（图表内嵌，可直接转发）
  报告.md     明细.xlsx   结果.json   charts/*.png
```

控制台会直接给出人话总结，例如：

```
[OK] 长度测量_长表.csv
     数据 120 行 | 变量 1 个 | 分组 4 组 | 模式 repeat
     自动识别: 分组列=仪器，自变量列=温度点℃
     · 长度测量值mm: 120 个数据点 | 推荐值 1000.24295 ± 0.00076（k=2 时 U=0.00152）
       | 一致性 χ²/dof=46.3 → 存在系统偏差 | 离群值 5 个（已剔除）
     图表 7 张 | 报告: ...\分析结果\长度测量_长表\报告.html
```

> 三个等价入口：`一键分析.bat`（双击/拖拽，Windows）、`一键分析.py`、`analyze.py`（纯 ASCII 文件名，便于脚本调用）。
> 只有当自动识别不准（例如列名太特殊）时，才需要用下面的高级命令显式指定列。

## 一分钟上手（高级用法）

```bash
# 1) 安装依赖（缺什么装什么；scipy 可选，未装时用内置纯 Python 数值实现）
pip install numpy pandas openpyxl pillow          # 可选: scipy

# 2) 一键演示：自动生成示例数据并跑通全流程（输出到 output/）
python run_demo.py

# 3) 分析自己的文件
python -m mda run -i 测量记录.csv --outdir output
python -m mda run -i 记录.xlsx --sheet "*" --x-col 标称值 --group-col 仪器 --value-col 示值
python -m mda inspect --input 测量记录.txt   # 只解析，先看它识别成了什么
python -m mda selftest                   # 内置自检测试（48 项）
python -m mda selftest --no-scipy        # 强制关闭 SciPy，测纯 Python 分布实现
```

输出（每个数据集一份）：

| 文件 | 内容 |
| --- | --- |
| `<数据集>_报告.md` | 全部统计表 + 自动结论（Markdown） |
| `<数据集>_报告.html` | 自包含报告，SVG 图表内嵌，可直接发同事 |
| `<数据集>_明细.xlsx` | 汇总 / 各组统计 / 对齐表 / 两两比较 / 拟合 / 派生量 |
| `<数据集>_结果.json` | 机读结果，便于二次开发 |
| `charts/*.png` | 每张图单独的 PNG |

## 输入格式与自动识别

支持 `.csv / .tsv / .txt / .dat / .log / .xlsx / .xlsm / .xls / .json`。

* **编码**：utf-8-sig → utf-8 → gb18030 → gbk → big5 → latin-1 依次尝试
* **分隔符**：逗号 / 分号 / Tab / 竖线 / 连续空白，自动嗅探
* **表头与单位行**：自动判断表头；若第二行是纯单位行（如 `mm  ℃`）会自动剥离并记录单位
* **注释行**：`#`、`//`、`%` 开头的行自动跳过（`--comment` 可改）
* **数值记法**：
  * `12.34`、`1.2e-3`、`1,234.5`、欧式小数 `12,34`（`--decimal ,`）
  * `12.34 ± 0.05`、`12.34+/-0.05` —— 值 + 不确定度
  * `12.3(4)`、`1.2345(67) mm` —— 括号记法（末位不确定度）自动展开为 `± 0.4`、`± 0.0067`
  * `23.5 °C`、`1250 mm` —— 自动剥离常见单位（长度/质量/时间/温度/电学/压力/频率…）
  * `NA -- 无 ?` 等缺失记号 → NaN 并计数
* **结构**（`--mode` 可强制）：
  * **长表**：有分组列 → 每组一列值
  * **宽表**：每个数值列一"组"（同一物理量的多次/多台测量）
  * **重复测量**：同一 (组, x) 出现多行 → 组内重复，自动算 s、u = s/√n，并在**单元内**做离群判别
* **Excel 多工作表**：`--sheet "*"` 合并所有工作表，并在解析说明里列出合并了哪些表

自动识别失败的场景可显式指定：`--group-col / --x-col / --value-col / --unc-col / --mode`。

## 误差与不确定度

| 功能 | 实现 |
| --- | --- |
| A 类不确定度 | 均值、样本标准差 s(ddof=1)、`u = s/√n`、t 分布区间 `U = t_{0.95,n-1}·u` |
| B 类不确定度 | 均匀 `a/√3`、三角 `a/√6`、正态 `a/k`、分辨力 `d/(2√3)` |
| 合成 | 方和根 `uc = √Σuᵢ²`，附各来源贡献占比 |
| 有效自由度 | Welch–Satterthwaite，`U = t_{p,νeff}·uc` |
| 传播 | 任意函数 `f(x₁…xₙ)` 数值偏导（中心差分）求灵敏度系数与合成不确定度 |
| 加权平均 | 反方差加权 + `χ²`、`χ²/dof`、Birge 比、p 值一致性判据 |
| 离群值 | 迭代 Grubbs（含临界值公式）、稳健 MAD 判据；重复测量按 `(组, x)` 单元判别 |
| 线性拟合 | 加权最小二乘，斜率/截距及其标准不确定度、R²、残差标准差、χ²/dof |
| 检验 | 单样本/配对/Welch 双样本 t 检验、单因素方差分析、Pearson/Spearman |
| 一致性 | Bland-Altman：偏差、95% 一致性界限、界限内百分比 |
| 修约 | `sig_round`/`format_measurement`：按不确定度保留 2 位有效数字并同步修约测量值 |

分位数（t、χ²、F、正态）由 `mda/dist.py` 提供：优先用 SciPy，没有 SciPy 时使用
正则化不完全 Beta / Gamma 连分式 + 二分求逆。**经与 SciPy 逐点比对（df、a ≤ 1e6）**，
各函数最大相对误差 ≤ 1e-8（cdf/sf 与不完全 Gamma 多数 ≤ 1e-10）；
`chi2/t/f_sf` 均为直接计算而非 `1 - cdf`，尾部概率不会被浮点消成 0。
装不装 SciPy 都可以用 `python -m mda selftest --no-scipy` 复现这组结论。

## 多组对齐与比较

```bash
python -m mda run -i 比对数据.csv --align nearest --align-tol 0.001 \
       --reference 仪器A --x-label "温度 / ℃"
```

* **对齐方式**：`union`（并集）/ `intersection`（交集）/ `nearest`（就近匹配 + 最小覆盖过滤）
* **参考值**：默认各网格点上的**反方差加权平均**，也可用 `--reference 组名`
* **偏差统计**：每组相对参考值的平均偏差 / RMS 偏差 / 最大绝对偏差 / 平均与最大相对偏差
* **两两比较**：配对 t 检验、Bland-Altman 偏差与一致性界限、组 A 对组 B 的回归斜率与 R²
* **一致性判据**：合并 χ²/dof、Birge 比、p 值、超差网格点数

## 图表

无需 matplotlib：内置"显示列表 + 双后端"渲染器（PNG 使用 Pillow，SVG 纯文本），
两后端图形一致，中文标签直接可用（PNG 自动挑选系统 CJK 字体）。

* 各组均值趋势 + 误差棒（含参考值）
* 相对参考值的偏差图
* 两组间平均相对偏差矩阵（热力图）
* Bland-Altman 一致性图
* 各组偏差分布箱线图（含散点）
* 全体数据直方图 + 正态曲线
* 线性拟合图（数据点 + 参考拟合线）
* 派生量不确定度构成条形图（帕累托）

`Plot` 也可以单独使用：

```python
from mda import Plot
p = Plot(title="示例", xlabel="x", ylabel="y")
p.errorbar([1, 2, 3], [1.0, 2.1, 2.9], [0.1, 0.1, 0.1], label="A")
p.line([1, 2, 3], [1, 2, 3], label="参考", dash=(5, 3))
p.render("fig.png", "png")     # 或 p.render("fig.svg", "svg") / p.svg()
```

## 配置文件（派生量与高级选项）

`--config 分析配置.json`：

```json
{
  "analysis": {
    "conf": 0.95, "k": 2.0,
    "outlier_method": "grubbs", "outlier_alpha": 0.05, "exclude_outliers": true,
    "align_method": "nearest", "fit": true
  },
  "derived": [
    {
      "name": "线膨胀系数 alpha",
      "formula": "slope / L0",
      "unit": "1/℃",
      "fit_source": "reference",
      "vars": { "L0": { "value": 1000.0, "u": 0.05, "dof": 50 } }
    }
  ]
}
```

`formula` 可用 `slope`、`intercept`（来自 `fit_source` 组的线性拟合）与 `vars` 中自定义变量，
经数值偏导做不确定度传播，输出结果 ± U、相对不确定度与主要来源。

## 目录结构

```
mda/
  parse.py        数据解析（多格式、脏数据、单位与 ± 记法 -> 长表）
  dist.py         分布与分位数（t / χ² / F / 正态；SciPy 可选）
  uncertainty.py  误差与不确定度（A/B 类、合成、传播、加权、拟合、检验）
  align.py        多组对齐与比较（对齐、参考值、偏差、Bland-Altman、两两比较）
  charts.py       图表渲染（SVG + PNG 双后端，无 matplotlib 依赖）
  analyze.py      流程编排（逐变量分析 -> 图表 + 结论文本）
  report.py       报告输出（Markdown / HTML / JSON / Excel）
  cli.py          命令行
  demo.py         示例数据生成 + 一键演示
  tests.py        内置自检（python -m mda selftest）
run_demo.py       一键演示入口
sample_data/      示例原始数据（演示时生成）
output/           分析结果（演示时生成）
```

## 在代码里调用

```python
from mda import read_dataset, AnalysisConfig, analyse_dataset, write_html, type_a, propagate

ds  = read_dataset("测量记录.csv", group_col="仪器", x_col="温度", value_cols=["长度"])
cfg = AnalysisConfig(conf=0.95, k=2, outlier_method="grubbs", outdir="out", chart_backend="both")
res = analyse_dataset(ds, cfg)
for v in res.variables:
    print(v.variable, v.comparison.overall["pooled_chi2_red"], v.conclusions[:2])
write_html(res, "out/报告.html")

# 单独用某个算法
print(type_a([10.02, 10.01, 10.03, 10.00, 10.02]).as_dict())
print(propagate(lambda a, b: a * b, {"a": (100.0, 1.0), "b": (2.0, 0.04)}).as_dict())
```

## 自检

```bash
python -m mda selftest              # 48 项
python -m mda selftest --no-scipy   # 强制走纯 Python 分布实现，同样 48 项
```

覆盖：分位数对照权威数值、脏数据解析（含表头/单位行、复合单位、列字典 JSON）、
A/B 类与传播解析解校验、加权平均、线性拟合、Grubbs、Bland-Altman、方差分析、
三种对齐方式、图表渲染、报告转义、Excel 工作表名合法化、端到端流水线（含派生量真值校验）。

测试分三部分：`mda/tests.py`（基础 22 项）、`mda/test_regressions_stats.py`（统计回归 10 项）、
`mda/test_regressions_parse.py`（解析回归 16 项）。
装了 SciPy 的机器默认只测 SciPy 分支，所以纯 Python 实现必须用 `--no-scipy` 才能测到。

## 教程

`教程/数据基础操作.py` 是一份独立的 pandas/numpy 上手示例（读写 CSV/Excel/JSON/Parquet、
清洗、聚合、透视），与主流水线无关，可单独运行。

## 说明与边界

* 不确定度评定遵循 GUM 思路：A 类取 t 区间，B 类按分布换算，合成用方和根；
  相关性/协方差尚未建模（各输入量按独立处理）。
* Grubbs 检验在 α=0.05 下每个单元约有 5% 的误报率，重复测量单元内判别尤其明显；
  多单元重复判别会放大——实测在干净数据集上"至少误标 1 个点"的概率约 60%。报告会同时给出
  判据值与 G 值，可用 `--keep-outliers` 只标记不剔除。
* 对齐默认 `nearest`，需要严格同点位比较时用 `intersection`。
* 一致性判据 `χ²/dof ≤ 2` 是工程约定（与 p 值判据实测 0 分歧），不是严格统计检验。
* Bland-Altman 的 LoA 默认 k = 1.96（文献口径），结果里同时给出 t 口径的界线；
  小样本（如 n = 5）两者相差约 29%，报告用哪个口径写得清清楚楚。
* 表头/单位行识别覆盖"表头 + 单位行 + 数据"与"标题行 + 表头 + 数据"；多层表头、
  含引号字段的 CSV 拆列、`_UNITS` 未收录的单位（如 `N/m2`、上标 `m/s²`）仍未处理。
* 在deepseek老师的帮助下完成的垃圾产物，纯新手练手用的，不要在意，要是大家能用上我也会很开心（
* 已修复缺陷与遗留问题清单见 [CHANGELOG.md](CHANGELOG.md)。

## 许可证

MIT，见 [LICENSE](LICENSE)。

