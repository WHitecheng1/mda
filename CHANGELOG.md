# 修复记录 / Changelog

本文件记录代码走查中发现并修复的真实缺陷。每条都给出**缺陷 → 证据 → 修复 → 回归测试**，
不做没有验证的声明。

---

## v1.1.0 —— 缺陷修复（本次）

### 1. 表头 + 单位行的表**完全不识别表头**（数据被污染）

* **现象**：最常见的实验室记录格式
  ```
  温度点℃,长度测量值mm
  ℃,mm
  20,1000.00
  ```
  被判定为"无表头"，列名退化成 `col0/col1`，**且表头行与单位行都被当成数据行**。
* **根因**：`parse.py` 的 `_detect_header` 要求第二行数值占比 > 0.6，而纯单位行占比为 0，
  于是判为无表头，后面"剥离单位行"的分支永远走不到。
* **修复**：识别"表头 + 单位行 + 数据"这一形态，剥离单位行并按列记录单位。
* **回归**：`mda/test_regressions_parse.py`；`python -m mda inspect --input <文件>`
  现在输出 `识别到单位行并已剥离`，数据行数正确。

### 2. 复合单位被静默截断

* **现象**：`1.5 m/s` → 单位 `m`；`9.81 m/s^2` → 单位 `m`（数值对、单位错）。
* **根因**：`_extract_unit` 未锚定到词尾，`m` 先于 `m/s` 命中。
* **修复**：按最长匹配 + 词尾锚定；补齐 `m/s`、`m/s^2`、`kg/m3`、`rad/s` 等。
* **回归**：`test_regressions_parse.py`。

### 3. `gammainc_p` 在大参数下误差达 16%

* **现象**：`gammainc_p(500000, 499999)` 相对误差 **15.7%**（下游 `chi2_ppf(df=1e6)` 误差 9e-6）。
* **根因**：`_gser` 的迭代上限**写死 1000**；a ≈ 5e5 时级数实际需要约 **5390** 项，被提前截断。
* **修复**：迭代上限随 a 自适应（`max(1000, 20√a + 200)`），未收敛时用连分式兜底。
* **效果**：`gammainc_p/q` 与 SciPy 的最大相对误差从 1.6e-01 降到 **6.8e-10**。
* **回归**：`test_regressions_stats.py::test_gammainc_large_a`。

### 4. 上尾概率被 `1 - cdf` 消成 0

* **现象**：`chi2_sf(100, 1)` 返回 `0.0`（真值 1.5e-23），`t_sf(100, 30)` 同样为 0。
* **根因**：`*_sf` 用 `1 - cdf` 相减，cdf 已饱和到 1。
* **修复**：`chi2_sf` 走新写的 `gammainc_q`，`t_sf` / `f_sf` 用不完全 Beta 的对称恒等式直接算。
* **回归**：`test_gamma_tail_not_zero`、`test_cdf_sf_complementary`。

### 5. 不确定度预算的"占比%"用了线性比

* **现象**：报告里 `slope 99.9988% + L0 0.4934% = **100.49%**`——占比之和大于 100%。
* **根因**：`report.py` 用 `u_i / uc`，而合成不确定度是**方和根**，份额应为 `(u_i/uc)²`；
  `uncertainty.py` 的 `Propagation.top()` 同样问题。
* **修复**：两处都改为方差份额。
* **效果**：同一份报告现在是 `99.9976% + 0.00243% = 100.0000%`。
* **回归**：`test_share_is_variance_ratio`。

### 6. Bland-Altman 的 LoA 与 t 口径混用且未标注

* **现象**：LoA 固定 1.96，同一函数里偏倚 CI 却用 t；n = 5 时 LoA 比 t 口径**窄 29.4%**，
  可能把不一致判成一致。
* **修复**：`loa_k` 显式化（默认仍为文献口径 1.96，可传 `D.t_crit(conf, n-1)`），
  结果里**同时**给出 `loa_low/high`、`loa_low_t/high_t` 与 `loa_k`；
  `u_loa = √(3s²/n)` 的出处写进 docstring。
* **回归**：`test_bland_altman_loa`。

### 7. 缺失的不确定度被静默填充

* **现象**：`sigmas` 里有 NaN 时，用**最大 sigma**（≈ 最小权重）填充，不提示；
  极端输入（σ = [0.1, 1e-300, 0.1]）会得到 `mean = nan` 只带一条 RuntimeWarning。
* **修复**：给出 `RuntimeWarning`，并在 `WeightedMean.n_missing_u` 里计数、进入 `as_dict()`。
* **回归**：`test_weighted_mean_missing_u`。

### 8. 非法工作表名导致 Excel 静默丢表

* **现象**：组名含 `[]:*?/\` 时该工作表**消失且无任何提示**。
* **根因**：只做了 `[:31]` 截断，未做字符合法化与重名去重。
* **修复**：新增 `_sheet_name()`：替换非法字符 → 截断 31 → 冲突时加 `~2` 后缀。
* **回归**：`test_excel_sheet_name_illegal`。

### 9. 报告 HTML 不转义（注入）

* **现象**：组名/列名/数据集名来自用户数据文件，`<img src=x onerror=...>` 会**原样写进
  HTML 与 Markdown**；而报告的宣传语是"可直接发同事"。
* **修复**：HTML 全路径 `html.escape`（含表格头、单元格、标题、结论、图表名）；
  Markdown 表格单元格转义竖线与换行。
* **回归**：`test_html_escaping`、`test_md_table_escapes_pipe`。

### 10. 其他

| 问题 | 修复 |
| --- | --- |
| `render('a.jpg', 'jpg')` 静默写出真 PNG | 非 png/svg 直接 `ValueError`（`test_render_rejects_format`） |
| `analyze.py` 三处 `except Exception: pass`（拟合/对齐失败后该组凭空消失） | 收集失败原因 → `RuntimeWarning` + 写入结果 `summary` |
| `emit()` 先写 `%TEMP%` 再重算一次 SVG（垃圾文件 + 无用功） | 直接 `plot.svg()` |
| `uncertainty.py` 文档字符串 `WELM-Satterthwaite` 拼写错误 | 改为 Welch–Satterthwaite |
| `report.py` 提示"完整数据见 CSV/Excel"，但从不输出 CSV | 改为"同名 Excel 明细" |
| 自动化测试只覆盖 SciPy 分支（装了 SciPy 就测不到纯 Python 实现） | 新增 `python -m mda selftest --no-scipy`，两种模式都在 CI/自检里跑 |

---

## 已知边界（未修复，属设计取舍，已写入 README）

* 不确定度评定按 GUM 独立输入处理，**未建模相关性/协方差**。
* Grubbs 检验 α = 0.05 时每个单元约 5% 误报；多单元重复判别会放大
  （实测干净数据集至少有 1 个点被误标的概率约 60%），默认会剔除，可用 `--keep-outliers` 只标记。
* 一致性判据 `χ²/dof ≤ 2` 是工程约定（与 p 值判据实测 0 分歧），不是严格统计检验。
* 单位行识别覆盖"表头 + 单位行 + 数据"，标题行在表头之前的情况也支持；
  更复杂的多层表头未处理。
