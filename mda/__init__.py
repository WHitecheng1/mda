"""mda —— 测量数据处理与不确定度分析工具包。

一条流水线：原始数据解析 -> 误差/不确定度评定 -> 多组对齐比较 -> 图表与结论报告。

快速上手::

    from mda import read_dataset, AnalysisConfig, analyse_dataset, write_html

    ds  = read_dataset("测量记录.csv")              # 自动识别分组列/自变量列/数值列
    cfg = AnalysisConfig(outdir="out", chart_backend="both")
    res = analyse_dataset(ds, cfg)                  # 统计 + 不确定度 + 对齐比较 + 图表
    write_html(res, "out/报告.html")                # 自包含 HTML（SVG 内嵌）
    write_markdown(res, "out/报告.md")              # Markdown 结论
    write_excel(res, "out/明细.xlsx")               # 逐点/逐组明细
"""

from .align import Alignment, Comparison, align_groups, compare_alignment, resample_time
from .analyze import (AnalysisConfig, AnalysisResult, GroupResult, VariableResult,
                      analyse_dataset, analyse_variable)
from .charts import Plot
from .dist import has_scipy
from .parse import (Dataset, Quantity, Schema, coerce_columns, infer_schema, load_table,
                    parse_quantity, read_dataset, to_long)
from .report import write_excel, write_html, write_json, write_markdown
from .uncertainty import (LinearFit, Propagation, Summary, TypeAResult, WeightedMean,
                          anova_oneway, bland_altman, combine_uncertainties,
                          expanded_uncertainty, format_measurement, linear_fit,
                          outliers_grubbs, outliers_mad, pearson, propagate, spearman,
                          stats_summary, t_test_1sample, t_test_2sample, t_test_paired,
                          type_a, type_b, welch_satterthwaite, weighted_mean)

__version__ = "1.0.0"

__all__ = [
    "read_dataset", "load_table", "parse_quantity", "coerce_columns", "infer_schema",
    "to_long", "Dataset", "Quantity", "Schema",
    "type_a", "type_b", "propagate", "combine_uncertainties", "expanded_uncertainty",
    "welch_satterthwaite", "weighted_mean", "linear_fit", "outliers_grubbs", "outliers_mad",
    "t_test_1sample", "t_test_paired", "t_test_2sample", "anova_oneway", "bland_altman",
    "pearson", "spearman", "format_measurement", "stats_summary",
    "Summary", "TypeAResult", "WeightedMean", "LinearFit", "Propagation",
    "align_groups", "compare_alignment", "resample_time", "Alignment", "Comparison",
    "AnalysisConfig", "analyse_dataset", "analyse_variable", "AnalysisResult",
    "VariableResult", "GroupResult", "Plot",
    "write_markdown", "write_json", "write_html", "write_excel", "has_scipy",
]
