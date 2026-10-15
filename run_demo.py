"""一键演示：生成示例原始数据 -> 跑通解析/不确定度/对齐比较/图表报告。

用法::

    python run_demo.py            # 输出到 output/
    python -m mda demo            # 等价写法（支持更多参数）

生成的数据位于 sample_data/，报告与图表位于 output/。
"""

from mda.demo import run_demo

if __name__ == "__main__":
    raise SystemExit(run_demo())
