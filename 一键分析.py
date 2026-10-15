"""最简单入口：双击本文件或运行 python 一键分析.py 即可。

    python 一键分析.py                    # 分析当前文件夹里的所有数据文件
    python 一键分析.py 我的数据.csv        # 只分析指定文件
    python 一键分析.py --demo             # 生成示例数据并演示
"""

from mda.simple import main

if __name__ == "__main__":
    raise SystemExit(main())
