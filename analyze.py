"""最简单入口（ASCII 文件名版本，供一键分析.bat 调用）。

    python analyze.py                 # 分析当前文件夹里的所有数据文件
    python analyze.py 我的数据.csv     # 只分析指定文件
    python analyze.py --demo          # 生成示例数据并演示
"""

from mda.simple import main

if __name__ == "__main__":
    raise SystemExit(main())
