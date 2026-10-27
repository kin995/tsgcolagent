"""coverage 启动器。

绿色版 Python 的扩展模块（unicodedata 等）在解释器目录下。`coverage run`
执行脚本时会改写 sys.path，导致这些模块找不到；这里在导入 pytest 前把
解释器目录补回去。
"""
from __future__ import annotations

import os
import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    exe_dir = os.path.dirname(sys.executable)
    if exe_dir and exe_dir not in sys.path:
        sys.path.insert(0, exe_dir)
    import pytest

    return int(pytest.main(args))


if __name__ == "__main__":
    raise SystemExit(main())
