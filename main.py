#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
兼容入口：保留 root `main.py` 方便本地直接运行。

Smithery 部署/运行请使用：
    biai_server.server:create_server
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    """让 `python main.py` 在未安装包的情况下也能 import src/ 下的包。"""
    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def main() -> None:
    _ensure_src_on_path()

    # 这里延迟导入，避免环境变量/依赖缺失时 import 立即失败
    from biai_server.server import create_server

    server = create_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()