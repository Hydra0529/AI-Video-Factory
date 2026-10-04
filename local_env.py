"""读取项目目录下的 .env，且不覆盖已经存在的环境变量。"""
from __future__ import annotations

import os
from pathlib import Path

_LOADED = False


def load_local_env() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    path = Path(__file__).with_name(".env")
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
