# -*- coding: utf-8 -*-
"""Deprecated thin wrapper: 全量构建已统一到 data.pull。

    python data/build_db.py  ==  python -m data.pull --full

⚠️ 全量拉取 2008~至今 全市场日线/复权因子/估值，约 13800 次 API、2~4 小时。
除非明确需要，日常请用增量：python -m data.pull
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.pull import run

if __name__ == "__main__":
    print("WARNING: full rebuild (2-4 hours, ~13800 API calls). "
          "Use `python -m data.pull` for daily incremental.")
    sys.exit(run(full=True))
