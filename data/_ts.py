# -*- coding: utf-8 -*-
"""共享的 Tushare pro 客户端初始化（quicksync 中转）与 API 重试封装。

原先 pull_adj.py / build_cyq.py / build_delist_info.py / build_industry.py
各有一份 _init_pro + _retry_api，统一收敛到此。
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import tushare as ts
import tushare.pro.client as client

client.DataApi._DataApi__http_url = "http://api.quicksync.cn"

log = logging.getLogger(__name__)

_PRO = None


def init_pro():
    global _PRO
    if _PRO is not None:
        return _PRO
    from dotenv import load_dotenv

    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path, encoding="utf-8-sig")
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("TUSHARE_TOKEN not found in .env")
    ts.set_token(token)
    pro = ts.pro_api()
    pro._DataApi__http_timeout = 120
    _PRO = pro
    return pro


def retry_api(fn, *args, max_retries=3, base_wait=2, **kwargs):
    """指数退避重试。最后一次失败时抛出原始异常。"""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = base_wait * (2 ** attempt)
            log.warning("Retry %d/%d after error: %s, waiting %ds...",
                        attempt + 1, max_retries, e, wait)
            time.sleep(wait)
