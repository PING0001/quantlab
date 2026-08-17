# -*- coding: utf-8 -*-
"""进程互斥锁：防止 pull / factors.update 并发写同一 DuckDB。

基于 fcntl.flock（独占）。崩溃后残留的锁通过锁文件内记录的 acquired
时间戳判定：超过 STALE_AFTER（2 小时，覆盖 19:30 启动 + cyq 延迟重试至
21:00 的最长路径）视为陈旧，自动抢占并告警。
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger(__name__)

STALE_AFTER_S = 2 * 60 * 60


def _read_acquired(path: Path) -> float | None:
    try:
        data = json.loads(path.read_text() or "{}")
        return float(data.get("acquired", 0)) or None
    except Exception:
        return None


@contextmanager
def file_lock(lock_path: Path, name: str = "job"):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)

    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            acquired = _read_acquired(lock_path)
            if acquired and time.time() - acquired > STALE_AFTER_S:
                log.warning("[%s lock] stale lock (held since %.0f min ago), stealing",
                            name, (time.time() - acquired) / 60)
                os.close(fd)
                fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                held = f"{(time.time() - acquired) / 60:.0f} min" if acquired else "unknown"
                raise RuntimeError(
                    f"[{name} lock] another process holds {lock_path} (held {held}); "
                    f"if it is dead, remove the file manually"
                ) from None

        os.ftruncate(fd, 0)
        os.write(fd, json.dumps(
            {"pid": os.getpid(), "acquired": int(time.time())}).encode())
        try:
            yield
        finally:
            os.ftruncate(fd, 0)
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
