# -*- coding: utf-8 -*-
"""One-off: 给 factor_values 加 PRIMARY KEY (code, date)，根治重复行风险。

- date 保持 VARCHAR（'YYYY-MM-DD'，字典序即时间序；下游 7 个消费方零改动）
- 保留 ai_gz2000_* 列与其数据（列所有权归 build_ai_factor.py，本脚本不动值）
- 全程事务：任一步失败 ROLLBACK，旧表无损
- 成功后旧表保留为 factor_values_bak_<date>，人工确认数日后可 DROP
- 若存在重复 (code, date)，重复行先备份到 parquet 再去重（keep 任意一份，
  同键多行应内容一致；不一致会在日志中列出供人工核查）

用法：
    python data/migrate_fv_pk.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DB_PATH

BACKUP_PATH = (DB_PATH.parent / "factor_values_dupes_backup.parquet").as_posix()


def main():
    con = duckdb.connect(str(DB_PATH))
    try:
        total_before = con.execute("SELECT COUNT(*) FROM factor_values").fetchone()[0]
        dup_groups = con.execute(
            "SELECT COUNT(*) FROM (SELECT code, date FROM factor_values "
            "GROUP BY code, date HAVING COUNT(*) > 1)"
        ).fetchone()[0]

        if dup_groups > 0:
            excess = con.execute(
                "SELECT COALESCE(SUM(c - 1), 0) FROM (SELECT COUNT(*) c FROM factor_values "
                "GROUP BY code, date HAVING COUNT(*) > 1)"
            ).fetchone()[0]
            print(f"发现 {dup_groups} 组重复 (code, date)，共 {excess} 条多余行，备份到 {BACKUP_PATH}")
            con.execute(
                f"COPY (SELECT t.*, COUNT(*) OVER (PARTITION BY code, date) AS dup_n "
                f"FROM factor_values t) TO '{BACKUP_PATH}' (FORMAT PARQUET)"
            )
            # 内容一致性检查：同键不同内容的组（去重 keep last 仍可能选错）
            inconsistent = con.execute(
                "SELECT COUNT(*) FROM ("
                "  SELECT code, date FROM factor_values GROUP BY code, date"
                "  HAVING COUNT(DISTINCT (alpha1, alpha2, close_price)) > 1)"
            ).fetchone()[0] if "alpha2" in {r[0] for r in con.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name='factor_values'"
            ).fetchall()} else 0
            if inconsistent:
                print(f"警告：{inconsistent} 组重复键内容不一致，请人工核查备份文件")
        else:
            print("无重复 (code, date)，直接加 PK")

        con.execute("BEGIN TRANSACTION")
        con.execute(
            "CREATE TABLE __fv_pk AS "
            "SELECT DISTINCT ON (code, date) * FROM factor_values"
        )
        con.execute("ALTER TABLE __fv_pk ADD PRIMARY KEY (code, date)")

        total_new = con.execute("SELECT COUNT(*) FROM __fv_pk").fetchone()[0]
        expected = total_before - (con.execute(
            "SELECT COALESCE(SUM(c - 1), 0) FROM (SELECT COUNT(*) c FROM factor_values "
            "GROUP BY code, date)"
        ).fetchone()[0] if dup_groups > 0 else 0)
        assert total_new == expected, f"行数校验失败: {total_new} != {expected}"

        bak = f"factor_values_bak_{date.today():%Y%m%d}"
        con.execute(f"DROP TABLE IF EXISTS {bak}")
        con.execute(f"ALTER TABLE factor_values RENAME TO {bak}")
        con.execute("ALTER TABLE __fv_pk RENAME TO factor_values")
        con.execute("COMMIT")
        con.execute("CHECKPOINT")

        # 后验：PK 存在、ai 列数据完整
        pk = con.execute(
            "SELECT COUNT(*) FROM information_schema.table_constraints "
            "WHERE table_name='factor_values' AND constraint_type='PRIMARY KEY'"
        ).fetchone()[0]
        ai = con.execute(
            "SELECT COUNT(*) FROM factor_values WHERE ai_gz2000_20d IS NOT NULL"
        ).fetchone()[0]
        print(f"迁移完成: {total_before} -> {total_new} 行, PK={pk == 1}, "
              f"ai_gz2000_20d 非空={ai}, 备份表={bak}")
    except Exception:
        try:
            con.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        con.close()


if __name__ == "__main__":
    main()
