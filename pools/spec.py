# -*- coding: utf-8 -*-
"""池注册表单源（2026-08-27 多池化重写）：PoolSpec 是池身份的唯一事实。

每池一等公民字段：
    name          池名（CLI --pool / env QUANTLAB_POOL 的取值）
    snap_table    时点快照表（结构 (effective_date, cutoff_date, code)）
    factor_table  横截面因子表（横截面 rank 参考系按池隔离，绝不可共表）
    band          流通市值带（亿元 @2026，通胀衰减见 membership；None=无带）
    data_since    数据加载锚（原 run_lgb.HISTORY_SINCE 语义：union_codes
                  的 since 参数 + 训练数据起点配套——= 覆盖训练起点 2020-01
                  的首档生效日；因子表实际起点可更早，多出的只是 lookback 余量）

本模块吸收两处旧注册表：pools.membership.POOLS（snap_table/band）与
config.FACTOR_TABLES（factor_table）。新池 = 在 POOLS 加一个 PoolSpec 条目，
勿在别处散落表名或池路径。池路径族（模型/预测/回测/报告目录）以 spec 方法
提供、ROOT 锚定（本仓代码所在仓的目录结构，而非 DB 所在仓——dev bench 经
QUANTLAB_DB 共享主仓 DB 时产物不越界）。

依赖方向：spec -> config（仅 ROOT/POOL_NAME），无环；membership/store/
dataset 等一切池感知模块经本模块取身份，spec 不依赖它们。

自描述 CLI（无参打印全部池；只读连接，不写库）：
    python -m pools.spec [--pool mainboard_all]
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import ROOT, POOL_NAME


@dataclass(frozen=True)
class PoolSpec:
    name: str
    snap_table: str
    factor_table: str
    band: tuple[float, float] | None
    data_since: str

    # ---- 路径族（原 config 池路径 helper 迁入；微盘路径逐字节不变）----
    # 2026-09-03 权重单文件纪律：折训练直接覆盖主权重路径，无折模型目录；
    # 折预测/meta/回测仍按折分目录（评估证据，非权重）。
    def model_dir(self) -> Path:
        return ROOT / "models" / self.name

    def lgb_model_path(self, model: str = "20d") -> Path:
        return self.model_dir() / f"lgb_{model}.joblib"

    def lgb_predictions_path(self, model: str = "20d", fold: str | None = None) -> Path:
        if fold:
            return ROOT / "data" / "folds" / fold / f"predictions__{self.name}_lgb_{model}.parquet"
        return ROOT / "data" / f"predictions__{self.name}_lgb_{model}.parquet"

    def lgb_predictions_meta_path(self, model: str = "20d", fold: str | None = None) -> Path:
        if fold:
            return ROOT / "data" / "folds" / fold / f"predictions__{self.name}_lgb_{model}_meta.json"
        return ROOT / "data" / f"predictions__{self.name}_lgb_{model}_meta.json"

    def backtest_dir(self, fold: str | None = None) -> Path:
        d = ROOT / "backtest" / self.name
        return d / "folds" / fold if fold else d

    def forecast_lgb_dir(self) -> Path:
        return ROOT / "forecast_display" / "html_lgb" / self.name

    def snapshots_json_path(self) -> Path:
        """人读版快照 JSON（构建器副产物，regenerable，gitignore）。"""
        return ROOT / "pools" / f"{self.name}_snapshots.json"


POOLS: dict[str, PoolSpec] = {
    # 微盘池（现役生产池）：快照带 1~40 亿流通市值（通胀调整），因子表沿用
    # 历史现名 factor_values（现役零改名契约）
    "mainboard_microcap": PoolSpec(
        name="mainboard_microcap",
        snap_table="pool_snapshots",
        factor_table="factor_values",
        band=(1.0, 40.0),
        data_since="2019-12-02",
    ),
    # 全 A 主板池（bench 2026-08-27）：主板全体（无市值带），横截面参考系独立
    "mainboard_all": PoolSpec(
        name="mainboard_all",
        snap_table="pool_snapshots_mainboard_all",
        factor_table="factor_values_mainboard_all",
        band=None,
        data_since="2019-12-02",
    ),
}


def get_pool(pool: str | None = None) -> PoolSpec:
    """池名 -> PoolSpec。缺省 = config.POOL_NAME（env QUANTLAB_POOL，默认微盘）。"""
    p = pool or POOL_NAME
    if p not in POOLS:
        raise ValueError(f"unknown pool {p!r}, expected one of {sorted(POOLS)}")
    return POOLS[p]


def _describe(spec: PoolSpec) -> None:
    band = f"{spec.band[0]:.0f}~{spec.band[1]:.0f} 亿流通市值带" if spec.band else "无市值带"
    print(f"[{spec.name}] {band}, data_since={spec.data_since}")
    print(f"  snap_table    = {spec.snap_table}")
    print(f"  factor_table  = {spec.factor_table}")
    print(f"  model_dir     = {spec.model_dir()}")
    try:
        import duckdb

        from config import DB_PATH

        con = duckdb.connect(str(DB_PATH), read_only=True)
        try:
            n_snap, lo, hi = con.execute(
                f"SELECT count(DISTINCT effective_date), min(effective_date), "
                f"max(effective_date) FROM {spec.snap_table}").fetchone()
            latest, n_members = con.execute(
                f"SELECT effective_date, count(DISTINCT code) FROM {spec.snap_table} "
                f"WHERE effective_date = (SELECT max(effective_date) FROM {spec.snap_table}) "
                f"GROUP BY effective_date").fetchone()
            n_rows, f_lo, f_hi = con.execute(
                f"SELECT count(*), min(date), max(date) FROM {spec.factor_table}"
            ).fetchone()
            n_cols = len(con.execute(
                f"SELECT * FROM {spec.factor_table} LIMIT 1").description)
            print(f"  snapshots     = {n_snap} 档（{lo} ~ {hi}），最新档 {latest}: {n_members} 只")
            print(f"  factor rows   = {n_rows:,} 行 / {n_cols} 列（{f_lo} ~ {f_hi}）")
        finally:
            con.close()
    except Exception as e:  # DB 不可达时仍打印静态身份
        print(f"  (DB 不可达，跳过运行时统计: {e})")


def main():
    import argparse

    ap = argparse.ArgumentParser(description="池注册表自描述（只读）")
    ap.add_argument("--pool", default=None, choices=sorted(POOLS),
                    help="只描述指定池（缺省=全部）")
    args = ap.parse_args()
    for spec in ([get_pool(args.pool)] if args.pool else list(POOLS.values())):
        _describe(spec)


if __name__ == "__main__":
    main()
