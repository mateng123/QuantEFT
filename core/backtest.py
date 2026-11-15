"""回测引擎：多因子打分 → 定期调仓 → 组合净值与绩效指标。

流程（每个调仓日）：
1. 用「当天及之前」的行情计算各因子原始值（不使用未来数据）；
2. 对每只 ETF 的每个因子做横截面 z-score 标准化；
3. 综合得分 = Σ(权重 × 方向 × z-score) / Σ权重；
4. 选取得分最高的 top_k 只 ETF，等权或按得分加权持有；
5. 计算换手率与交易成本，持有到下一个调仓日。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .factors import FACTOR_MAP, compute_series, zscore_cross_section


@dataclass
class BacktestConfig:
    start: pd.Timestamp
    end: pd.Timestamp
    rebalance_days: int = 20
    top_k: int = 3
    weighting: str = "equal"  # equal = 等权；score = 按得分加权
    cost_rate: float = 0.0005  # 单边交易成本（0.0005 = 万 5）
    risk_free: float = 0.0


@dataclass
class BacktestResult:
    config: BacktestConfig
    equity: pd.Series
    benchmark: pd.Series
    drawdown: pd.Series
    daily_returns: pd.Series
    benchmark_returns: pd.Series
    metrics: Dict[str, float]
    weight_history: pd.DataFrame
    rebalance_log: pd.DataFrame
    holdings_history: Optional[pd.DataFrame] = None
    latest_detail: Optional[pd.DataFrame] = None
    monthly_returns: Optional[pd.DataFrame] = None
    monthly_benchmark: Optional[pd.DataFrame] = None


def _prepare_close(data_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = {}
    for code, df in data_map.items():
        s = df.set_index("date")["close"].astype(float)
        frames[code] = s[~s.index.duplicated(keep="last")]
    close = pd.DataFrame(frames).sort_index()
    # 停牌/缺失行情用前值填充，避免因子与收益中断
    close = close.ffill()
    return close


def _build_factor_frames(data_map, weights, params, index):
    frames = {}
    for key, weight in weights.items():
        if weight <= 0:
            continue
        param = int(params.get(key, FACTOR_MAP[key]["default_param"]))
        cols = {}
        for code, df in data_map.items():
            series = compute_series(df, key, param)
            cols[code] = series[~series.index.duplicated(keep="last")]
        frame = pd.DataFrame(cols).reindex(index).ffill(limit=10)
        frames[key] = frame
    return frames


def _composite_scores(
    factor_snapshots: Dict[str, pd.Series],
    weights: Dict[str, float],
    total_weight: float,
    close_row: pd.Series,
):
    codes = set()
    for snap in factor_snapshots.values():
        codes |= set(snap.dropna().index)
    codes = [c for c in codes if pd.notna(close_row.get(c))]
    if not codes:
        return pd.Series(dtype=float), pd.DataFrame()

    composite = pd.Series(0.0, index=codes)
    valid_counts = pd.Series(0, index=codes)
    detail = pd.DataFrame(index=codes)
    detail.index.name = "code"

    for key, snap in factor_snapshots.items():
        raw = snap.reindex(codes)
        z = zscore_cross_section(raw)
        detail[key] = raw
        detail[f"{key}_z"] = np.nan
        if not z.empty:
            detail.loc[z.index, f"{key}_z"] = z
            composite.loc[z.index] += z * weights[key] * FACTOR_MAP[key]["direction"] / total_weight
            valid_counts.loc[z.index] += 1

    composite = composite[valid_counts > 0]
    if composite.empty:
        return composite, detail

    detail["composite"] = composite
    return composite, detail


def _allocate(selected: pd.Series, method: str) -> pd.Series:
    if selected.empty:
        return pd.Series(dtype=float)
    if method == "score":
        shifted = selected - selected.min() + 1.0
        return shifted / shifted.sum()
    return pd.Series(1.0 / len(selected), index=selected.index)


def _target_from_weights(previous: pd.Series, target: pd.Series):
    index = previous.index.union(target.index)
    old = previous.reindex(index).fillna(0.0)
    new = target.reindex(index).fillna(0.0)
    traded = float((new - old).abs().sum())  # 双边成交额 / 组合净值
    return traded


def _metrics(
    equity: pd.Series,
    benchmark: pd.Series,
    daily_returns: pd.Series,
    config: BacktestConfig,
    traded_list,
) -> Dict[str, float]:
    if equity.empty:
        return {}

    n = len(equity)
    years = max(n / 252.0, 1e-9)
    total_return = float(equity.iloc[-1] - 1.0)
    annual_return = float((1.0 + total_return) ** (1.0 / years) - 1.0) if total_return > -1 else -1.0
    vol = float(daily_returns.std(ddof=1) * np.sqrt(252))
    sharpe = float((daily_returns.mean() * 252 - config.risk_free) / vol) if vol > 0 else 0.0
    drawdown = equity / equity.cummax() - 1
    max_dd = float(drawdown.min())
    calmar = float(annual_return / abs(max_dd)) if max_dd < 0 else 0.0

    bench_total = float(benchmark.iloc[-1] - 1.0) if not benchmark.empty else 0.0
    monthly = (1 + daily_returns).groupby([daily_returns.index.year, daily_returns.index.month]).prod() - 1
    avg_traded = float(np.mean(traded_list)) if traded_list else 0.0

    return {
        "总收益率": total_return,
        "年化收益率": annual_return,
        "年化波动率": vol,
        "夏普比率": sharpe,
        "最大回撤": max_dd,
        "卡玛比率": calmar,
        "日胜率": float((daily_returns > 0).mean()),
        "月胜率": float((monthly > 0).mean()) if len(monthly) else 0.0,
        "基准总收益率": bench_total,
        "超额收益率": total_return - bench_total,
        "平均单边换手率": avg_traded / 2.0,
        "年调仓次数": 252.0 / config.rebalance_days,
        "回测天数": n,
    }


def _monthly_table(returns: pd.Series) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame()
    monthly = (1 + returns).groupby([returns.index.year, returns.index.month]).prod() - 1
    table = monthly.unstack(level=1)
    table.columns = [f"{int(m)}月" for m in table.columns]
    table.index.name = "年份"
    return table


def run_backtest(
    data_map: Dict[str, pd.DataFrame],
    weights: Dict[str, float],
    params: Dict[str, int],
    config: BacktestConfig,
) -> BacktestResult:
    if not data_map:
        raise ValueError("没有可用的行情数据，请先在「数据下载」页下载数据")
    if len(data_map) < 2:
        raise ValueError("至少需要 2 只 ETF 才能做横截面比较")

    active = {k: float(v) for k, v in weights.items() if v and v > 0 and k in FACTOR_MAP}
    if not active:
        raise ValueError("请至少为一个因子设置大于 0 的权重")
    total_weight = sum(active.values())

    close = _prepare_close(data_map)
    returns = close.pct_change()

    dates = close.index[(close.index >= pd.Timestamp(config.start)) & (close.index <= pd.Timestamp(config.end))]
    if len(dates) < 5:
        raise ValueError("回测区间内交易日太少，请扩大时间范围或先下载数据")
    if len(dates) < config.rebalance_days:
        raise ValueError("回测区间短于一个调仓周期，请缩短调仓周期或扩大时间范围")

    factor_frames = _build_factor_frames(data_map, active, params, close.index)
    rebalance_dates = dates[:: config.rebalance_days]
    rebalance_set = set(rebalance_dates)

    weight_history = pd.DataFrame(0.0, index=dates, columns=close.columns)
    portfolio_returns = pd.Series(0.0, index=dates)
    current_weights = pd.Series(dtype=float)
    traded_list = []
    records = []
    holdings_rows = []
    latest_detail = None

    for i, day in enumerate(dates):
        # 记录当天实际持有的权重（调仓在收盘后生效，次日开始承担新权重）
        if not current_weights.empty:
            weight_history.loc[day, current_weights.index] = current_weights.values

        if i > 0 and not current_weights.empty:
            day_ret = returns.loc[day].reindex(current_weights.index).fillna(0.0)
            portfolio_returns.iloc[i] = float((current_weights * day_ret).sum())

        if day in rebalance_set:
            snapshots = {key: frame.loc[day] for key, frame in factor_frames.items()}
            composite, detail = _composite_scores(snapshots, active, total_weight, close.loc[day])
            if len(composite) < 2:
                # 有效标的不足 2 只时无法做横截面比较，跳过本次调仓
                continue

            selected = composite.sort_values(ascending=False).head(max(1, int(config.top_k)))
            target = _allocate(selected, config.weighting)
            traded = _target_from_weights(current_weights, target)
            traded_list.append(traded)
            portfolio_returns.iloc[i] -= traded * config.cost_rate

            current_weights = target
            records.append(
                {
                    "调仓日": day.date(),
                    "持仓数量": len(target),
                    "双边换手": round(traded, 4),
                    "交易成本": round(traded * config.cost_rate, 6),
                    "持仓明细": "、".join(
                        f"{code}({w:.0%})" for code, w in target.sort_values(ascending=False).items()
                    ),
                }
            )
            detail["selected"] = detail.index.isin(target.index)
            detail["weight"] = target.reindex(detail.index).fillna(0.0)
            latest_detail = detail
            holdings_rows.append(target.rename(day))

    if not records:
        raise ValueError("回测区间内没有产生有效调仓，请检查数据覆盖范围")

    equity = (1.0 + portfolio_returns).cumprod()
    benchmark_returns = returns.loc[dates].mean(axis=1).fillna(0.0)
    benchmark = (1.0 + benchmark_returns).cumprod()

    metrics = _metrics(equity, benchmark, portfolio_returns, config, traded_list)
    drawdown = equity / equity.cummax() - 1

    result = BacktestResult(
        config=config,
        equity=equity,
        benchmark=benchmark,
        drawdown=drawdown,
        daily_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        metrics=metrics,
        weight_history=weight_history,
        rebalance_log=pd.DataFrame(records),
        holdings_history=pd.DataFrame(holdings_rows).fillna(0.0) if holdings_rows else pd.DataFrame(),
        latest_detail=latest_detail,
        monthly_returns=_monthly_table(portfolio_returns),
        monthly_benchmark=_monthly_table(benchmark_returns),
    )
    return result
