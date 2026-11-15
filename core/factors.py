"""因子库：因子的定义、释义、参数说明与计算逻辑。

每个因子都是一条「给 ETF 打分」的规则：
- direction = +1：数值越大越值得买
- direction = -1：数值越大越不值得买（例如波动率、回撤）

回测时会把因子值做横截面标准化（z-score），再按权重加总成综合得分。
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

FACTORS = [
    {
        "key": "momentum",
        "name": "动量因子",
        "tag": "趋势",
        "direction": 1,
        "direction_text": "越大越好（涨得越多越强）",
        "description": "衡量标的过去一段时间的涨幅，捕捉「强者更强」的趋势延续效应。它假设近期表现好的资产，短期还会继续表现好。",
        "formula": "动量 = 收盘价 / N 个交易日前收盘价 - 1",
        "param_name": "回看天数 N",
        "param_help": "计算涨幅所用的交易日数量。周期越短越敏感（噪声大），越长越平滑（反应慢）。常用 20 日（约 1 个月）、60 日（约 1 季度）。",
        "param_min": 2,
        "param_max": 250,
        "default_param": 20,
        "default_weight": 0.20,
        "note": "与「短期反转」是相反假设，同时开启时两者会互相抵消，建议只侧重其中一个。",
    },
    {
        "key": "reversal",
        "name": "短期反转因子",
        "tag": "反转",
        "direction": 1,
        "direction_text": "越大越好（近期跌得越多越优先）",
        "description": "取最近 N 日收益率的相反数：近期跌得多的标的得分更高。它假设短期存在「超跌反弹」效应，是 A 股较常见的短期规律。",
        "formula": "反转 = -(收盘价 / N 个交易日前收盘价 - 1)",
        "param_name": "回看天数 N",
        "param_help": "观察短期涨跌所用的交易日数量。反转效应通常看 3~10 个交易日，周期太长会变成动量。",
        "param_min": 2,
        "param_max": 60,
        "default_param": 5,
        "default_weight": 0.05,
        "note": "短周期反转在震荡市中较有效，在单边趋势市中可能反复被打脸。",
    },
    {
        "key": "volatility",
        "name": "波动率因子",
        "tag": "风险",
        "direction": -1,
        "direction_text": "越小越好（低波动更优）",
        "description": "日收益率标准差的年化值，衡量价格上下波动的剧烈程度。低波动组合通常回撤更小、持有体验更好（低波动异象）。",
        "formula": "波动率 = 日收益率标准差(N 日) × √252",
        "param_name": "回看天数 N",
        "param_help": "计算波动率所用的交易日数量。窗口越长，结果越稳定；窗口越短，对近期风险变化越敏感。",
        "param_min": 5,
        "param_max": 250,
        "default_param": 20,
        "default_weight": 0.15,
        "note": "牛市中低波动因子可能跑输高弹性品种，它更像「防守型」因子。",
    },
    {
        "key": "ma_bias",
        "name": "均线偏离因子",
        "tag": "趋势",
        "direction": 1,
        "direction_text": "越大越好（站上均线且偏离越多越强）",
        "description": "收盘价相对 N 日均线的偏离百分比。大于 0 表示价格在均线上方、趋势偏强；小于 0 表示跌破均线、趋势偏弱。",
        "formula": "偏离度 = 收盘价 / N 日均线 - 1",
        "param_name": "均线周期 N",
        "param_help": "均线的天数。20 日≈月线，60 日≈季线。周期越长，趋势信号越慢但越少假信号。",
        "param_min": 3,
        "param_max": 250,
        "default_param": 20,
        "default_weight": 0.10,
        "note": "偏离过大也可能意味着短期过热，可结合 RSI 一起看。",
    },
    {
        "key": "rsi",
        "name": "RSI 相对强弱",
        "tag": "反转",
        "direction": -1,
        "direction_text": "越小越好（超卖更优）",
        "description": "衡量近期上涨力量与下跌力量对比的指标，取值 0~100。数值高说明买盘强但可能超买，数值低说明卖压重但可能超卖。",
        "formula": "RSI = 100 - 100 / (1 + 平均涨幅 / 平均跌幅)",
        "param_name": "RSI 周期",
        "param_help": "计算平均涨跌幅所用的交易日数量。常用 6、14、24。周期越短越敏感。",
        "param_min": 2,
        "param_max": 100,
        "default_param": 14,
        "default_weight": 0.05,
        "note": "本系统按「超卖反弹」思路使用：RSI 越低得分越高（方向为负）。",
    },
    {
        "key": "volume_ratio",
        "name": "量比因子",
        "tag": "情绪",
        "direction": 1,
        "direction_text": "越大越好（放量更优）",
        "description": "短期均量 ÷ 长期均量，衡量成交活跃度变化。大于 1 表示近期相对放量，往往代表资金关注度提升。",
        "formula": "量比 = 5 日均成交量 / N 日均成交量",
        "param_name": "长期均量天数 N",
        "param_help": "作为对比基准的长期窗口。N 越大，基准越平稳。短期窗口固定为 5 日。",
        "param_min": 10,
        "param_max": 250,
        "default_param": 20,
        "default_weight": 0.10,
        "note": "放量既可能是资金进场，也可能是恐慌出逃，需要结合价格方向判断。",
    },
    {
        "key": "turnover",
        "name": "换手率因子",
        "tag": "情绪",
        "direction": 1,
        "direction_text": "越大越好（交易越活跃）",
        "description": "N 日平均换手率，反映 ETF 的成交活跃度与市场情绪热度。较高的换手通常意味着流动性好、关注度高。",
        "formula": "换手率 = N 日换手率均值",
        "param_name": "平均天数 N",
        "param_help": "计算换手率均值的天数。周期越长越能代表常态活跃度。",
        "param_min": 3,
        "param_max": 250,
        "default_param": 20,
        "default_weight": 0.05,
        "note": "换手率过高也可能是短期炒作过热，注意与波动率配合使用。",
    },
    {
        "key": "drawdown",
        "name": "近端回撤因子",
        "tag": "风险",
        "direction": -1,
        "direction_text": "越小越好（回撤越浅）",
        "description": "过去 N 日内的最大回撤幅度（正数表示回撤百分比）。数值越小，说明近期走势越稳、回撤越浅。",
        "formula": "回撤 = -min(收盘价 / N 日最高收盘价 - 1) × 100",
        "param_name": "回看天数 N",
        "param_help": "统计回撤的时间窗口。建议与回测区间匹配，常用 60 日（约 1 季度）或 120 日。",
        "param_min": 20,
        "param_max": 250,
        "default_param": 60,
        "default_weight": 0.15,
        "note": "回撤因子只描述过去，不代表未来；在快速反弹行情中低回撤标的可能涨得慢。",
    },
    {
        "key": "sharpe",
        "name": "夏普动量因子",
        "tag": "趋势",
        "direction": 1,
        "direction_text": "越大越好（风险调整后收益高）",
        "description": "N 日平均收益除以 N 日收益波动，衡量「每承担一份波动能换来多少收益」。比单纯涨幅更稳健，避免高波动品种靠运气上榜。",
        "formula": "夏普动量 = 日均收益率 / 日收益率标准差",
        "param_name": "回看天数 N",
        "param_help": "计算均值与波动所用的交易日数量。常用 60 日，能较好平衡敏感度与稳定性。",
        "param_min": 10,
        "param_max": 250,
        "default_param": 60,
        "default_weight": 0.15,
        "note": "波动率极低时该比值会被放大，系统已做横截面标准化，影响有限。",
    },
]

FACTOR_MAP: Dict[str, dict] = {f["key"]: f for f in FACTORS}


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_down = down.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def compute_series(df: pd.DataFrame, key: str, param: int) -> pd.Series:
    """计算单只 ETF 的因子时间序列（index 为日期）。"""
    data = df.copy()
    if "date" in data.columns:
        data = data.set_index("date")
    close = pd.to_numeric(data["close"], errors="coerce")
    ret = close.pct_change()
    param = int(param)

    if key == "momentum":
        series = close / close.shift(param) - 1
    elif key == "reversal":
        series = -(close / close.shift(param) - 1)
    elif key == "volatility":
        series = ret.rolling(param).std() * np.sqrt(252)
    elif key == "ma_bias":
        series = close / close.rolling(param).mean() - 1
    elif key == "rsi":
        series = _rsi(close, param)
    elif key == "volume_ratio":
        volume = pd.to_numeric(data["volume"], errors="coerce")
        series = volume.rolling(5).mean() / volume.rolling(param).mean()
    elif key == "turnover":
        turnover = pd.to_numeric(data.get("turn"), errors="coerce")
        series = turnover.rolling(param).mean()
    elif key == "drawdown":
        roll_max = close.rolling(param).max()
        series = -(close / roll_max - 1) * 100
    elif key == "sharpe":
        mean = ret.rolling(param).mean()
        std = ret.rolling(param).std()
        series = mean / std.replace(0, np.nan)
    else:
        raise KeyError(f"未知因子：{key}")

    return series.replace([np.inf, -np.inf], np.nan)


def factor_snapshot(
    data_map: Dict[str, pd.DataFrame],
    key: str,
    param: int,
    asof: pd.Timestamp,
    max_stale_days: int = 15,
) -> pd.Series:
    """取某个日期各 ETF 的因子原始值。"""
    asof = pd.Timestamp(asof)
    values = {}
    for code, df in data_map.items():
        series = compute_series(df, key, param)
        series = series[series.index <= asof].dropna()
        if series.empty:
            continue
        last_date = series.index[-1]
        if (asof - last_date).days > max_stale_days:
            continue
        values[code] = float(series.iloc[-1])
    return pd.Series(values, dtype=float)


def zscore_cross_section(series: pd.Series) -> pd.Series:
    """横截面标准化：减去均值再除以标准差，并截断极端值。"""
    values = series.dropna()
    if values.empty:
        return values
    if len(values) == 1:
        return pd.Series(0.0, index=values.index)
    std = values.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=values.index)
    z = (values - values.mean()) / std
    return z.clip(-3, 3)


def default_weights() -> Dict[str, float]:
    return {f["key"]: f["default_weight"] for f in FACTORS}


def default_params() -> Dict[str, int]:
    return {f["key"]: f["default_param"] for f in FACTORS}
