"""ETF 多因子量化学习系统（Streamlit 界面）。

启动方式：
    streamlit run app.py
"""

from __future__ import annotations

import datetime as dt
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core.backtest import BacktestConfig, run_backtest
from core.data_loader import (
    DATA_DIR,
    coverage,
    download_many,
    load_meta,
    load_series_map,
    local_codes,
    normalize_code,
    read_cache,
)
from core.factors import FACTORS, FACTOR_MAP
from core.universe import DEFAULT_CODES, ETF_UNIVERSE

st.set_page_config(page_title="ETF 多因子量化学习系统", page_icon="📈", layout="wide")

COLOR_MAIN = "#2563eb"
COLOR_BENCH = "#9ca3af"
COLOR_WARN = "#ef4444"


# ------------------------------------------------------------------
# 通用小工具
# ------------------------------------------------------------------
def pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def init_state() -> None:
    for f in FACTORS:
        st.session_state.setdefault(f"w_{f['key']}", int(round(f["default_weight"] * 100)))
        st.session_state.setdefault(f"p_{f['key']}", int(f["default_param"]))


def reset_factors() -> None:
    for f in FACTORS:
        st.session_state[f"w_{f['key']}"] = int(round(f["default_weight"] * 100))
        st.session_state[f"p_{f['key']}"] = int(f["default_param"])


def equalize_factors() -> None:
    for f in FACTORS:
        st.session_state[f"w_{f['key']}"] = 100


def current_weights() -> dict:
    return {f["key"]: st.session_state[f"w_{f['key']}"] / 100.0 for f in FACTORS}


def current_params() -> dict:
    return {f["key"]: int(st.session_state[f"p_{f['key']}"]) for f in FACTORS}


def date_range_fields(prefix: str, default_start, default_end, help_text: str = ""):
    """两个可直接键盘输入的日期框，返回 (开始文本, 结束文本)。"""
    c1, c2 = st.columns(2)
    start = c1.text_input(
        "开始日期",
        value=pd.Timestamp(default_start).strftime("%Y-%m-%d"),
        key=f"{prefix}_start",
        placeholder="YYYY-MM-DD",
        help=help_text,
    )
    end = c2.text_input(
        "结束日期",
        value=pd.Timestamp(default_end).strftime("%Y-%m-%d"),
        key=f"{prefix}_end",
        placeholder="YYYY-MM-DD",
        help=help_text,
    )
    return start, end


def parse_date_text(text: str):
    try:
        return dt.datetime.strptime(str(text).strip(), "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def validate_date_range(start_text, end_text, min_date: dt.date, max_date: dt.date):
    """校验输入日期，返回 (start, end, 是否有效)。"""
    start = parse_date_text(start_text)
    end = parse_date_text(end_text)
    if start is None or end is None:
        st.error("日期格式不正确，请按 YYYY-MM-DD 输入，例如 2024-01-01。")
        return None, None, False
    if start < min_date or end > max_date:
        st.error(f"日期需在 {min_date} 至 {max_date} 之间。")
        return start, end, False
    if start >= end:
        st.error("开始日期必须早于结束日期。")
        return start, end, False
    return start, end, True


ETF_INFO = {e["code"]: e for e in ETF_UNIVERSE}


def sec_name(code: str) -> str:
    """优先使用本地名称表，其次内置 ETF 名单，最后回退为代码。"""
    name = load_meta().get(code, {}).get("name")
    if name:
        return name
    item = ETF_INFO.get(code)
    return item["name"] if item else code


def sec_label(code: str) -> str:
    name = sec_name(code)
    return code if name == code else f"{name}（{code}）"


def sec_kind(code: str) -> str:
    item = ETF_INFO.get(code)
    if item:
        return item["category"]
    return load_meta().get(code, {}).get("kind") or "其它"


def parse_custom_codes(text: str):
    """解析用户输入的证券代码，返回（合法代码, 无法识别的片段）。"""
    tokens = [t for t in re.split(r"[,，;；\s]+", (text or "").strip()) if t]
    codes, invalid = [], []
    for token in tokens:
        code = normalize_code(token)
        if code is None:
            invalid.append(token)
        elif code not in codes:
            codes.append(code)
    return codes, invalid


def data_span(codes):
    starts, ends = [], []
    for code in codes:
        cov = coverage(code)
        if cov["rows"]:
            starts.append(pd.Timestamp(cov["start"]))
            ends.append(pd.Timestamp(cov["end"]))
    if not starts:
        return None
    return min(starts), max(ends)


def chart_layout(fig, height: int = 420, y_title: str = "") -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        yaxis_title=y_title,
        template="plotly_white",
        xaxis_tickformat="%Y-%m-%d",
        xaxis_hoverformat="%Y-%m-%d",
    )
    return fig


# ------------------------------------------------------------------
# 侧边栏
# ------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown("## 📈 ETF 量化学习系统")
        st.caption("数据来源：baostock 真实行情（前复权日线）")

        codes = st.multiselect(
            "标的池（回测候选）",
            options=sorted({e["code"] for e in ETF_UNIVERSE} | set(local_codes())),
            default=[c for c in DEFAULT_CODES if c in set(local_codes())],
            format_func=sec_label,
            help="ETF、股票、指数都可以。已下载到本地的代码会自动出现在这里，回测只使用本地数据。",
        )

        st.divider()
        st.markdown("**推荐使用顺序**")
        st.markdown(
            "1. 在「数据下载」把行情保存为 CSV\n"
            "2. 在「数据查看」核对数据、选择区间\n"
            "3. 在「因子配置」调整权重和参数\n"
            "4. 在「策略回测」选择区间并运行\n"
            "5. 对照「学习指南」理解结果"
        )
        st.divider()
        st.caption(f"本地 CSV 目录：`{DATA_DIR}`（名称表 `_meta.json`）")
        st.warning("仅用于量化学习与研究，不构成投资建议。历史回测不代表未来收益。", icon="⚠️")

    return codes


# ------------------------------------------------------------------
# Tab 1：数据
# ------------------------------------------------------------------
def _show_messages() -> None:
    for level, text in st.session_state.pop("dl_messages", []):
        getattr(st, level)(text)


def _run_download(targets, start_date, end_date) -> None:
    bar = st.progress(0.0, text="准备下载…")

    def cb(i, total, code):
        bar.progress(i / total, text=f"正在下载 {sec_label(code)}（{i}/{total}）")

    try:
        _, errors = download_many(
            targets, start=str(start_date), end=str(end_date), progress_cb=cb
        )
        bar.empty()
        messages = [("warning", f"{sec_label(code)}：{msg}") for code, msg in errors.items()]
        ok = len(targets) - len(errors)
        if ok:
            messages.append(
                ("success", f"成功更新 {ok} 个标的，CSV 已保存到 data/ 目录，名称已记录到 _meta.json。")
            )
        else:
            messages.append(("error", "全部标的下载失败，请检查网络后重试。"))
    except Exception as exc:
        bar.empty()
        messages = [("error", f"下载失败：{exc}")]

    st.session_state["dl_messages"] = messages
    st.rerun()


def render_data_tab(codes) -> None:
    st.subheader("1. 数据下载（保存到本地 CSV）")
    st.caption(
        "行情全部来自 baostock 真实日线（前复权），下载后按 `市场_代码.csv` 保存到本地 data/ 目录。"
        "回测和查看只读取本地文件；再次下载会与已有历史取并集、整段刷新。"
    )
    _show_messages()

    st.markdown("**下载区间**")
    start_text, end_text = date_range_fields(
        "dl",
        dt.date(2018, 1, 1),
        dt.date.today(),
        help_text="支持直接键盘输入，格式 YYYY-MM-DD。上市前的日期会自动忽略。",
    )
    start_date, end_date, dates_ok = validate_date_range(
        start_text, end_text, dt.date(2005, 1, 1), dt.date.today()
    )
    st.caption(
        "说明：前复权价格会随之后的分红除权整体变化。更新数据时会自动重拉该标的的全部历史并整体替换缓存，"
        "避免新旧两套复权基准拼接造成价格断层。"
    )

    custom_text = st.text_area(
        "额外证券代码（可留空）",
        placeholder="例如：sh.600519, 000001, 510300, 600036.SH",
        help="支持股票、ETF、指数；多个代码用逗号、空格或换行分隔，大小写和前缀都可以省略。下载时会自动查询并保存证券名称。",
        key="custom_codes",
        height=80,
    )
    custom_codes, invalid = parse_custom_codes(custom_text)
    if invalid:
        st.error("以下代码无法识别：" + "、".join(invalid))

    targets = list(dict.fromkeys(list(codes) + custom_codes))
    st.caption(
        f"本次将处理 {len(targets)} 个标的："
        + ("、".join(sec_label(c) for c in targets[:6]) + ("…" if len(targets) > 6 else "") if targets else "（无）")
    )

    b1, b2 = st.columns(2)
    with b1:
        clicked = st.button(
            "下载 / 更新数据",
            type="primary",
            width="stretch",
            disabled=not dates_ok or bool(invalid) or not targets,
        )
    with b2:
        all_cached = local_codes()
        update_all = st.button(
            "更新全部本地标的到最新",
            width="stretch",
            disabled=not all_cached,
            help="对 data/ 下所有已有标的重新拉取最新行情（历史部分也会一并刷新）。",
        )

    if clicked:
        _run_download(targets, start_date, end_date)
    if update_all:
        _run_download(all_cached, dt.date.today() - dt.timedelta(days=30), dt.date.today())

    st.divider()
    st.markdown("**本地数据文件（data/*.csv）**")
    rows = []
    for code in all_cached:
        cov = coverage(code)
        if not cov["rows"]:
            continue
        rows.append(
            {
                "代码": code,
                "名称": sec_name(code),
                "类别": sec_kind(code),
                "在回测池": "✅" if code in codes else "",
                "缓存条数": cov["rows"],
                "开始日期": str(cov["start"]),
                "结束日期": str(cov["end"]),
                "更新时间": cov["updated"] or "-",
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=360)
    else:
        st.info("本地还没有数据，请先选择区间并点击「下载 / 更新数据」。")

    st.caption("数据导出（下载到浏览器）请到「📊 数据查看」页选择标的与日期区间。")


# ------------------------------------------------------------------
# Tab 2：数据查看与导出
# ------------------------------------------------------------------
def price_chart(df: pd.DataFrame, title: str) -> go.Figure:
    plot_df = df.copy()
    plot_df["ret"] = plot_df["close"].pct_change() * 100

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.08,
        subplot_titles=("前复权收盘价", "日涨跌幅"),
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["date"],
            y=plot_df["close"],
            name="收盘价",
            line=dict(color=COLOR_MAIN, width=1.6),
            hovertemplate="%{x|%Y-%m-%d}<br>收盘价 %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    bar_colors = ["#ef4444" if v >= 0 else "#22c55e" for v in plot_df["ret"].fillna(0)]
    fig.add_trace(
        go.Bar(
            x=plot_df["date"],
            y=plot_df["ret"],
            name="日涨跌幅",
            marker_color=bar_colors,
            hovertemplate="%{x|%Y-%m-%d}<br>涨跌幅 %{y:.2f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        height=480,
        margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_white",
        title=title,
        showlegend=False,
        hovermode="x unified",
    )
    fig.update_xaxes(tickformat="%Y-%m-%d", hoverformat="%Y-%m-%d")
    fig.update_yaxes(title_text="前复权收盘价", row=1, col=1)
    fig.update_yaxes(title_text="涨跌幅", ticksuffix="%", row=2, col=1)
    return fig


def render_view_tab() -> None:
    st.subheader("2. 数据查看与导出")
    all_cached = local_codes()
    if not all_cached:
        st.info("本地还没有数据，请先到「📥 数据下载」页下载。")
        return

    code = st.selectbox("选择标的", all_cached, format_func=sec_label, key="view_code")
    df = read_cache(code)
    if df is None or df.empty:
        st.warning("该标的的本地文件为空，请重新下载。")
        return

    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    st.caption(
        f"数据来源：baostock 真实行情（前复权）｜ 本地共 {len(df):,} 条，{min_date} ~ {max_date}"
    )

    default_start = max(min_date, (df["date"].max() - pd.Timedelta(366, unit="D")).date())
    start_text, end_text = date_range_fields(
        "view",
        default_start,
        max_date,
        help_text="输入 YYYY-MM-DD。区间越短，短期波动看得越清楚。",
    )
    vs, ve, dates_ok = validate_date_range(start_text, end_text, min_date, max_date)
    if not dates_ok:
        return

    sub = df[(df["date"] >= pd.Timestamp(vs)) & (df["date"] <= pd.Timestamp(ve))].copy()
    if sub.empty:
        st.warning("所选区间内没有数据，请调整日期。")
        return

    st.plotly_chart(price_chart(sub, sec_label(code)), width="stretch")

    ret = sub["close"].iloc[-1] / sub["close"].iloc[0] - 1
    ret_series = sub["close"].pct_change() * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("区间涨跌幅", f"{ret:+.2%}")
    c2.metric("数据条数", f"{len(sub):,}")
    c3.metric("交易日覆盖", f"{sub['date'].min().date()} ~ {sub['date'].max().date()}")
    if ret_series.notna().any():
        best_idx = ret_series.idxmax()
        worst_idx = ret_series.idxmin()
        st.caption(
            f"最大单日涨幅 {sub.loc[best_idx, 'date'].date()} +{ret_series.loc[best_idx]:.2f}%；"
            f"最大单日跌幅 {sub.loc[worst_idx, 'date'].date()} {ret_series.loc[worst_idx]:.2f}%。"
            "红色为上涨、绿色为下跌。"
        )

    st.download_button(
        "导出所选区间 CSV（浏览器下载）",
        data=sub.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{code.replace('.', '_')}_{vs}_{ve}.csv",
        mime="text/csv",
        key="export_range",
    )

    st.markdown("**数据明细**")
    show = sub.sort_values("date", ascending=False).copy()
    show["date"] = show["date"].dt.strftime("%Y-%m-%d")
    st.dataframe(show, hide_index=True, width="stretch", height=320)


# ------------------------------------------------------------------
# Tab 2：因子配置
# ------------------------------------------------------------------
def render_factor_tab() -> None:
    st.subheader("3. 因子权重与参数")
    st.markdown(
        "每个因子都是一条「给 ETF 打分」的规则。系统会在每个调仓日对候选池内所有 ETF "
        "计算因子值、做横截面标准化，再按下面的权重加总成**综合得分**，得分越高越优先买入。"
    )
    st.info(
        "权重是相对值：拖动后系统会自动归一化，例如两个因子权重为 30 和 10，"
        "等价于 75% 和 25%。把某个因子权重拖到 0 即表示停用该因子。",
        icon="💡",
    )

    b1, b2, _ = st.columns([1, 1, 2])
    b1.button("恢复默认权重与参数", on_click=reset_factors, width="stretch")
    b2.button("全部因子等权", on_click=equalize_factors, width="stretch")

    col_a, col_b = st.columns(2)
    for idx, f in enumerate(FACTORS):
        target = col_a if idx % 2 == 0 else col_b
        with target:
            with st.container(border=True):
                direction_icon = "🔼" if f["direction"] > 0 else "🔽"
                st.markdown(f"#### {direction_icon} {f['name']}  `{f['tag']}`")
                st.caption(f["description"])
                st.slider(
                    "权重（%），系统自动归一化",
                    min_value=0,
                    max_value=100,
                    step=5,
                    key=f"w_{f['key']}",
                    help=f"该因子在综合得分中的相对重要性。{f['description']}",
                )
                st.number_input(
                    f["param_name"],
                    min_value=int(f["param_min"]),
                    max_value=int(f["param_max"]),
                    step=1,
                    key=f"p_{f['key']}",
                    help=f["param_help"],
                )
                st.caption(f"计算公式：`{f['formula']}`")
                st.caption(f"打分方向：{f['direction_text']}")
                st.caption(f"使用提示：{f['note']}")

    st.divider()
    weights = current_weights()
    params = current_params()
    total = sum(weights.values())
    if total <= 0:
        st.error("所有因子权重都是 0，请至少给一个因子设置权重。")
        return

    st.subheader("当前生效的因子权重")
    norm = pd.Series({f["name"]: weights[f["key"]] / total for f in FACTORS})
    norm = norm[norm > 0].sort_values()
    fig = go.Figure(
        go.Bar(
            x=norm.values * 100,
            y=norm.index,
            orientation="h",
            marker_color=COLOR_MAIN,
            text=[f"{v*100:.1f}%" for v in norm.values],
            textposition="outside",
        )
    )
    fig.update_layout(
        height=120 + 36 * len(norm),
        margin=dict(l=10, r=30, t=10, b=10),
        template="plotly_white",
        xaxis_title="归一化权重（%）",
        xaxis_range=[0, max(10, norm.max() * 100 * 1.25)],
    )
    st.plotly_chart(fig, width="stretch")

    table = pd.DataFrame(
        [
            {
                "因子": f["name"],
                "类别": f["tag"],
                "权重(%)": weights[f["key"]],
                "归一化后": f"{weights[f['key']] / total:.1%}",
                "参数": f"{f['param_name']}={params[f['key']]}",
                "打分方向": f["direction_text"],
            }
            for f in FACTORS
        ]
    )
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption("调整完权重和参数后，切换到「策略回测」页运行即可看到结果变化。")


# ------------------------------------------------------------------
# Tab 3：回测
# ------------------------------------------------------------------
def render_backtest_tab(codes) -> None:
    st.subheader("4. 策略回测")
    if not codes:
        st.warning("请先在左侧选择 ETF 池。")
        return

    span = data_span(codes)
    missing = [c for c in codes if coverage(c)["rows"] == 0]
    if missing:
        st.warning(
            "以下标的还没有本地数据，回测时会被忽略：" + "、".join(sec_name(c) for c in missing)
        )
    if span is None:
        st.info("还没有可用数据，请先到「数据下载」页下载真实行情。")
        return

    span_start, span_end = span
    default_start = max(span_start, span_end - pd.DateOffset(years=3))
    st.caption(
        f"当前数据覆盖区间：{span_start.date()} ~ {span_end.date()}；"
        f"共 {len(codes) - len(missing)} 只 ETF 参与回测。"
    )

    with st.container(border=True):
        st.markdown("**回测区间**")
        start_text, end_text = date_range_fields(
            f"bt_{span_start.date()}_{span_end.date()}_{'_'.join(codes)}",
            default_start,
            span_end,
            help_text="支持直接键盘输入，格式 YYYY-MM-DD。因子计算会额外使用区间之前的历史数据，避免「未来函数」。",
        )
        bt_start, bt_end, dates_ok = validate_date_range(
            start_text, end_text, span_start.date(), span_end.date()
        )
        st.caption(f"可用范围：{span_start.date()} ~ {span_end.date()}")

        c1, c2, c3 = st.columns(3)
        with c1:
            rebalance_days = st.selectbox(
                "调仓周期",
                options=[5, 10, 20, 60],
                index=2,
                format_func=lambda d: {
                    5: "每 5 个交易日（约 1 周）",
                    10: "每 10 个交易日（约 2 周）",
                    20: "每 20 个交易日（约 1 月）",
                    60: "每 60 个交易日（约 1 季）",
                }[d],
                help="每隔多少个交易日重新打分并调仓。周期越短交易越频繁、成本越高；周期越长越稳定但反应越慢。",
            )
        with c2:
            max_k = max(1, min(8, len(codes) - len(missing)))
            top_k = st.slider(
                "持仓数量 top_k",
                min_value=1,
                max_value=max_k,
                value=min(3, max_k),
                help="每个调仓日买入综合得分最高的前几只 ETF。持仓越集中，弹性越大、波动也越大。",
            )
        with c3:
            weighting = st.radio(
                "组合加权方式",
                options=["equal", "score"],
                format_func=lambda x: "等权持有" if x == "equal" else "按得分加权",
                horizontal=True,
                help="等权：入选 ETF 各占相同仓位；按得分加权：得分越高仓位越高，更集中。",
            )

        c4, c5 = st.columns(2)
        with c4:
            cost_bp = st.number_input(
                "单边交易成本（万分之）",
                min_value=0.0,
                max_value=50.0,
                value=5.0,
                step=1.0,
                help="每次买卖按成交金额收取的费用，包含佣金与冲击成本。万 5 = 0.05%。",
            )
        with c5:
            st.metric("基准", "等权 ETF 池", help="把所有候选 ETF 等权持有并每日再平衡，作为对照基准。")

    run = st.button("运行回测", type="primary", width="stretch", disabled=not dates_ok)

    if run:
        weights = current_weights()
        params = current_params()
        if sum(weights.values()) <= 0:
            st.error("请至少为一个因子设置大于 0 的权重。")
            return

        warmup_start = pd.Timestamp(bt_start) - pd.Timedelta(450, unit="D")
        data_map = load_series_map(
            codes, start=warmup_start, end=pd.Timestamp(bt_end)
        )
        config = BacktestConfig(
            start=pd.Timestamp(bt_start),
            end=pd.Timestamp(bt_end),
            rebalance_days=int(rebalance_days),
            top_k=int(top_k),
            weighting=weighting,
            cost_rate=float(cost_bp) / 10000.0,
        )
        try:
            with st.spinner("正在计算因子并回测…"):
                result = run_backtest(data_map, weights, params, config)
            st.session_state["bt_payload"] = {
                "result": result,
                "codes": list(codes),
                "weights": weights,
                "params": params,
            }
        except Exception as exc:
            st.error(f"回测失败：{exc}")
            return

    payload = st.session_state.get("bt_payload")
    if not payload:
        st.info("设置好参数后点击「运行回测」查看结果。建议先用默认参数跑一遍，再逐步调整因子权重。")
        return
    render_result(payload)


def render_result(payload: dict) -> None:
    res = payload["result"]
    if not hasattr(res, "holdings_history"):
        st.session_state.pop("bt_payload", None)
        st.warning("检测到旧版本的回测结果，请重新点击「运行回测」。")
        return
    m = res.metrics
    cfg = res.config
    weights = payload["weights"]
    active = {k: v for k, v in weights.items() if v > 0}
    total_w = sum(active.values())

    st.divider()
    st.subheader("回测结果")
    st.caption(
        f"数据：baostock 真实行情 ｜ "
        f"区间 {cfg.start.date()} ~ {cfg.end.date()} ｜ 调仓周期 {cfg.rebalance_days} 个交易日 ｜ "
        f"持仓 {cfg.top_k} 只 ｜ "
        f"{'等权' if cfg.weighting == 'equal' else '得分加权'} ｜ 单边成本 {cfg.cost_rate*10000:.1f} 万分之一 ｜ "
        f"因子：" + "、".join(
            f"{FACTOR_MAP[k]['name']} {v/total_w:.0%}" for k, v in sorted(active.items(), key=lambda x: -x[1])
        )
    )

    c = st.columns(4)
    c[0].metric("总收益率", pct(m["总收益率"]), delta=f"超额 {pct(m['超额收益率'])}")
    c[1].metric("年化收益率", pct(m["年化收益率"]))
    c[2].metric("最大回撤", pct(m["最大回撤"]))
    c[3].metric("夏普比率", f"{m['夏普比率']:.2f}")
    c = st.columns(4)
    c[0].metric("年化波动率", pct(m["年化波动率"]))
    c[1].metric("卡玛比率", f"{m['卡玛比率']:.2f}")
    c[2].metric("日胜率", pct(m["日胜率"], 1))
    c[3].metric("基准总收益", pct(m["基准总收益率"]))

    st.markdown("#### 净值曲线")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=res.equity.index, y=res.equity, name="多因子组合",
            line=dict(color=COLOR_MAIN, width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=res.benchmark.index, y=res.benchmark, name="等权 ETF 池（基准）",
            line=dict(color=COLOR_BENCH, width=1.5, dash="dash"),
        )
    )
    chart_layout(fig, 420, "净值")
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("#### 回撤")
        fig = go.Figure(
            go.Scatter(
                x=res.drawdown.index, y=res.drawdown * 100, name="回撤",
                fill="tozeroy", line=dict(color=COLOR_WARN, width=1),
            )
        )
        fig.update_layout(
            height=300, margin=dict(l=10, r=10, t=30, b=10),
            template="plotly_white", yaxis_ticksuffix="%",
            xaxis_tickformat="%Y-%m-%d", xaxis_hoverformat="%Y-%m-%d",
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        st.markdown("#### 持仓权重变化")
        wdf = res.weight_history
        cols = [c for c in wdf.columns if wdf[c].max() > 0]
        fig = go.Figure()
        for code in cols:
            fig.add_trace(
                go.Scatter(
                    x=wdf.index, y=wdf[code] * 100, name=sec_name(code),
                    stackgroup="one", line=dict(width=0.4),
                )
            )
        fig.update_layout(
            height=300, margin=dict(l=10, r=10, t=30, b=10),
            template="plotly_white", yaxis_ticksuffix="%", yaxis_title="仓位",
            legend=dict(orientation="v", x=1.02, y=1),
            xaxis_tickformat="%Y-%m-%d", xaxis_hoverformat="%Y-%m-%d",
        )
        st.plotly_chart(fig, width="stretch")

    st.markdown("#### 月度收益热力图")
    which = st.radio(
        "查看对象", ["多因子组合", "等权基准"], horizontal=True, key="heatmap_target"
    )
    monthly = res.monthly_returns if which == "多因子组合" else res.monthly_benchmark
    if monthly is None or monthly.empty:
        st.info("区间太短，暂无月度数据。")
    else:
        z = monthly.values * 100
        text = [
            [f"{v:.1f}%" if pd.notna(v) else "" for v in row] for row in z
        ]
        fig = go.Figure(
            go.Heatmap(
                z=z, x=list(monthly.columns), y=[str(i) for i in monthly.index],
                colorscale="RdYlGn", zmid=0, text=text, texttemplate="%{text}",
                colorbar=dict(ticksuffix="%"),
                hovertemplate="%{y} %{x}<br>收益 %{z:.2f}%<extra></extra>",
            )
        )
        fig.update_layout(
            height=120 + 40 * len(monthly), margin=dict(l=10, r=10, t=10, b=10),
            template="plotly_white", xaxis_title="月份", yaxis_title="年份",
        )
        st.plotly_chart(fig, width="stretch")

    st.markdown("#### 最新一次调仓的因子明细")
    detail = res.latest_detail
    if detail is None or detail.empty:
        st.info("暂无因子明细。")
    else:
        rows = []
        for code, row in detail.iterrows():
            record = {"代码": code, "名称": sec_name(code)}
            for key in active:
                if key in detail.columns:
                    record[FACTOR_MAP[key]["name"]] = round(float(row[key]), 4)
            record["综合得分"] = round(float(row["composite"]), 4)
            record["权重"] = f"{float(row['weight']):.1%}" if row["weight"] > 0 else "-"
            record["入选"] = "✅" if row["selected"] else ""
            rows.append(record)
        table = pd.DataFrame(rows).sort_values("综合得分", ascending=False)
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption(
            "表中为各因子的原始数值（未标准化）。综合得分已经按权重和方向加总，"
            "所以得分排序才是最终选股依据。该表只反映最近一次调仓，历史各次的持仓请看下方调仓记录。"
        )

    st.markdown("#### 调仓记录")
    log = res.rebalance_log
    holdings = res.holdings_history if res.holdings_history is not None else pd.DataFrame()
    if not holdings.empty:
        display_rows = []
        prev = pd.Series(dtype=float)
        for day, row in holdings.iterrows():
            cur = row[row > 0].sort_values(ascending=False)
            if prev.empty:
                change = "建仓"
            else:
                bought = [c for c in cur.index if c not in prev.index]
                sold = [c for c in prev.index if c not in cur.index]
                parts = []
                if bought:
                    parts.append("买入 " + "、".join(sec_name(c) for c in bought))
                if sold:
                    parts.append("卖出 " + "、".join(sec_name(c) for c in sold))
                change = "；".join(parts) if parts else "无变化"
            display_rows.append(
                {
                    "调仓日": pd.Timestamp(day).date(),
                    "持仓明细": "、".join(f"{sec_name(c)} {w:.0%}" for c, w in cur.items()),
                    "持仓变动": change,
                    "持仓数量": len(cur),
                    "双边换手": log["双边换手"].iloc[len(display_rows)],
                    "交易成本": log["交易成本"].iloc[len(display_rows)],
                }
            )
            prev = cur
        log_display = pd.DataFrame(display_rows)
    else:
        log_display = log

    st.dataframe(
        log_display,
        hide_index=True,
        width="stretch",
        column_config={"持仓明细": st.column_config.TextColumn(width="large")},
    )
    st.caption(
        "持仓明细是每个调仓日收盘后确定的目标权重；如果相邻两次完全一样，"
        "「持仓变动」会显示「无变化」，此时双边换手为 0、也不产生交易成本。"
    )

    if not holdings.empty:
        with st.expander("按调仓日查看持仓权重矩阵（%）"):
            matrix = holdings.loc[:, holdings.max() > 0].copy()
            matrix.columns = [sec_name(c) for c in matrix.columns]
            matrix.index = [pd.Timestamp(d).date() for d in matrix.index]
            st.dataframe((matrix * 100).round(1), width="stretch")

    output = pd.DataFrame(
        {
            "日期": res.equity.index,
            "组合净值": res.equity.values,
            "基准净值": res.benchmark.values,
            "组合日收益": res.daily_returns.values,
            "组合回撤": res.drawdown.values,
        }
    )
    d1, d2 = st.columns(2)
    d1.download_button(
        "下载净值与回撤数据（CSV）",
        data=output.to_csv(index=False).encode("utf-8-sig"),
        file_name="backtest_equity.csv",
        mime="text/csv",
        width="stretch",
    )
    d2.download_button(
        "下载调仓记录（CSV）",
        data=log_display.to_csv(index=False).encode("utf-8-sig"),
        file_name="backtest_rebalance.csv",
        mime="text/csv",
        width="stretch",
    )

    st.caption(
        "提示：回测结果对区间、参数、交易成本都很敏感。建议换几段不同的时间区间、"
        "以及不同的因子权重再做对比，避免被单次回测结果误导。"
    )


# ------------------------------------------------------------------
# Tab 4：学习指南
# ------------------------------------------------------------------
def render_guide_tab() -> None:
    st.subheader("4. 学习指南")
    st.markdown(
        """
### 一、这个系统在做什么？

它把「选 ETF」变成一个可以重复执行的打分流程：

1. **候选池**：选定若干只 ETF（宽基、行业、跨境、商品、债券都可以混合）。
2. **因子打分**：每个调仓日，对每只 ETF 计算动量、波动率、回撤等指标。
3. **横截面标准化**：把同一因子在不同 ETF 之间比较，转换成 z-score（均值 0、标准差 1）。
4. **综合得分**：按你设置的权重和因子方向加权求和。
5. **调仓**：买入得分最高的 top_k 只，持有到下一个调仓日，并扣除交易成本。

> 因子投资的核心思想：不预测单只资产的涨跌，而是用一组可解释的指标，
> 在候选池里做「相对强弱」的排序。

### 二、因子速查表
        """
    )
    table = pd.DataFrame(
        [
            {
                "因子": f["name"],
                "衡量什么": f["description"],
                "计算方式": f["formula"],
                "方向": f["direction_text"],
                "默认参数": f"{f['param_name']}={f['default_param']}",
            }
            for f in FACTORS
        ]
    )
    st.dataframe(table, hide_index=True, width="stretch")

    st.markdown(
        """
### 三、绩效指标怎么看？

| 指标 | 含义 | 参考 |
| --- | --- | --- |
| 总收益率 | 整个回测区间的累计收益 | 与基准对比才有意义 |
| 年化收益率 | 折算成一年的复利收益 | 长期更可比 |
| 年化波动率 | 收益的波动程度 | 越低越稳，但过低可能收益也低 |
| 夏普比率 | 每承担 1 单位波动获得的超额收益 | >1 较好，>2 优秀（需结合区间） |
| 最大回撤 | 从历史高点最多跌了多少 | 衡量最难受的持有体验 |
| 卡玛比率 | 年化收益 ÷ 最大回撤 | 衡量「收益/痛苦」的性价比 |
| 日胜率 / 月胜率 | 上涨天数（月份）占比 | 高胜率不等于高收益 |
| 换手率 | 每次调仓买卖的比例 | 换手越高，成本影响越大 |

### 四、常见回测陷阱

- **未来函数**：用当天收盘后才知道的信息在当天买入。本系统在调仓日收盘计算得分，收益从下一交易日开始计算，避免这一点。
- **幸存者偏差**：只挑现在还存在、表现好的 ETF。本系统提供的是固定 ETF 池，真实研究中应包含已退市标的。
- **忽略交易成本**：频繁调仓会显著侵蚀收益，可把成本调到万 5~万 10 观察差异。
- **参数过拟合**：反复在同一段历史上调参数，很容易「拟合噪声」。建议留出一段没调过参的区间做样本外检验。
- **复权方式**：分红除权会造成价格跳空。本系统统一使用前复权价格。

### 五、建议的实验路线

1. 先**只用等权基准**跑一遍，看看候选池本身的表现。
2. 只保留**一个因子**（例如动量），理解它的作用。
3. 再逐步加入波动率、回撤等因子，观察收益与回撤的变化。
4. 调整**调仓周期**（5 / 20 / 60 日），体会「反应速度 vs 交易成本」的权衡。
5. 换不同**时间区间**（牛市、熊市、震荡市）分别回测，检验策略是否稳定。
6. 把交易成本从 0 调到万 10，看看策略是否还成立。

### 六、数据和运行说明

- 行情来源：**baostock**（免费、无需 token），日线、前复权，全部为真实行情。
- 数据下载：在「数据下载」页选择区间，可下载内置 ETF 池，也可以在「额外证券代码」里输入任意股票 / ETF / 指数代码（如 `600519`、`sh.510300`、`000001`），系统会自动查询并记录名称。
- 数据存储：项目根目录 `data/` 下的 CSV 文件（UTF-8），命名规则 `市场_代码.csv`，名称表为 `data/_meta.json`，可直接用 Excel / pandas 查看。
- 数据查看：在「数据查看」页选择标的和日期区间，图表、指标、明细表会同步刷新；「导出所选区间 CSV」会通过浏览器下载。
- 更新方式：再次下载会与已有历史取并集、整段重新拉取并替换；也可点「更新全部本地标的到最新」一键刷新。
- 回测只读取本地 CSV，不会在回测过程中联网。
- 本系统只做多、不做空，不考虑融资融券、停牌、涨跌停无法成交等细节。
        """
    )


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def main() -> None:
    init_state()
    codes = render_sidebar()

    st.title("📈 ETF 多因子量化学习系统")
    st.caption(
        "用可调权重的因子给 ETF / 股票打分并定期调仓，支持自选回测区间、参数说明、本地数据查看与导出。"
        "数据来源：baostock 真实行情。"
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📥 数据下载", "📊 数据查看", "🧩 因子配置", "🚀 策略回测", "📚 学习指南"]
    )
    with tab1:
        render_data_tab(codes)
    with tab2:
        render_view_tab()
    with tab3:
        render_factor_tab()
    with tab4:
        render_backtest_tab(codes)
    with tab5:
        render_guide_tab()


if __name__ == "__main__":
    main()
