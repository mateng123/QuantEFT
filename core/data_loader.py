"""数据层：baostock 真实行情下载、本地 CSV 缓存与读取。

设计要点：
1. 数据落地到项目根目录的 data/ 文件夹（每只 ETF 一个 CSV），回测只读本地缓存。
2. 覆权方式默认「前复权」(adjustflag=2)，回测时价格序列连续，适合计算收益与因子。
3. baostock 需要先登录再查询，批量下载时复用同一个会话。
4. 前复权价格以最新交易日为基准，分红除权后会整体变化，因此更新时整段重拉并覆盖。
"""

from __future__ import annotations

import glob
import json
import os
import re
from contextlib import contextmanager
from typing import Callable, Dict, Iterable, Optional, Tuple

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
META_PATH = os.path.join(DATA_DIR, "_meta.json")

FIELDS = ["date", "code", "open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"]
NUMERIC_FIELDS = ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"]


def ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def cache_path(code: str) -> str:
    ensure_data_dir()
    return os.path.join(DATA_DIR, f"{code.replace('.', '_')}.csv")


# ------------------------------------------------------------------
# 证券代码、名称与本地文件清单
# ------------------------------------------------------------------
def normalize_code(raw: str) -> Optional[str]:
    """把用户输入的代码统一成 baostock 格式（sh.600000 / sz.000001）。

    支持：sh.600000、sh600000、600000、600000.SH 等常见写法。
    纯 6 位数字时，5/6/9 开头归为上交所，其余归为深交所。
    """
    s = (raw or "").strip().lower().replace(" ", "")
    if not s:
        return None
    m = re.fullmatch(r"(sh|sz)[.]?(\d{6})", s)
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    m = re.fullmatch(r"(\d{6})[.](sh|sz)", s)
    if m:
        return f"{m.group(2)}.{m.group(1)}"
    if re.fullmatch(r"\d{6}", s):
        return f"{'sh' if s[0] in '569' else 'sz'}.{s}"
    return None


def load_meta() -> Dict[str, dict]:
    """读取本地证券名称表（data/_meta.json）。"""
    ensure_data_dir()
    if not os.path.exists(META_PATH):
        return {}
    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_meta(meta: Dict[str, dict]) -> None:
    ensure_data_dir()
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def upsert_meta(code: str, name: Optional[str] = None, kind: Optional[str] = None) -> None:
    meta = load_meta()
    item = meta.get(code, {})
    if name:
        item["name"] = name
    if kind:
        item["kind"] = kind
    meta[code] = item
    save_meta(meta)


def local_codes() -> list:
    """列出 data/ 下已有 CSV 的全部证券代码。"""
    ensure_data_dir()
    codes = []
    for path in glob.glob(os.path.join(DATA_DIR, "*.csv")):
        base = os.path.basename(path)[:-4]
        if "_" not in base:
            continue
        market, num = base.split("_", 1)
        if market in ("sh", "sz") and len(num) == 6 and num.isdigit():
            codes.append(f"{market}.{num}")
    return sorted(codes)


def _kind_label(type_code: Optional[str]) -> Optional[str]:
    return {
        "1": "股票",
        "2": "指数",
        "3": "其它",
        "4": "可转债",
        "5": "ETF",
    }.get(str(type_code or ""), None)


def _fetch_name(bs, code: str) -> None:
    """从 baostock 查询证券名称并写入本地名称表。"""
    try:
        rs = bs.query_stock_basic(code=code)
        if rs.error_code == "0" and rs.next():
            row = dict(zip(rs.fields, rs.get_row_data()))
            upsert_meta(code, name=row.get("code_name") or None, kind=_kind_label(row.get("type")))
    except Exception:
        pass


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """统一列类型、去重、按日期排序。"""
    if df is None or df.empty:
        return pd.DataFrame(columns=FIELDS)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    for col in NUMERIC_FIELDS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"])
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return df


@contextmanager
def baostock_session():
    """baostock 登录/登出上下文，批量下载时只登录一次。"""
    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock 登录失败：{lg.error_msg}")
    try:
        yield bs
    finally:
        bs.logout()


def fetch_baostock(code: str, start: str, end: str, adjustflag: str = "2") -> pd.DataFrame:
    """从 baostock 拉取单只 ETF 的日线数据（不含本地缓存合并）。"""
    import baostock as bs

    rs = bs.query_history_k_data_plus(
        code,
        ",".join(FIELDS),
        start_date=start,
        end_date=end,
        frequency="d",
        adjustflag=adjustflag,
    )
    if rs.error_code != "0":
        raise RuntimeError(f"{code} 下载失败：{rs.error_msg}")

    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return pd.DataFrame(columns=FIELDS)

    df = pd.DataFrame(rows, columns=rs.fields)
    df["code"] = code
    return _normalize(df)


def read_cache(code: str) -> Optional[pd.DataFrame]:
    path = cache_path(code)
    if not os.path.exists(path):
        return None
    try:
        return _normalize(pd.read_csv(path))
    except Exception:
        return None


def write_cache(code: str, df: pd.DataFrame, replace: bool = False) -> pd.DataFrame:
    """写入本地 CSV 缓存。

    replace=True 时整体覆盖；否则与旧缓存合并（同日期以新数据为准）。
    前复权价格以最新交易日为基准，分红除权后整段历史都会变化，
    所以行情下载使用 replace=True，避免新旧两套复权基准拼接出价格断层。
    """
    if not replace:
        old = read_cache(code)
        if old is not None and not old.empty:
            df = pd.concat([df, old], ignore_index=True)
    df = _normalize(df)
    df.to_csv(cache_path(code), index=False, encoding="utf-8-sig")
    return df


def download_many(
    codes: Iterable[str],
    start: str,
    end: str,
    adjustflag: str = "2",
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """批量下载并写入缓存，返回（成功字典, 错误字典）。"""
    codes = list(codes)
    results: Dict[str, pd.DataFrame] = {}
    errors: Dict[str, str] = {}

    with baostock_session() as bs:
        for i, code in enumerate(codes, start=1):
            try:
                _fetch_name(bs, code)

                # 前复权价格会随之后的除权分红整体变化，因此下载区间与已有缓存取并集后
                # 整段重新拉取并覆盖，保证缓存里只有一套复权基准。
                fetch_start = pd.Timestamp(start)
                fetch_end = pd.Timestamp(end)
                old = read_cache(code)
                if old is not None and not old.empty:
                    fetch_start = min(fetch_start, old["date"].min())
                    fetch_end = max(fetch_end, old["date"].max())

                rs = bs.query_history_k_data_plus(
                    code,
                    ",".join(FIELDS),
                    start_date=fetch_start.strftime("%Y-%m-%d"),
                    end_date=fetch_end.strftime("%Y-%m-%d"),
                    frequency="d",
                    adjustflag=adjustflag,
                )
                if rs.error_code != "0":
                    errors[code] = rs.error_msg
                else:
                    rows = []
                    while rs.next():
                        rows.append(rs.get_row_data())
                    if not rows:
                        errors[code] = "该区间没有行情数据"
                    else:
                        df = pd.DataFrame(rows, columns=rs.fields)
                        df["code"] = code
                        results[code] = write_cache(code, df, replace=True)
            except Exception as exc:  # 单只失败不影响其它标的
                errors[code] = str(exc)

            if progress_cb is not None:
                progress_cb(i, len(codes), code)

    return results, errors


def load_series_map(
    codes: Iterable[str],
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> Dict[str, pd.DataFrame]:
    """读取多只 ETF 的缓存数据，按时间区间裁剪。"""
    result: Dict[str, pd.DataFrame] = {}
    for code in codes:
        df = read_cache(code)
        if df is None or df.empty:
            continue
        if start is not None:
            df = df[df["date"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["date"] <= pd.Timestamp(end)]
        if not df.empty:
            result[code] = df.reset_index(drop=True)
    return result


def coverage(code: str) -> Dict[str, object]:
    """返回某只 ETF 的缓存概况，用于界面展示。"""
    df = read_cache(code)
    if df is None or df.empty:
        return {"rows": 0, "start": None, "end": None, "updated": None}
    path = cache_path(code)
    updated = pd.Timestamp(os.path.getmtime(path), unit="s", tz="Asia/Shanghai")
    return {
        "rows": len(df),
        "start": df["date"].min().date(),
        "end": df["date"].max().date(),
        "updated": updated.strftime("%Y-%m-%d %H:%M"),
    }
