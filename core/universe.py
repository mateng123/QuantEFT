"""ETF 池定义：代码、名称、类别与简介。

baostock 的证券代码格式为 "sh.510300" / "sz.159915"。
所有标的均使用「前复权」日线数据，便于计算收益率与回测。
"""

ETF_UNIVERSE = [
    # 宽基
    {"code": "sh.510050", "name": "上证50ETF", "category": "宽基", "desc": "跟踪上证50指数，大盘蓝筹、低估值风格"},
    {"code": "sh.510300", "name": "沪深300ETF", "category": "宽基", "desc": "跟踪沪深300指数，A股核心资产代表"},
    {"code": "sh.510500", "name": "中证500ETF", "category": "宽基", "desc": "跟踪中证500指数，中盘成长风格"},
    {"code": "sz.159915", "name": "创业板ETF", "category": "宽基", "desc": "跟踪创业板指数，高波动成长风格"},
    {"code": "sh.588000", "name": "科创50ETF", "category": "宽基", "desc": "跟踪科创50指数，硬科技成长风格"},
    # 行业 / 主题
    {"code": "sh.512880", "name": "证券ETF", "category": "行业", "desc": "跟踪证券公司指数，高弹性、强周期"},
    {"code": "sh.512690", "name": "酒ETF", "category": "行业", "desc": "跟踪中证酒指数，消费白马风格"},
    {"code": "sh.512480", "name": "半导体ETF", "category": "行业", "desc": "跟踪半导体指数，科技成长、波动大"},
    {"code": "sh.512010", "name": "医药ETF", "category": "行业", "desc": "跟踪沪深300医药指数，防御与成长兼备"},
    {"code": "sh.515030", "name": "新能源车ETF", "category": "行业", "desc": "跟踪新能源车指数，制造业成长风格"},
    {"code": "sh.515790", "name": "光伏ETF", "category": "行业", "desc": "跟踪光伏产业指数，新能源主题"},
    {"code": "sz.159949", "name": "创业板50ETF", "category": "行业", "desc": "跟踪创业板50指数，成长龙头集中"},
    # 跨境
    {"code": "sh.513100", "name": "纳指ETF", "category": "跨境", "desc": "跟踪纳斯达克100指数，美股科技龙头"},
    {"code": "sh.513050", "name": "中概互联网ETF", "category": "跨境", "desc": "跟踪中国互联网指数，港股/中概龙头"},
    # 商品 / 债券
    {"code": "sh.518880", "name": "黄金ETF", "category": "商品", "desc": "跟踪上海金价，避险与抗通胀资产"},
    {"code": "sh.511260", "name": "十年国债ETF", "category": "债券", "desc": "跟踪十年期国债，利率债低波动资产"},
]

ETF_MAP = {e["code"]: e for e in ETF_UNIVERSE}

# 默认选中的 ETF（覆盖宽基、行业、跨境、商品，便于观察轮动效果）
DEFAULT_CODES = [
    "sh.510300",
    "sh.510500",
    "sz.159915",
    "sh.512880",
    "sh.512480",
    "sh.518880",
    "sh.513100",
    "sh.511260",
]


def etf_label(code: str) -> str:
    """把代码转换成可读标签，例如 'sh.510300 沪深300ETF'。"""
    item = ETF_MAP.get(code)
    return f"{code} {item['name']}" if item else code


def etf_name(code: str) -> str:
    item = ETF_MAP.get(code)
    return item["name"] if item else code
