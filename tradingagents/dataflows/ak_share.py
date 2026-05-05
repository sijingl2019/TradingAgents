"""AKShare-based stock data fetching.

CN A-shares and HK stocks use AKShare APIs.
US / other markets fall back to yfinance automatically.
"""

import os
import logging
from datetime import datetime
from typing import Annotated, Optional
import pandas as pd
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Symbol detection & conversion
# ═══════════════════════════════════════════════════════════════════════════

def _detect_market(symbol: str) -> str:
    """Return 'cn', 'hk', or 'us' (covers all non-CN/HK markets)."""
    s = symbol.upper().strip()
    if s.endswith(('.SS', '.SH', '.SZ')):
        return 'cn'
    if s.endswith('.HK'):
        return 'hk'
    return 'us'


def _to_ak_cn_symbol(symbol: str) -> str:
    """'600000.SS' → '600000',  '000001.SZ' → '000001'."""
    for suffix in ('.SS', '.SH', '.SZ'):
        if symbol.upper().endswith(suffix):
            return symbol[: -len(suffix)]
    return symbol


def _to_ak_hk_symbol(symbol: str) -> str:
    """'0700.HK' → '00700' (5-digit, zero-padded)."""
    code = symbol.upper().replace('.HK', '')
    return code.zfill(5)


def _ymd(date_str: str) -> str:
    """YYYY-MM-DD → YYYYMMDD (AKShare date format)."""
    return date_str.replace('-', '')


# ═══════════════════════════════════════════════════════════════════════════
# Private OHLCV fetch helpers (return Date/Open/High/Low/Close/Volume)
# ═══════════════════════════════════════════════════════════════════════════

_OHLCV_COL_MAP = {
    '日期': 'Date',
    '开盘': 'Open',
    '收盘': 'Close',
    '最高': 'High',
    '最低': 'Low',
    '成交量': 'Volume',
}

_OHLCV_KEEP = ('Date', 'Open', 'High', 'Low', 'Close', 'Volume')


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={k: v for k, v in _OHLCV_COL_MAP.items() if k in df.columns})
    return df[[c for c in _OHLCV_KEEP if c in df.columns]]


def _fetch_cn_ohlcv(ak_sym: str, start_ak: str, end_ak: str) -> pd.DataFrame:
    import akshare as ak
    df = ak.stock_zh_a_hist(
        symbol=ak_sym, period="daily",
        start_date=start_ak, end_date=end_ak, adjust="hfq",
    )
    return _normalize_ohlcv(df)


def _fetch_hk_ohlcv(ak_sym: str, start_ak: str, end_ak: str) -> pd.DataFrame:
    import akshare as ak
    df = ak.stock_hk_hist(
        symbol=ak_sym, period="daily",
        start_date=start_ak, end_date=end_ak, adjust="hfq",
    )
    return _normalize_ohlcv(df)


# ═══════════════════════════════════════════════════════════════════════════
# Public OHLCV API (mirrors get_YFin_data_online interface)
# ═══════════════════════════════════════════════════════════════════════════

def get_akshare_data_online(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Fetch OHLCV data via AKShare. Automatically falls back to yfinance for US/other markets."""
    market = _detect_market(symbol)

    try:
        if market == 'cn':
            data = _fetch_cn_ohlcv(_to_ak_cn_symbol(symbol), _ymd(start_date), _ymd(end_date))
        elif market == 'hk':
            data = _fetch_hk_ohlcv(_to_ak_hk_symbol(symbol), _ymd(start_date), _ymd(end_date))
        else:
            from .y_finance import get_YFin_data_online
            return get_YFin_data_online(symbol, start_date, end_date)

        if data is None or data.empty:
            return f"No data found for symbol '{symbol}' between {start_date} and {end_date}"

        for col in ('Open', 'High', 'Low', 'Close'):
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce').round(2)

        header = (
            f"# Stock data for {symbol.upper()} from {start_date} to {end_date}\n"
            f"# Total records: {len(data)}\n"
            f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        return header + data.to_csv(index=False)

    except Exception as e:
        logger.warning("AKShare OHLCV failed for %s: %s — falling back to yfinance", symbol, e)
        from .y_finance import get_YFin_data_online
        return get_YFin_data_online(symbol, start_date, end_date)


# ═══════════════════════════════════════════════════════════════════════════
# OHLCV loader for stockstats / technical indicators
# ═══════════════════════════════════════════════════════════════════════════

def load_ohlcv_akshare(symbol: str, curr_date: str) -> pd.DataFrame:
    """Fetch 5-year OHLCV with local cache, filtered to curr_date (look-ahead guard).

    CN/HK → AKShare.  US/other → yfinance (via _load_ohlcv_yfinance).
    """
    from .config import get_config

    market = _detect_market(symbol)

    if market == 'us':
        from .stockstats_utils import _load_ohlcv_yfinance
        return _load_ohlcv_yfinance(symbol, curr_date)

    config = get_config()
    curr_date_dt = pd.to_datetime(curr_date)
    today = pd.Timestamp.today()
    start = today - pd.DateOffset(years=5)
    start_ak = start.strftime("%Y%m%d")
    end_ak = today.strftime("%Y%m%d")

    os.makedirs(config["data_cache_dir"], exist_ok=True)
    cache_file = os.path.join(
        config["data_cache_dir"],
        f"{symbol}-AKShare-{start_ak}-{end_ak}.csv",
    )

    if os.path.exists(cache_file):
        data = pd.read_csv(cache_file, on_bad_lines="skip", encoding="utf-8")
    else:
        try:
            if market == 'cn':
                data = _fetch_cn_ohlcv(_to_ak_cn_symbol(symbol), start_ak, end_ak)
            else:  # hk
                data = _fetch_hk_ohlcv(_to_ak_hk_symbol(symbol), start_ak, end_ak)
            data.to_csv(cache_file, index=False, encoding="utf-8")
        except Exception as e:
            logger.warning("AKShare OHLCV cache fill failed for %s: %s — using yfinance", symbol, e)
            from .stockstats_utils import _load_ohlcv_yfinance
            return _load_ohlcv_yfinance(symbol, curr_date)

    from .stockstats_utils import _clean_dataframe
    data = _clean_dataframe(data)
    return data[data["Date"] <= curr_date_dt]


# ═══════════════════════════════════════════════════════════════════════════
# Financial statement helpers
# ═══════════════════════════════════════════════════════════════════════════

def _filter_by_date_rows(data: pd.DataFrame, date_col: str, curr_date: Optional[str]) -> pd.DataFrame:
    """Filter rows where date_col <= curr_date (AKShare financials layout: dates in rows)."""
    if curr_date is None or data.empty or date_col not in data.columns:
        return data
    cutoff = pd.Timestamp(curr_date)
    data = data.copy()
    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
    return data[data[date_col] <= cutoff]


def _get_ak_financial(
    ak_func_quarterly,
    ak_func_yearly,
    ak_sym: str,
    freq: str,
    curr_date: Optional[str],
    label: str,
    ticker: str,
) -> str:
    """Generic helper for balance-sheet / cashflow / income-statement."""
    try:
        if freq.lower() == "quarterly":
            data = ak_func_quarterly(symbol=ak_sym)
        else:
            data = ak_func_yearly(symbol=ak_sym)

        if data is None or data.empty:
            return f"No {label} data found for symbol '{ticker}'"

        # AKShare financial DFs: rows are reporting periods, first column is date
        # Try common date column names
        for date_col in ('报告期', 'REPORT_DATE', 'report_date'):
            if date_col in data.columns:
                data = _filter_by_date_rows(data, date_col, curr_date)
                break
        else:
            # Fallback: try yfinance-style column-as-date filter
            from .stockstats_utils import filter_financials_by_date
            data = filter_financials_by_date(data, curr_date)

        if data.empty:
            return f"No {label} data found for symbol '{ticker}' before {curr_date}"

        header = (
            f"# {label} for {ticker.upper()} ({freq})\n"
            f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        return header + data.to_csv(index=False)

    except Exception as e:
        return f"Error retrieving {label} for {ticker}: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════
# Fundamentals
# ═══════════════════════════════════════════════════════════════════════════

_CN_FIELD_MAP = {
    '股票代码': 'Stock Code',
    '股票简称': 'Company Name',
    '行业': 'Industry',
    '上市时间': 'Listing Date',
    '总股本': 'Total Shares',
    '流通股': 'Float Shares',
    '总市值': 'Market Cap',
    '流通市值': 'Float Market Cap',
}


def get_akshare_fundamentals(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "current date"] = None,
) -> str:
    """Get company fundamentals. Uses AKShare for CN stocks, yfinance for others."""
    market = _detect_market(ticker)

    if market not in ('cn', 'hk'):
        from .y_finance import get_fundamentals as _yf_fundamentals
        return _yf_fundamentals(ticker, curr_date)

    if market == 'hk':
        # AKShare HK fundamentals coverage is limited; yfinance is more complete
        from .y_finance import get_fundamentals as _yf_fundamentals
        return _yf_fundamentals(ticker, curr_date)

    try:
        import akshare as ak
        ak_sym = _to_ak_cn_symbol(ticker)

        # Basic company info (returns DataFrame: item | value)
        info_df = ak.stock_individual_info_em(symbol=ak_sym)
        if info_df is None or info_df.empty:
            return f"No fundamentals data found for symbol '{ticker}'"

        # Flatten to dict regardless of column names
        cols = list(info_df.columns)
        info_dict = dict(zip(info_df[cols[0]], info_df[cols[1]]))

        # Try to enrich with real-time quote data (PE, PB, etc.)
        try:
            spot = ak.stock_zh_a_spot_em()
            row = spot[spot['代码'] == ak_sym]
            if not row.empty:
                r = row.iloc[0]
                for cn_key, val_key in [
                    ('市盈率(动态)', '市盈率-动态'),
                    ('市净率', '市净率'),
                ]:
                    v = r.get(val_key)
                    if v is not None:
                        info_dict[cn_key] = v
        except Exception:
            pass

        lines = []
        for cn_key, en_label in _CN_FIELD_MAP.items():
            if cn_key in info_dict:
                lines.append(f"{en_label}: {info_dict[cn_key]}")

        # Append remaining fields as-is (may be in Chinese)
        mapped = set(_CN_FIELD_MAP.keys())
        for k, v in info_dict.items():
            if k not in mapped:
                lines.append(f"{k}: {v}")

        header = (
            f"# Company Fundamentals for {ticker.upper()}\n"
            f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        return header + "\n".join(lines)

    except Exception as e:
        logger.warning("AKShare fundamentals failed for %s: %s — using yfinance", ticker, e)
        from .y_finance import get_fundamentals as _yf_fundamentals
        return _yf_fundamentals(ticker, curr_date)


# ═══════════════════════════════════════════════════════════════════════════
# Balance Sheet / Cash Flow / Income Statement
# ═══════════════════════════════════════════════════════════════════════════

def get_akshare_balance_sheet(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    market = _detect_market(ticker)
    if market != 'cn':
        from .y_finance import get_balance_sheet as _yf_bs
        return _yf_bs(ticker, freq, curr_date)

    try:
        import akshare as ak
        ak_sym = _to_ak_cn_symbol(ticker)
        return _get_ak_financial(
            ak.stock_balance_sheet_by_report_em,
            ak.stock_balance_sheet_by_yearly_em,
            ak_sym, freq, curr_date, "Balance Sheet", ticker,
        )
    except Exception as e:
        logger.warning("AKShare balance sheet failed for %s: %s — using yfinance", ticker, e)
        from .y_finance import get_balance_sheet as _yf_bs
        return _yf_bs(ticker, freq, curr_date)


def get_akshare_cashflow(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    market = _detect_market(ticker)
    if market != 'cn':
        from .y_finance import get_cashflow as _yf_cf
        return _yf_cf(ticker, freq, curr_date)

    try:
        import akshare as ak
        ak_sym = _to_ak_cn_symbol(ticker)
        return _get_ak_financial(
            ak.stock_cash_flow_sheet_by_report_em,
            ak.stock_cash_flow_sheet_by_yearly_em,
            ak_sym, freq, curr_date, "Cash Flow", ticker,
        )
    except Exception as e:
        logger.warning("AKShare cashflow failed for %s: %s — using yfinance", ticker, e)
        from .y_finance import get_cashflow as _yf_cf
        return _yf_cf(ticker, freq, curr_date)


def get_akshare_income_statement(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    market = _detect_market(ticker)
    if market != 'cn':
        from .y_finance import get_income_statement as _yf_is
        return _yf_is(ticker, freq, curr_date)

    try:
        import akshare as ak
        ak_sym = _to_ak_cn_symbol(ticker)
        return _get_ak_financial(
            ak.stock_profit_sheet_by_report_em,
            ak.stock_profit_sheet_by_yearly_em,
            ak_sym, freq, curr_date, "Income Statement", ticker,
        )
    except Exception as e:
        logger.warning("AKShare income statement failed for %s: %s — using yfinance", ticker, e)
        from .y_finance import get_income_statement as _yf_is
        return _yf_is(ticker, freq, curr_date)


# ═══════════════════════════════════════════════════════════════════════════
# News
# ═══════════════════════════════════════════════════════════════════════════

def get_news_akshare(
    ticker: str,
    start_date: str,
    end_date: str,
) -> str:
    """Fetch stock-specific news via AKShare (CN/HK) or yfinance (US/other)."""
    market = _detect_market(ticker)

    if market not in ('cn', 'hk'):
        from .yfinance_news import get_news_yfinance
        return get_news_yfinance(ticker, start_date, end_date)

    try:
        import akshare as ak
        ak_sym = _to_ak_cn_symbol(ticker) if market == 'cn' else _to_ak_hk_symbol(ticker)
        news_df = ak.stock_news_em(symbol=ak_sym)

        if news_df is None or news_df.empty:
            return f"No news found for {ticker}"

        # Flexible column detection
        col_title = next((c for c in news_df.columns if '标题' in c or 'title' in c.lower()), None)
        col_content = next((c for c in news_df.columns if '内容' in c or 'content' in c.lower()), None)
        col_date = next((c for c in news_df.columns if '时间' in c or '日期' in c or 'date' in c.lower()), None)
        col_source = next((c for c in news_df.columns if '来源' in c or 'source' in c.lower()), None)
        col_link = next((c for c in news_df.columns if '链接' in c or 'link' in c.lower() or 'url' in c.lower()), None)

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        news_str = ""
        count = 0
        for _, row in news_df.iterrows():
            # Date filter
            if col_date:
                try:
                    pub_date = pd.to_datetime(row[col_date])
                    if not (start_dt <= pub_date.to_pydatetime().replace(tzinfo=None) <= end_dt + relativedelta(days=1)):
                        continue
                except Exception:
                    pass

            title = row[col_title] if col_title else "No title"
            source = row[col_source] if col_source else "Unknown"
            content = str(row[col_content])[:300] if col_content else ""
            link = row[col_link] if col_link else ""

            news_str += f"### {title} (source: {source})\n"
            if content:
                news_str += f"{content}\n"
            if link:
                news_str += f"Link: {link}\n"
            news_str += "\n"
            count += 1

        if count == 0:
            return f"No news found for {ticker} between {start_date} and {end_date}"

        return f"## {ticker} News, from {start_date} to {end_date}:\n\n{news_str}"

    except Exception as e:
        logger.warning("AKShare news failed for %s: %s — using yfinance", ticker, e)
        from .yfinance_news import get_news_yfinance
        return get_news_yfinance(ticker, start_date, end_date)


def get_global_news_akshare(
    curr_date: str,
    look_back_days: int = 7,
    limit: int = 10,
) -> str:
    """Fetch global/macro financial news via AKShare (falls back to yfinance)."""
    try:
        import akshare as ak
        curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        start_dt = curr_dt - relativedelta(days=look_back_days)
        start_date = start_dt.strftime("%Y-%m-%d")

        # Try CLS (财联社) telegraph news — comprehensive financial news feed
        news_df = ak.news_cls_telegraph(symbol="全部")

        if news_df is None or news_df.empty:
            raise ValueError("Empty CLS news response")

        col_title = next((c for c in news_df.columns if '标题' in c or 'title' in c.lower() or '内容' in c), None)
        col_date = next((c for c in news_df.columns if '时间' in c or '日期' in c or 'date' in c.lower()), None)

        news_str = ""
        count = 0
        seen = set()

        for _, row in news_df.iterrows():
            if count >= limit:
                break
            # Date filter (look-ahead guard)
            if col_date:
                try:
                    pub_dt = pd.to_datetime(row[col_date]).to_pydatetime().replace(tzinfo=None)
                    if pub_dt > curr_dt + relativedelta(days=1):
                        continue
                    if pub_dt < start_dt:
                        continue
                except Exception:
                    pass

            title = str(row[col_title]) if col_title else str(row.iloc[0])
            if title in seen:
                continue
            seen.add(title)

            news_str += f"### {title}\n\n"
            count += 1

        if count == 0:
            raise ValueError("No articles after filtering")

        return f"## Global Market News, from {start_date} to {curr_date}:\n\n{news_str}"

    except Exception as e:
        logger.warning("AKShare global news failed: %s — using yfinance", e)
        from .yfinance_news import get_global_news_yfinance
        return get_global_news_yfinance(curr_date, look_back_days, limit)


# ═══════════════════════════════════════════════════════════════════════════
# Insider Transactions
# ═══════════════════════════════════════════════════════════════════════════

def get_akshare_insider_transactions(
    ticker: Annotated[str, "ticker symbol of the company"],
) -> str:
    """Get insider (executive shareholding change) data via AKShare for CN stocks."""
    market = _detect_market(ticker)

    if market != 'cn':
        from .y_finance import get_insider_transactions as _yf_insider
        return _yf_insider(ticker)

    try:
        import akshare as ak
        ak_sym = _to_ak_cn_symbol(ticker)

        # East Money executive shareholding change data
        data = ak.stock_em_ggcg(symbol=ak_sym)

        if data is None or data.empty:
            return f"No insider transaction data found for symbol '{ticker}'"

        header = (
            f"# Insider Transactions (Executive Shareholding Changes) for {ticker.upper()}\n"
            f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        return header + data.to_csv(index=False)

    except Exception as e:
        logger.warning("AKShare insider transactions failed for %s: %s — using yfinance", ticker, e)
        from .y_finance import get_insider_transactions as _yf_insider
        return _yf_insider(ticker)


# ═══════════════════════════════════════════════════════════════════════════
# Price history helper for _fetch_returns (return calculation)
# ═══════════════════════════════════════════════════════════════════════════

# CN benchmark: CSI 300 Index; HK benchmark: Hang Seng Index
_CN_BENCHMARK = "000300"   # CSI 300 (沪深300)
_HK_BENCHMARK = "HSI"      # Hang Seng Index (yfinance ticker ^HSI)


def fetch_price_history(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Return a DataFrame with at least a 'Close' column for the given date range.

    Used by _fetch_returns in trading_graph.py.
    Returns None on failure.
    """
    market = _detect_market(symbol)
    start_ak = _ymd(start_date)
    end_ak = _ymd(end_date)

    try:
        if market == 'cn':
            # Determine if this is an index (6-digit starting with '000', '399', etc.)
            ak_sym = _to_ak_cn_symbol(symbol)
            if ak_sym.startswith(('000', '399', '8')):
                # Try as index first
                import akshare as ak
                try:
                    df = ak.stock_zh_index_daily_em(symbol=f"sh{ak_sym}" if ak_sym.startswith(('000', '1')) else f"sz{ak_sym}")
                    df = df.rename(columns={'date': 'Date', 'close': 'Close'})
                    df['Date'] = pd.to_datetime(df['Date'])
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date)
                    return df[(df['Date'] >= start_dt) & (df['Date'] <= end_dt)].reset_index(drop=True)
                except Exception:
                    pass
            # Regular A-share stock
            import akshare as ak
            df = ak.stock_zh_a_hist(symbol=ak_sym, period="daily",
                                     start_date=start_ak, end_date=end_ak, adjust="hfq")
            df = df.rename(columns={'日期': 'Date', '收盘': 'Close'})
            df['Date'] = pd.to_datetime(df['Date'])
            return df[['Date', 'Close']].reset_index(drop=True)

        elif market == 'hk':
            import akshare as ak
            ak_sym = _to_ak_hk_symbol(symbol)
            df = ak.stock_hk_hist(symbol=ak_sym, period="daily",
                                   start_date=start_ak, end_date=end_ak, adjust="hfq")
            df = df.rename(columns={'日期': 'Date', '收盘': 'Close'})
            df['Date'] = pd.to_datetime(df['Date'])
            return df[['Date', 'Close']].reset_index(drop=True)

        else:
            import yfinance as yf
            df = yf.Ticker(symbol).history(start=start_date, end=end_date)
            if df.empty:
                return None
            df = df.reset_index()[['Date', 'Close']]
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            return df

    except Exception as e:
        logger.warning("fetch_price_history failed for %s: %s", symbol, e)
        return None


def get_benchmark_symbol(market: str) -> str:
    """Return the benchmark ticker symbol for a given market."""
    return {
        'cn': _CN_BENCHMARK,
        'hk': _HK_BENCHMARK,
    }.get(market, 'SPY')
