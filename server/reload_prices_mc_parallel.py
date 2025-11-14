import argparse
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from typing import Optional, Dict
from sqlalchemy import Table, Column, String, Float, DateTime, BigInteger, MetaData
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.engine.base import Engine
from sqlalchemy import create_engine, text
from ratelimit import limits, sleep_and_retry
from tenacity import retry, stop_after_attempt, wait_exponential
import concurrent.futures

# Import required connections
from app_imports import printnlog, parallelLoggingSetter, SQLALCHEMY_DATABASE_URI, FINNHUB_API_KEY, EOD_API_KEY
from ta_signals_mc_parallel import get_symbols_forwatchlist

logger = None

# Configuration
STOCK_LIST = ['AAPL', 'MSFT', 'GOOGL']  # Example list - modify as needed
BATCH_SIZE = 100

class APIConfig:
    FINNHUB = {
        'base_url': "https://finnhub.io/api/v1/stock/candle",
        'api_key': FINNHUB_API_KEY,
        'table_name': 'finnhub_stock_prices'
    }
    EOD = {
        'base_url': "https://eodhd.com/api/eod",
        'api_key': EOD_API_KEY,
        'table_name': 'eod_stock_prices'
    }

def initialize_config(price_source):
    config = {}
    config['PRICE_SOURCE'] = price_source.upper()
    if config['PRICE_SOURCE'] == 'FINNHUB':
        config['WATCHLIST_TABLENAME'] = "watchlist"
        config['price_table_name'] = 'finnhub_stock_prices'
    elif config['PRICE_SOURCE'] == 'EOD':
        config['WATCHLIST_TABLENAME'] = "eod_watchlist"
        config['price_table_name'] = 'eod_stock_prices'
    else:
        raise ValueError(f"Invalid PRICE_SOURCE: {config['PRICE_SOURCE']}")
    return config

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

@sleep_and_retry
@limits(calls=60, period=60)
def call_api_with_rate_limit(url: str, params: Dict = None) -> requests.Response:
    return requests.get(url, params=params, timeout=10)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def fetch_stock_data(symbol: str, start_time: int = int((datetime.utcnow() - timedelta(days=730)).timestamp()), end_time: int = int(datetime.utcnow().timestamp()), source: str = 'FINNHUB') -> Optional[pd.DataFrame]:
    try:
        if source == 'FINNHUB':
            return fetch_finnhub_data(symbol, start_time, end_time)
        else:
            start_date = datetime.utcfromtimestamp(start_time).strftime('%Y-%m-%d')
            end_date = datetime.utcfromtimestamp(end_time).strftime('%Y-%m-%d')
            return fetch_eod_data(symbol, start_date, end_date)
    except Exception as e:
        logger.error(f"Error fetching data for {symbol} from {source}: {str(e)}")
        return None

def fetch_finnhub_data(symbol: str, from_unix: int, to_unix: int) -> Optional[pd.DataFrame]:
    params = {
        'symbol': symbol,
        'resolution': 'D',
        'from': from_unix,
        'to': to_unix,
        'token': APIConfig.FINNHUB['api_key']
    }
    
    resp = call_api_with_rate_limit(APIConfig.FINNHUB['base_url'], params)
    resp.raise_for_status()
    data = resp.json()
    
    if data.get('s') != 'ok':
        logger.error(f"Finnhub API error for {symbol}: {data}")
        return None
        
    df = pd.DataFrame({
        'symbol': symbol,
        'date': pd.to_datetime(data['t'], unit='s', utc=True),
        'open': data['o'],
        'high': data['h'],
        'low': data['l'],
        'close': data['c'],
        'volume': data['v']
    })
    return df.dropna() if df.isnull().any().any() else df

def fetch_eod_data(symbol: str, from_date: str, to_date: str) -> Optional[pd.DataFrame]:
    url = f"{APIConfig.EOD['base_url']}/{symbol}"
    
    params = {
        'from': from_date,
        'to': to_date,
        'period': 'd',
        'api_token': APIConfig.EOD['api_key'],
        'fmt': 'json'
    }
    
    resp = call_api_with_rate_limit(url, params)
    resp.raise_for_status()
    data = resp.json()
    
    if not data:
        logger.error(f"EOD API error for {symbol}: No data returned")
        return None
        
    df = pd.DataFrame(data)
    df['symbol'] = symbol
    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={
        'adjusted_open': 'open',
        'adjusted_high': 'high',
        'adjusted_low': 'low'
    })
    if 'adjusted_close' in df.columns:
        df.drop('adjusted_close', axis=1, inplace=True)
    
    return df.dropna() if df.isnull().any().any() else df

def setup_database(engine: Engine, table_name: str) -> None:
    meta = MetaData()
    Table(
        table_name, meta,
        Column('symbol', String(16), primary_key=True),
        Column('date', DateTime(timezone=True), primary_key=True),
        Column('open', Float, nullable=False),
        Column('high', Float, nullable=False),
        Column('low', Float, nullable=False),
        Column('close', Float, nullable=False),
        Column('volume', BigInteger, nullable=False),
        mysql_charset='utf8mb4',
        extend_existing=True
    )
    meta.create_all(engine)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10), reraise=True)
def upsert_stock_data(df: pd.DataFrame, engine: Engine, table_name: str, batch_size: int) -> None:
    if df is None or df.empty:
        logger.warning("No data to upsert")
        return
        
    meta = MetaData()
    table = Table(
        table_name, meta,
        Column('symbol', String(16), primary_key=True),
        Column('date', DateTime(timezone=True), primary_key=True),
        Column('open', Float, nullable=False),
        Column('high', Float, nullable=False),
        Column('low', Float, nullable=False),
        Column('close', Float, nullable=False),
        Column('volume', BigInteger, nullable=False),
        extend_existing=True
    )
    records = df.to_dict(orient='records')
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        with engine.begin() as conn:
            stmt = mysql_insert(table).values(batch)
            stmt = stmt.on_duplicate_key_update(
                **{
                    'open': stmt.inserted['open'],
                    'high': stmt.inserted['high'],
                    'low': stmt.inserted['low'],
                    'close': stmt.inserted['close'],
                    'volume': stmt.inserted['volume']
                }
            )
            conn.execute(stmt)
    logger.info(f"Successfully upserted {len(records)} records")

def run_etl(source: str = 'FINNHUB') -> None:
    logger.info(f"Starting ETL job - {source} API")
    config = APIConfig.FINNHUB if source == 'FINNHUB' else APIConfig.EOD
    engine = create_engine(SQLALCHEMY_DATABASE_URI)
    setup_database(engine, config['table_name'])

    def process_symbol(symbol):
        local_engine = create_engine(SQLALCHEMY_DATABASE_URI)
        stock_start_time = time.time()
        df = fetch_stock_data(symbol, source=source)  # Always use default last 2 years
        time_taken = round(time.time() - stock_start_time, 4)
        printnlog(f"Processed stock {symbol} in {time_taken} seconds", my_logger=logger)
        if df is not None:
            upsert_stock_data(df, local_engine, config['table_name'], BATCH_SIZE)
        else:
            logger.info(f"None df for {symbol}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(process_symbol, STOCK_LIST))

    logger.info("ETL job completed successfully")
    
def get_symbol_prices(symbol: str, source: str = 'FINNHUB') -> Optional[pd.DataFrame]:
    logger.info(f"Fetching price data for {symbol} from {source}")
    config = initialize_config(price_source=source)
    engine = create_engine(SQLALCHEMY_DATABASE_URI)
    price_table_name = config['price_table_name']
    
    df = fetch_stock_data(symbol, source=source)  # Always use default last 2 years
    if df is None or df.empty:
        logger.warning(f"No data returned for {symbol}")
        return None
    
    upsert_stock_data(df, engine, price_table_name, BATCH_SIZE)
    
    with engine.connect() as conn:
        query = f"""
            SELECT * FROM {price_table_name}
            WHERE symbol = :symbol
            ORDER BY date DESC
        """
        df_final = pd.read_sql(
            text(query),
            conn,
            params={'symbol': symbol}
        )
    return df_final

def main():
    global STOCK_LIST, logger
    parser = argparse.ArgumentParser(description='Stock Data ETL')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-w', '--watchlist', help='Watchlist name')
    group.add_argument('-t', '--ticker', help='Ticker Symbol')
    parser.add_argument('-s', '--source', choices=['FINNHUB', 'EOD'], default='FINNHUB', help='Price source')
    args = parser.parse_args()

    if args.ticker:
        logger = parallelLoggingSetter('reload_prices_mc_parallel')
        logger.info(f"Processing single ticker: {args.ticker.upper()}")
        df = get_symbol_prices(args.ticker.upper(), source=args.source.upper())
        if df is not None:
            logger.info(f"Successfully retrieved data for {args.ticker.upper()}")
            return df
        else:
            logger.error(f"Failed to retrieve data for {args.ticker.upper()}")
            return None

    WATCHLIST = args.watchlist.upper()
    price_source = args.source.upper()

    logger = parallelLoggingSetter(f'reload_prices_mc_parallel_{WATCHLIST}')
    printnlog(f"ta_signals_prices_mc_{WATCHLIST}", my_logger=logger)

    config = initialize_config(price_source=price_source)

    symbols_df = get_symbols_forwatchlist(watchlist=WATCHLIST, config=config)
    printnlog(f"[Count of total Symbols for watchlist {WATCHLIST} = {len(symbols_df)}]", my_logger=logger)
    STOCK_LIST = symbols_df['symbol'].tolist()

    run_etl(source=price_source)

if __name__ == "__main__":
    main()
