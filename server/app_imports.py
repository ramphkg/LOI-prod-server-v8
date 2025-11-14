#Todo: Centralising DB table anme vs global variables in each file..
import configparser
import datetime
import logging
import sys
import os
from datetime import timedelta
import pytz
from sqlalchemy import create_engine

def getConfig(section: str, key: str):
    config = configparser.ConfigParser()
    config_filename = 'config.cfg'
    config_exists = os.path.isfile(config_filename)
    config_val = None

    if not config_exists:
        print(f"[Config file {config_filename} does not exist]")
    else:
        config.read(config_filename)
        section_upper = section.upper()
        section_lower = section.lower()
        if config.has_section(section_upper) or config.has_section(section_lower):
            actual_section = section_upper if config.has_section(section_upper) else section_lower
            for search_key in config[actual_section]:
                if search_key.upper() == key.upper():
                    config_val = config[actual_section][key]
                    return config_val

    print(config_val)
    return config_val

# Configuration parameters
ENV = getConfig(section='ENV', key='ENV_NAME')
Datadir_root = getConfig(section=ENV, key='Datadir_root')
Configdata_root = getConfig(section=ENV, key='Configdata_root')
SQLALCHEMY_DATABASE_URI = getConfig(section=ENV, key='SQLALCHEMY_DATABASE_URI')
FINNHUB_API_KEY = getConfig(section=ENV, key='FINNHUB_API_KEY')
EOD_API_KEY = getConfig(section=ENV, key='EOD_API_KEY')

# Date formats
format_str = '%Y-%m-%d'
format_str_withtime = "%Y-%m-%d %H:%M:%S"
format_str_utc = '%Y-%m-%dT%H:%M:%S%z'

# Current dates
today = datetime.date.today()
utcToday = datetime.datetime.utcnow().replace(tzinfo=pytz.utc).date()

def utcNow() -> datetime.datetime:
    return datetime.datetime.utcnow().replace(tzinfo=pytz.utc)

def utcNowTimestampInt() -> int:
    return int(datetime.datetime.utcnow().replace(tzinfo=pytz.utc).timestamp())

def daysdiff_utcTimestampInt(days: int) -> int:
    return int((datetime.datetime.utcnow().replace(tzinfo=pytz.utc) - timedelta(days=days)).timestamp())

strUtcToday = datetime.datetime.utcnow().replace(tzinfo=pytz.utc).strftime(format_str)

def strUtcNow() -> str:
    return datetime.datetime.utcnow().replace(tzinfo=pytz.utc).strftime(format_str_withtime)

def loggingSetter(module = None):
    logger =  logging.getLogger()
    if module is not None:
        logfilename = os.path.join(Datadir_root, "../log", f"{module}.log")
        logging.basicConfig(
            # level=logging.DEBUG,  # Set the minimum logging level
            level=logging.INFO,
            # format='[%(asctime)s %(levelname)s %(name)s: %(message)s]',  # Log message format
            # format="[%(filename)20s:%(lineno)3s - %(funcName)20s()] %(message)s",
            format='[%(asctime)s] [%(filename)s:%(lineno)s - %(funcName)s()] %(message)s',
            handlers=[
                logging.FileHandler(logfilename, mode='w'),  # Log to a file
                # logging.StreamHandler(sys.stdout)  # Optionally log to console
            ]
        )
        logger = logging.getLogger(module)
    return logger

# --- PARALLEL-SAFE LOGGING SETTER ---
def parallelLoggingSetter(module=None):
    """
    Returns a logger with a unique file handler per module/process.
    The log file name will include the process ID for uniqueness.
    """
    logger_name = module if module else "default"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        log_dir = os.path.join(Datadir_root, "../log")
        os.makedirs(log_dir, exist_ok=True)
        logfilename = os.path.join(log_dir, f"{logger_name}_{os.getpid()}.log")
        file_handler = logging.FileHandler(logfilename, mode='a')
        formatter = logging.Formatter('[%(asctime)s] [%(filename)s:%(lineno)s - %(funcName)s()] %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

def printnlog(log_str, my_logger = loggingSetter()):
    my_logger.info(log_str)
    print(log_str)

def getDbConnection():
    # Always create a new engine and connection per call
    engine = create_engine(SQLALCHEMY_DATABASE_URI, pool_recycle=3600)
    return engine.connect()

