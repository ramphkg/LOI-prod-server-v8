#Todo: Centralising DB table anme vs global variables in each file..
from app_imports import *
my_logger = loggingSetter()

import pandas as pd
from eod import EodHistoricalData
import requests

ENV = getConfig(section='ENV',key='ENV_NAME')
EOD_API_KEY = getConfig(section=ENV,key='EOD_API_KEY')

eod_client = EodHistoricalData(EOD_API_KEY)

api_call_ctr = 1
api_call_throttle = 25000

def check_eod_api_usage_exceeded() -> bool:
    """
    Checks if the number of API requests used has reached or exceeded the specified threshold.
    """
    try:
        api_url = f"https://eodhd.com/api/user?api_token={EOD_API_KEY}"
        # Make the GET request to the API endpoint
        response = requests.get(api_url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse the JSON response
        data = response.json()
        
        # Extract the number of API requests used
        api_requests = data.get('apiRequests')
        if api_requests is None:
            my_logger.info("API response does not contain 'apiRequests' field.")
            return False

        dailyRateLimit = data.get('dailyRateLimit')
        if dailyRateLimit is None:
            my_logger.info("API response does not contain 'dailyRateLimit' field.")
            return False
        
        # Log the current usage
        my_logger.info(f"API Requests Used: {api_requests} out of {dailyRateLimit} daily limit.")
        
        # Check if the usage has reached or exceeded the threshold
        if api_requests >= dailyRateLimit - 5000: #Leave 5000 behind
            printnlog(
                f"API usage has reached the threshold of {dailyRateLimit - 5000}/{dailyRateLimit} requests."
            )
            return True
        else:
            remaining_requests = dailyRateLimit - api_requests
            printnlog(
                f"API usage below threshold. {remaining_requests} requests remaining for today."
            )
            return False
    
    except requests.RequestException as e:
        my_logger.error(f"Error while making API request: {e}")
        return False
    except ValueError as e:
        my_logger.error(f"Error parsing JSON response: {e}")
        return False
    except Exception as e:
        my_logger.error(f"An unexpected error occurred: {e}")
        return False


def get_db_closing_prices(symbol, start_date=None, end_date=None):
    # Establish database connection
    con = getDbConnection()  # Assuming this function exists to get the connection

    # Construct the SQL query
    query = f"""
    SELECT Date, Open, High, Low, Close, Volume
    FROM eod_closing_prices
    WHERE symbol = '{symbol}'
    """

    # Add date range filter if provided
    if start_date:
        query += f" AND Date >= '{start_date}'"
    if end_date:
        query += f" AND Date <= '{end_date}'"

    query += " ORDER BY Date"

    # Execute the query and load results into a DataFrame
    df = pd.read_sql(query, con)

    # Set 'Date' as the index
    # df.set_index('Date', inplace=True)

    # Close the database connection
    # con.close()

    return df

def get_eod_stock_code(underlying, country=None): #Todo: Not used anywhere. Remove?
    stock_code=underlying.upper()
    stock_code_short = stock_code
    if country is None:
        country = 'USA'
    if underlying.endswith('.HK'):
        stock_code = underlying.split('.')[0]
        stock_code_short = stock_code.lstrip('0')
        stock_code = stock_code.zfill(4) + '.HK'
        country = 'Hong Kong'
    elif underlying.startswith('HKG:'):
        stock_code = underlying.split(':')[1].lstrip('0')
        stock_code_short = stock_code.lstrip('0')
        stock_code = stock_code.zfill(4) + '.HK'
        country = 'Hong Kong'
    elif underlying.isdecimal():
        stock_code = underlying.lstrip('0')
        stock_code_short = stock_code
        country = 'Hong Kong'
        stock_code = stock_code.zfill(4) + '.HK'
    elif underlying.isalpha():
        stock_code_short = stock_code
        stock_code = stock_code + '.US'
        country = 'USA'
    return {'eod_stock_code':stock_code, 'stock_code_short':stock_code_short,'country':country}

def get_historic_prices_from_eod(symbol, days_diff=365):
    #symbol examples : AAPL, 3MINDIA.NSE, 0001.HK
    print('[get_historic_prices_from_eod : symbol {}]'.format(symbol))
    df = None
    try :
        fromDate = (today - datetime.timedelta(days=days_diff)).strftime(format_str)
        resp = eod_client.get_prices_eod(symbol, period='d', order='a', from_=fromDate)
        df=pd.DataFrame(resp)
    except Exception as exception:
        print ("[get_historic_prices_from_eod :  ERROR - {} ]".format(exception))
        raise

    df.rename(columns = {'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume','date':'Date'}, inplace = True) 
    cols = ['Date','Open','High','Low','Close','Volume']
    
    return df[cols]
  
def eod_get_contractsDf(underlying , put_call_type, from_date, to_date, expiry_type):
    put_call_type = put_call_type.upper() ; expiry_type = expiry_type.upper()
    my_logger.info('[underlying : {}, put_call_type : {}, from_date : {}, to_date : {}, expiry_type : {}]'.format(underlying, put_call_type, from_date, 
            to_date, expiry_type))
    
    if put_call_type not in ['PUT', 'CALL']:
        my_logger.info('[eod_get_contractsDf : Error in put_call_type {} ]'.format(put_call_type))
        return None
    
    resp = eod_client.get_stock_options(underlying, from_=from_date, to=to_date) # Documnetation : https://github.com/LautaroParada/eod-data
    options_df = pd.DataFrame()
    for data in resp['data']:
        if put_call_type == 'PUT':
            #options_df = options_df.append(pd.json_normalize(data['options']['PUT']))
            options_df = pd.concat([options_df, pd.json_normalize(data['options']['PUT'])], ignore_index=True)
        else:
            #options_df = options_df.append(pd.json_normalize(data['options']['CALL']))
            options_df = pd.concat([options_df, pd.json_normalize(data['options']['CALL'])], ignore_index=True)
            
    if not isinstance(options_df, pd.DataFrame) or  len(options_df) <= 0:
        my_logger.info("[Empty options_df.... ]")
        return None
            
    options_df.rename(columns = {'type':'putCall','strike':'strikePrice','contractName':'optionSymbol',
                                 'daysBeforeExpiration':'daysToExpiration','lastPrice':'last',}, inplace = True)

    if expiry_type in ['MONTHLY', 'WEEKLY'] :
        options_df = options_df[options_df.contractPeriod == expiry_type]

    #June 15 '23 - WM request to exclude strikes with decimals other than 0.5
    #options_df['check']=options_df[np.where(options_df['strikePrice'].apply(lambda x: x.is_integer()), 0, 1)]
    options_df['check']=options_df['strikePrice'].apply(lambda x: x - int(x))
    options_df = options_df[(options_df.check == 0) | (options_df.check == 0.5)]
    options_df = options_df.loc[:, options_df.columns != 'check']
        
    return options_df

def get_sectorType(Sector):
    SectorMapping = {
        'Basic Materials' : 'Cyclical',
        'Consumer Cyclical' : 'Cyclical',
        'Financial Services' : 'Cyclical',
        'Real Estate': 'Cyclical',
        'Consumer Defensive' : 'Defensive',
        'Healthcare' : 'Defensive',
        'Utilities' : 'Defensive',
        'Communication Services' : 'Sensitive',
        'Energy' : 'Sensitive',
        'Industrials' : 'Sensitive',
        'Technology': 'Sensitive'
    }
    
    if Sector in SectorMapping.keys():
        return SectorMapping[Sector]
    else:
        my_logger.info("eod_lastPriceDetails : obtain SectorType - Error unknown Sector {} ".format(Sector))
        return Sector

def eod_lastPriceDetails(underlying):
    # Initialize all variables to None
    Pct2H52 = None
    PctfL52 = None
    High52 = None
    Low52 = None
    TodayPrice = None
    PE = None
    EPS = None
    Sector = None
    TargetPrice = None
    Rating = None
    SectorType = None
    marketCap = None
    GEM_Rank = None
    CountryName = None

    try:
        ticker = eod_client.get_fundamental_equity(underlying)
        High52 = ticker['Technicals'].get('52WeekHigh', None)
        Low52 = ticker['Technicals'].get('52WeekLow', None)
        EPS = ticker['Highlights'].get('EarningsShare', None)
        Sector = ticker['General'].get('Sector', None)
        PE = ticker['Valuation'].get('TrailingPE', None)
        marketCap = ticker['Highlights'].get('MarketCapitalizationMln', None)
    except Exception as e:
        my_logger.info(f'[eod_lastPriceDetails: {underlying} : ERROR! from eod_client.get_fundamental_equity - {e}]')

    try:
        price_data = eod_client.get_prices_live(underlying)
        TodayPrice = price_data.get('close', None)
        
        # Ensure High52 and TodayPrice are not None and are numbers before calculation
        if High52 is not None and TodayPrice is not None and isinstance(High52, (int, float)) and isinstance(TodayPrice, (int, float)):
            Pct2H52 = (1 - (TodayPrice / High52)) * 100
        
        if Low52 is not None and TodayPrice is not None and isinstance(Low52, (int, float)) and isinstance(TodayPrice, (int, float)):
            PctfL52 = (1 - (Low52 / TodayPrice)) * 100
        
        # Handle missing AnalystRatings gracefully
        analyst_ratings = ticker.get('AnalystRatings', {})
        Rating = analyst_ratings.get('Rating', None)
        TargetPrice = analyst_ratings.get('TargetPrice', None)
    except Exception as e:
        my_logger.info(f'[eod_lastPriceDetails: {underlying} : ERROR! from eod_client.get_prices_live - {e}]')

    # Handle unknown SectorType
    SectorType = get_sectorType(Sector) if Sector else 'Unknown'

    try:
        db_details = db_gemDetails(underlying)
        GEM_Rank = db_details.get('GEM_Rank', None)
        CountryName = db_details.get('CountryName', None)
    except Exception as e:
        my_logger.info(f"[eod_lastPriceDetails : {underlying} : Obtain GEM_Rank from db_lastPriceDetails - ERROR! {e}]")

    details = {
        'TodayPrice': TodayPrice,
        'High52': High52,
        'Low52': Low52,
        'Pct2H52': Pct2H52,
        'PctfL52': PctfL52,
        'PE': PE,
        'EPS': EPS,
        'Sector': Sector,
        'SectorType': SectorType,
        'Rating': Rating,
        'TargetPrice': TargetPrice,
        'marketCap': marketCap,
        'GEM_Rank': GEM_Rank,
        'CountryName': CountryName
    }

    return details

def db_gemDetails(underlying, country=None):
    sql = "select GEM_Rank, CountryName from eod_gem_listings where symbol = '{}';".format(underlying)
    df=pd.read_sql(sql,con=getDbConnection())
    GEM_Rank = None
    CountryName = None
    if isinstance(df, pd.DataFrame) and len(df) > 0:
        GEM_Rank = df['GEM_Rank'].iat[0]
        CountryName = df['CountryName'].iat[0]
    return {'GEM_Rank':GEM_Rank, 'CountryName':CountryName}
    
