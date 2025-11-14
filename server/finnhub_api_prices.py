from app_imports import *
my_logger = loggingSetter()

import pandas as pd
import requests
import finnhub

finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

api_call_ctr = 1
api_call_throttle = 25000

def finnhub_get_today_price(symbol):
    today_price = None
    try:
        today_price = float(finnhub_client.quote(symbol)['c'])
    except Exception as e:
        my_logger.info('[get_finnhub_today_price - ERROR! - {}]'.format(e))
    return today_price

def finnhub_lastPriceDetails(symbol):
    result = {'TodayPrice':None,'High52': None, 'Low52':None, 'Pct2H52':None, 'PctfL52':None,'GEM_Rank':None,
              'Sector':None, 'marketCap':None}
    
    result ['TodayPrice'] = finnhub_get_today_price(symbol)
    try:
        metrics = finnhub_client.price_metrics(symbol)
        result ['High52'] = metrics['data']['52WeekHigh']
        result ['Low52'] = metrics['data']['52WeekLow']
        result ['Pct2H52']= (1 - ( result ['TodayPrice'] /  result ['High52'])) * 100
        result ['PctfL52']= (1 - ( result ['Low52'] /  result ['TodayPrice'])) * 100
    except Exception as e:
        my_logger.info('[finnhub_lastPriceDetails - ERROR! - {}]'.format(e))

    # sql = 'select symbol, GEM_Rank, gsector, marketCapitalization from finnhub_gem_listings where symbol = "{}";'.format(symbol)
    sql = 'select symbol, GEM_Rank, Sector, MarketCapitalizationMln from finnhub_gem_listings where symbol = "{}";'.format(symbol)
    gemranks_df = pd.read_sql(sql,con=getDbConnection())
    if isinstance(gemranks_df, pd.DataFrame) and len(gemranks_df) > 0:
        # Use .iat for scalar extraction from a single-row DataFrame/Series
        result['GEM_Rank'] = gemranks_df['GEM_Rank'].iat[0]
        result['Sector'] = gemranks_df['Sector'].iat[0]
        result['marketCap'] = gemranks_df['MarketCapitalizationMln'].iat[0]
    else:
        log_str = '[Could not load GEM_Ranks for symbol {}]'.format(symbol)
        my_logger.info(log_str)
    
    result['CountryName'] = 'USA' #Todo  change this to CountryName from finnhub_fundamental_listings -> finnhub_gem_listings  

    return result

def get_historic_prices_from_finnhub(symbol, days_diff=400):
    my_logger.info('[get_historic_prices_from_finnhub : symbol {}]'.format(symbol))
    df = None
    try :
        resp = finnhub_client.stock_candles(symbol, 'D', daysdiff_utcTimestampInt(days_diff), utcNowTimestampInt())
        df=pd.DataFrame(resp)
        df.rename(columns = {'o':'Open','h':'High','l':'Low','c':'Close','v':'Volume','t':'Date'}, inplace = True) 
        df['ScanDate'] = strUtcNow()
        #df['Symbol'] = symbol
        #cols = ['Date','Open','High','Low','Close','Volume','Symbol','ScanDate']
        cols = ['Date','Open','High','Low','Close','Volume','ScanDate']
        df['Date'] = pd.to_datetime(df['Date'], unit='s').dt.strftime(format_str)
        df=df[cols]

    except Exception as exception:
        my_logger.info ("[get_historic_prices_from_finnhub :  ERROR - {} ]".format(exception))
        raise
    return df

def finnhub_get_contractsDf(underlying , put_call_type, expiry_type, dte_start, dte_end):
    underlying = underlying.upper(); put_call_type = put_call_type.upper() ; expiry_type = expiry_type.upper()
    log_str = f'[underlying = "{underlying}", put_call_type = "{put_call_type}", dte_start = {dte_start}, dte_end = {dte_end}, expiry_type = "{expiry_type}"]'
    my_logger.info(log_str)
    
    if put_call_type not in ['PUT', 'CALL']:
        my_logger.info('[finnhub_get_contractsDf : Error in put_call_type {} ]'.format(put_call_type))
        return None

    restUrl = 'https://finnhub.io/api/v1/stock/option-chain?symbol={}&token={}'.format(underlying,FINNHUB_API_KEY)
    response = requests.get(restUrl)
    if response.status_code != 200:
        my_logger.error(f"finnhub_get_contractsDf Error: Received response code {response.status_code}")
        return None

    data=response.json()
    df = pd.DataFrame()
    for d in data['data']:
        # df = df.append(pd.DataFrame(d['options'][put_call_type])) # append Deprecated
        if put_call_type in d['options']:
            new_data = pd.DataFrame(d['options'][put_call_type])
            df = pd.concat([df, new_data], ignore_index=True)
        else:
            my_logger.error(f"Key '{put_call_type}' not found in options data.")
    log_str = '[response.status_code = {} , df rowcount = {} ]'.format(response.status_code, len(df))
    my_logger.info(log_str)
    my_logger.info(log_str)
    
    if not isinstance(df, pd.DataFrame) or  len(df) <= 0:
        my_logger.info("[Empty df.... ]")
        return None
            
    df.rename(columns = {'type':'putCall','strike':'strikePrice','contractName':'optionSymbol',
                                 'daysBeforeExpiration':'daysToExpiration','lastPrice':'last',}, inplace = True)      
    my_logger.info('expiry_type : |{}|'.format(expiry_type))

    if expiry_type in ['MONTHLY', 'WEEKLY'] :
        df = df[df['contractPeriod'] == expiry_type]
    
    df = df [ (df.daysToExpiration >= dte_start) & (df.daysToExpiration <= dte_end)]

    #June 15 '23 - WM request to exclude strikes with decimals other than 0.5
    #df['check']=df[np.where(df['strikePrice'].apply(lambda x: x.is_integer()), 0, 1)]
    df['check']=df['strikePrice'].apply(lambda x: x - int(x))
    df = df[(df.check == 0) | (df.check == 0.5)]
    df = df.loc[:, df.columns != 'check']

    return df


def get_finnhub_company_metrics(symbol):
    df = pd.DataFrame()
    try:
        #get ratios 
        df=pd.DataFrame(finnhub_client.company_basic_financials(symbol, 'metric'))
        #df.drop(df[df.score < 50].index, inplace=True)
        df.drop(['annual', 'quarterly'], inplace = True) #delete rows by index value
        df.drop('series', inplace=True, axis=1) #delete column by column name
        #df.insert(0, 'symbol', symbol) 
    except Exception as e:
        my_logger.info('[get_finnhub_company_metrics : ERROR! - {}]'.format(e))

    df = df.reset_index()[['index','metric']]
    df.rename(columns = {'index':'metric','metric':'value'}, inplace = True)

    return df

if __name__ == '__main__':
    my_logger.info("In __finnhub_api_prices Main")
    pass