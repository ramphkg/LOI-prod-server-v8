from sqlalchemy import text
import pandas as pd
import app_imports

SQL = """
SELECT RSIUturnTypeOld, COUNT(*) AS cnt
FROM finnhub_tas_listings
WHERE CountryName = 'USA'
GROUP BY RSIUturnTypeOld
ORDER BY cnt DESC
"""

def main():
    try:
        with app_imports.getDbConnection() as con:
            df = pd.read_sql(text(SQL), con)
        print(df)
        print('\nDistinct values:', df['RSIUturnTypeOld'].tolist())
        print('Total distinct:', len(df))
    except Exception as e:
        print('QUERY_ERROR:', e)

if __name__ == '__main__':
    main()
