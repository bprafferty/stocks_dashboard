import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.sql import text
import time

start_time = time.time()

engine = create_engine('postgresql://postgres:postgres@localhost:5432/stocks_dashboard')

with engine.connect() as con:
    file = open(r"queries\query_nasdaq_filter_symbol.sql")
    query = text(file.read())

    data = con.execute(query, x='GOOGL')
    results_as_dict = data.mappings().all()
    df = pd.DataFrame(results_as_dict)
    print(df.head())

print("---SQL %s seconds ---" % (time.time() - start_time))

start_time = time.time()

data = pd.read_csv(r'NASDAQ\back_data.csv')
data = data.rename(columns={'Unnamed: 0': 'Index'})
data = data.set_index('Index')
selection = data.loc[data['symbol'] == 'GOOGL']
print(selection.head())

print("---Python %s seconds ---" % (time.time() - start_time))