import pandas as pd
from sqlalchemy import create_engine


data_paths = [r'NYSE\back_data.csv', r'NASDAQ\back_data.csv']
# engine = create_engine('postgresql://username:password@localhost:5432/mydatabase')
engine = create_engine('postgresql://postgres:postgres@localhost:5432/stocks_dashboard')

for path in data_paths:
    table_name = path.split('\\')[0]
    data = pd.read_csv(path)
    data = data.rename(columns={'Unnamed: 0': 'Index'})
    data = data.set_index('Index')
    data.to_sql(table_name, engine, index_label='Index')