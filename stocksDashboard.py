import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import predict
from sqlalchemy import create_engine
from sqlalchemy.sql import text

@st.cache
def filter_symbol(page, symbol):
    if page == 'NYSE':
        file_name = r'queries\query_nyse_filter_symbol.sql'
    else:
        file_name = r'queries\query_nasdaq_filter_symbol.sql'
    
    engine = create_engine('postgresql://postgres:postgres@localhost:5432/stocks_dashboard')

    with engine.connect() as con:
        file = open(file_name)
        query = text(file.read())

        data = con.execute(query, x=symbol)
        results_as_dict = data.mappings().all()
        df = pd.DataFrame(results_as_dict)
        return df

@st.cache
def get_symbols(page):
    if page == 'NYSE':
        file_name = r'queries\query_nyse_symbols.sql'
    else:
        file_name = r'queries\query_nasdaq_symbols.sql'
    engine = create_engine('postgresql://postgres:postgres@localhost:5432/stocks_dashboard')

    with engine.connect() as con:
        file = open(file_name)
        query = text(file.read())

        data = con.execute(query)
        symbols = [x[0] for x in data]
        return symbols

def view(page):
    html_str = f'''
    <h1 class="a">{page} Dashboard</h1> 
    '''
    st.markdown(html_str, unsafe_allow_html=True)

    symbols = get_symbols(page)

    symbol = st.sidebar.selectbox('Select a Stock', symbols)

    result_df, prediction_array = predict.get_prediction(symbol, page)

    selection = filter_symbol(page, symbol)

    selection = selection.iloc[10:]

    selection['Predicted Price'] = prediction_array

    price_filter = st.sidebar.selectbox('Choose Price to Display', ('Closing', 'High', 'Low', 'Opening'))

    price_lookup = {'Closing': 'close', 'High': 'high', 'Low': 'low', 'Opening': 'open'}

    if price_filter in price_lookup:
        # SELECT date, :inputPrice
        # FROM selection
        selection = selection[['date', price_lookup[price_filter], 'Predicted Price']]
    
    date_filter = st.sidebar.selectbox('Filter Date', ('1 Week', '1 Month', '3 Months', '6 Months', '1 Year', '2 Years', '3 Years'))

    date_lookup = {'1 Week': -5, '1 Month': -21, '3 Months': -63, '6 Months': -130, '1 Year': -261, '2 Years': -522}

    if date_filter in date_lookup:
        # SELECT *
        # FROM selection
        # ORDER BY date DESC
        # LIMIT :inputDate
        selection = selection.iloc[date_lookup[date_filter]:]

    current_price = selection.iloc[-1]
    first_price = selection.iloc[0]

    current_price = current_price[price_lookup[price_filter]].round(2)
    first_price = first_price[price_lookup[price_filter]].round(2)

    price_delta = (current_price - first_price).round(2)
    if price_delta < 0:
        price_delta_str = price_delta * -1
        price_delta_str = '- $' + str(price_delta_str)
    else:
        price_delta_str = '+ $' + str(price_delta)
    if price_delta_str[-3] != '.':
        price_delta_str += '0'

    price_delta_percent = ((current_price / first_price) * 100).round(2)
    if price_delta_percent < 100:
        price_delta_percent = (100 - price_delta_percent).round(2)
        price_delta_percent_str = '- ' + str(price_delta_percent) + '%'
    else:
        price_delta_percent_str = '+ ' + str(price_delta_percent) + '%'

    current_price_str = '$' + str(current_price)
    if current_price_str[-3] != '.':
        current_price_str += '0'

    predicted_price = result_df['Predicted Current Price'].item()
    predicted_price_str = '$' + str(predicted_price)
    if predicted_price_str[-3] != '.':
        predicted_price_str += '0'
    
    mae = str(result_df['Mean Absolute Error'].item())
    r2 = str(result_df['Coefficient of Determination'].item())

    stats = pd.DataFrame([[current_price_str, price_delta_str, price_delta_percent_str]], columns=['Current Price', 'Change in Value', 'Percent Change'])
    prediction_metrics = pd.DataFrame([[predicted_price_str,mae, r2]], columns=['Predicted Current Price', 'Mean Absolute Error', 'Coefficient of Determination'])


    html_str = f'''
    <h3 class="a">Selection: {date_filter} of {symbol}'s {price_filter} Prices on {page}</h3> 
    '''
    st.sidebar.markdown(html_str, unsafe_allow_html=True)

    def color_price_delta(val):
        color = 'red' if val[0] == '-' else 'green'
        return f'color: {color}'

    st.table(stats.style.applymap(color_price_delta, subset=['Change in Value', 'Percent Change']))
    st.table(prediction_metrics)

    #df = selection[[ 'Predicted Price', price_lookup[price_filter]]]

    if stats['Change in Value'].item()[0] == '-':

        chart = alt.Chart(selection).mark_line().encode(
            x=alt.X('date:T', axis=alt.Axis(grid=False, title='Date')),
            y=alt.Y(price_lookup[price_filter], axis=alt.Axis(format='$f', title='Price')),
            color=alt.value('red')
        ).properties(
            width='container',
            height=500
        )
        #alt.layer()
    else:
        chart = alt.Chart(selection).mark_line().encode(
            x=alt.X('date:T', axis=alt.Axis(grid=False, title='Date')),
            y=alt.Y(price_lookup[price_filter], axis=alt.Axis(format='$f', title='Price')),
            color=alt.value('green')
        ).properties(
            width='container',
            height=500
        )
        #alt.layer( *[chart.encode(y=col) for col in df.columns], data=df)

    st.altair_chart(chart, use_container_width=True)

st.sidebar.header('User Input Features')
page = st.sidebar.radio("Select Stock Exchange: ", ('NYSE', 'NASDAQ'))

view(page)