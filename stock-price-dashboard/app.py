import streamlit as st
import datetime
import pandas as pd

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

# App title
st.markdown('''
# Stock Price Dashboard

- developped by [`@happy-jihye`](https://github.com/happy-jihye)
- To run this app, you must install the `yfinance, fbprophet, plotly` package.
---
''')


# Sidebar
start_date = st.sidebar.date_input("Start date", datetime.date(2021, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.date.today())
tickerSymbol = st.sidebar.text_input('Ticker Symbol', '035420.KS')
st.sidebar.markdown('Search for stock ticker symbol for any company in [Yahoo Finance](https://finance.yahoo.com/)')

tickerData = yf.Ticker(tickerSymbol)
tickerDf = tickerData.history(start=start_date, end=end_date)
ticker_data = yf.download(tickerSymbol, start_date, end_date)
ticker_data.reset_index(inplace=True)

# Ticker information
string_logo = '<img src=%s width = 25>' % tickerData.info['logo_url']
string_name = tickerData.info['longName']
st.markdown(f'## {string_logo}   {string_name}', unsafe_allow_html=True)

string_summary = tickerData.info['longBusinessSummary']
st.info(string_summary)

st.subheader('Raw data')
days = st.slider('Days', 1, 10, value=5)
st.dataframe(tickerDf.tail(days))

st.subheader(f'Stock Price Ticker')
st.line_chart(tickerDf.Close)

plotly = st.expander('plotly')

with plotly:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ticker_data['Date'], y=ticker_data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=ticker_data['Date'], y=ticker_data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


st.subheader(f'Stock Volume Ticker')
st.line_chart(tickerDf.Volume)


# Predicting Stock Prices Using Facebook’s Prophet Mode
st.subheader('Predicting Stock Prices Using Facebook’s Prophet Mode')

n_days = st.slider('Days of prediction:', 1, 365, value=31)

df_train = ticker_data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=n_days)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_days} days')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)