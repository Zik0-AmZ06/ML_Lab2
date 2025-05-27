import yfinance as yf
import streamlit as st
#Define the stock symbol , start data , and end data

st.header('Stock price prediction')
stock_symbol = 'AAPL'
start_date = '2020-01-01'
end_date = '2023-01-01'

data = yf.download(stock_symbol, start=start_date,end=end_date)

print(data)