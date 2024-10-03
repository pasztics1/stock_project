import yfinance as yf
import os

data=yf.download('AAPL',period='2y',interval='60m')

data.to_csv(os.path.join(os.getcwd(),'AAPL_5y_60min.csv'))