import yfinance as yf
import os
#cd C:\Users\Surface\Desktop\binary_classifier_project\data


data=yf.download('AAPL',period='5d',interval='60s') #['AAPL']: must be one of ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']")

data.to_csv(os.path.join(os.getcwd(),'AAPL_5d_5min.csv')) #Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]