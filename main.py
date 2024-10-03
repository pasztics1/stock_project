#predicting; a particular stock (AAPL,TSLA,MSFT,AMZN,NVDA,Alphabet), an index, maybe an asset or making a generalized model
#newsapi.org
#yahoofinance
import pandas as pd
import numpy as np
import yfinance as yf

data=pd.read_csv('AAPL_5y_60min.csv')
print(data['Close'].head())
