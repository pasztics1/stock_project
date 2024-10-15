import yfinance as yf
import os
#cd C:\Users\Surface\Desktop\binary_classifier_project\data
#Welcome to Alpha Vantage! Here is your API key: 5D82EK7F5SOHY41I. Please record this API key at a safe place for future data access.


"""
data=yf.download('AAPL',period='5d',interval='60s') #['AAPL']: must be one of ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']")

data.to_csv(os.path.join(os.getcwd(),'AAPL_5d_5min.csv')) #Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]
"""
import requests
import time
from datetime import datetime, timezone

api_key = 'cs6igf9r01qkeuli3ff0cs6igf9r01qkeuli3ffg'
symbol = 'AAPL'

bid_high = float('-inf')
bid_low = float('inf')
ask_high = float('-inf')
ask_low = float('inf')


i = 0
while True:
    # Get real-time quote
    url = f'https://finnhub.io/api/v1/quote?symbol={symbol}&token={api_key}'
    response = requests.get(url)
    data = response.json()

    # Extract BID and ASK prices
    bid_price = data['bid']
    ask_price = data['ask']

    # Update highs and lows
    if bid_price > bid_high:
        bid_high = bid_price
    if bid_price < bid_low:
        bid_low = bid_price
    if ask_price > ask_high:
        ask_high = ask_price
    if ask_price < ask_low:
        ask_low = ask_price

    print(f"BID High: {bid_high}, BID Low: {bid_low}")
    print(f"ASK High: {ask_high}, ASK Low: {ask_low}")
    
    if i%3==0:
        #create the ds
        #get prediction
        close_price = data['c']  # Current price
        volume = data['v']
        
        current_gmt_time = datetime.now(timezone.utc)
        formatted_gmt_time = current_gmt_time.strftime('%Y-%m-%d %H:%M:%S %Z')
        print(bid_price,ask_price,ask_low,ask_high,bid_low,bid_high,close_price,volume,formatted_gmt_time)
                
        
        #set highs and lows back 
        bid_high = float('-inf')
        bid_low = float('inf')
        ask_high = float('-inf')
        ask_low = float('inf')
        

    # sleep before getting the next update
    time.sleep(3) 
    i+=3