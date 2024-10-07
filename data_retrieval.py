import yfinance as yf
data = yf.download('AAPL', start='2010-01-01', end='2023-01-01')
data.reset_index(inplace=True)  # Convert the index to a 'Date' column
data.rename(columns={'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'}, inplace=True)
