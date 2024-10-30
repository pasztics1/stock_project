#(Datetime), Open, High, Low, Close, Adj Close, Volume, y
#these are the current types of data being used.

#Testing with data normalization could be used (however, in this context this doesn't really make sense...)

#Ideas:

##Timestamps##
#None/0/1 whether it's monday or friday (to chatch trading behavior assosiated with the start/end of the week)
#month (to detect seasonal trends)
#hour

#Market features
#fear/greed index
#relative strenght index (!)
#moving avarages

#Previous market data - maybe the use of lagged features
#previous day's close (we already have it, would be easy to calculate)
#previous day's (or other time period(s)) high and low  

#Market sentiment --REAL TIME DATA--
#articles
#news
#(social media)

#FOR LAGGED FEATURES, SUPPORT & RESISTANCE LEVELS THE FIRST AND LAST VALUES ARE NON

import os
from datetime import datetime
import time

import pandas as pd
import numpy as np

from read_data import correct_format

def add_earnings_feature_unix(df, earnings_dates, date_column='Datetime'):
    """
    Adds a feature to the dataframe indicating the difference in Unix timestamps until the next earnings call.
    
    Parameters:
    - df: The input dataframe containing a datetime column
    - earnings_dates: List of earnings call dates as datetime objects
    - date_column: The name of the column containing the datetime information
    
    Returns:
    - df: DataFrame with the added 'Unix_Diff_Until_Earnings' column
    """
    def get_unix_diff_until_next_earnings(date):
        date_unix = int(time.mktime(date.timetuple()))  # Current date in Unix timestamp

        # Find the next earnings call date that occurs after the given date
        next_earnings_date = next((ed for ed in earnings_dates if ed > date), None)

        if not next_earnings_date:
            return None

        next_earnings_unix = int(time.mktime(next_earnings_date.timetuple()))
        return next_earnings_unix - date_unix

    # Apply the function to each row in the dataframe
    df['Unix_Diff_Until_Earnings'] = df[date_column].apply(lambda x: get_unix_diff_until_next_earnings(x))
    return df

def add_features(ask_file_name, bid_file_name, PRECENTAGE, y_type, delta_t, label=True, dayfirst=True):
    """
    Adds engineered features to the merged ask and bid data.

    Parameters:
    - ask_file_name: Filename for ask data CSV
    - bid_file_name: Filename for bid data CSV
    - PRECENTAGE: Percentage of data to include (for quicker fitting)
    - y_type: Type of target variable (e.g., 'binary_classifier')
    - delta_t: Time delta for target variable (in hours)
    - label: Whether to add labels
    - dayfirst: Whether the date format is day-first

    Returns:
    - None (saves the processed DataFrame to a CSV file)
    """
    
    # Merging dataframes with bid and ask data 
    df_bid = correct_format(bid_file_name, dayfirst)
    df_ask = correct_format(ask_file_name, dayfirst)
    
    # Merge on 'Datetime' column
    df = pd.merge(df_bid, df_ask, on="Datetime", suffixes=('_BID', '_ASK'))
    
    # Calculating BID and ASK features
    df['Mid_Price'] = (df['Close_BID'] + df['Close_ASK']) / 2
    df['Spread'] = df['Close_ASK'] - df['Close_BID']
    
    # Since Volume_ASK != Volume_BID
    df['Volume'] = df[['Volume_ASK', 'Volume_BID']].mean(axis=1)

    # Calculate Mid Prices for OHLC
    df['Open_Mid'] = (df['Open_BID'] + df['Open_ASK']) / 2
    df['High_Mid'] = (df['High_BID'] + df['High_ASK']) / 2
    df['Low_Mid'] = (df['Low_BID'] + df['Low_ASK']) / 2
    df['Close_Mid'] = (df['Close_BID'] + df['Close_ASK']) / 2

    # Timestamps
    # Adding a feature that tracks how long until the next earnings call
    # Create simplified earnings call dates
    apple_earnings_calls = sorted([
        datetime(2025, 2, 6), datetime(2025, 5, 1), datetime(2025, 7, 31), datetime(2025, 10, 30),
        datetime(2024, 2, 1), datetime(2024, 5, 2), datetime(2024, 8, 1), datetime(2024, 11, 7),
        datetime(2023, 2, 2), datetime(2023, 5, 4), datetime(2023, 8, 3), datetime(2023, 11, 2),
        datetime(2022, 1, 27), datetime(2022, 4, 28), datetime(2022, 7, 28), datetime(2022, 10, 27),
        datetime(2021, 1, 27), datetime(2021, 4, 28), datetime(2021, 7, 27), datetime(2021, 10, 28),
        datetime(2020, 1, 28), datetime(2020, 4, 30), datetime(2020, 7, 30), datetime(2020, 10, 29),
        datetime(2019, 1, 29), datetime(2019, 4, 30), datetime(2019, 7, 30), datetime(2019, 10, 30),
    ])
    df = add_earnings_feature_unix(df, apple_earnings_calls)

    # Adjusting time format 
    # Assuming the data is in UTC, convert to Eastern Time
    import pytz

    df['Datetime'] = df['Datetime'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
    df['Month'] = df['Datetime'].dt.month
    df['Day_Type'] = df['Datetime'].dt.weekday.apply(lambda x: 1 if x == 0 else (2 if x == 4 else 0))  # 1 for Monday, 2 for Friday, 0 otherwise
    df['Week_of_Year'] = df['Datetime'].dt.isocalendar().week
    df['Quarter'] = df['Datetime'].dt.quarter

    # Hour of the day (0-23)
    df['Hour'] = df['Datetime'].dt.hour

    # Is it the first hour of trading?
    df['Is_Open_Hour'] = (df['Hour'] == 9).astype(int)

    # Is it the last hour of trading?
    df['Is_Close_Hour'] = (df['Hour'] == 15).astype(int)

    # Day of the week
    df['Day_of_Week'] = df['Datetime'].dt.dayofweek  # 0 = Monday, 6 = Sunday

    # Is it lunch time? (Market often slows down during lunch hours)
    df['Is_Lunch_Time'] = df['Hour'].isin([12, 13]).astype(int)
    
    # Time since market open
    df['Time_Since_Open'] = df['Hour'] - 9.5  # Market opens at 9:30 AM
    df['Time_Since_Open'] = df['Time_Since_Open'].clip(lower=0)

    # Market features

    # Fear / Greed CNN (definitely not from a network pull)
    # Read fear/greed
    path = os.path.join(os.getcwd(), "data")
    file_path = os.path.join(path, "fear_greed_historical.csv")
    df_fear_greed = pd.read_csv(file_path, parse_dates=['date'])
    df_fear_greed['date'] = pd.to_datetime(df_fear_greed['date']).dt.date

    # Merge it with the df
    df['date'] = df['Datetime'].dt.date  # Gonna drop it later
    df_merged = pd.merge(df, df_fear_greed, left_on='date', right_on='date', how='left')

    df_merged = df_merged.drop(columns=['date'])  # Dropped it

    # Momentum Indicators

    # Moving averages
    # Short term moving avg
    df_merged['SMA_3'] = df_merged['Mid_Price'].rolling(window=3).mean()
    df_merged['SMA_6'] = df_merged['Mid_Price'].rolling(window=6).mean()
    df_merged['SMA_12'] = df_merged['Mid_Price'].rolling(window=12).mean()
    # Long term moving avg
    df_merged['SMA_5'] = df_merged['Mid_Price'].rolling(window=5).mean()
    df_merged['SMA_10'] = df_merged['Mid_Price'].rolling(window=10).mean()
    df_merged['SMA_20'] = df_merged['Mid_Price'].rolling(window=20).mean()
    # Exponential moving avg
    df_merged['EMA_3'] = df_merged['Mid_Price'].ewm(span=3, adjust=False).mean()
    df_merged['EMA_6'] = df_merged['Mid_Price'].ewm(span=6, adjust=False).mean()

    # Moving average convergence divergence (MACD)
    df_merged['EMA_12'] = df_merged['Mid_Price'].ewm(span=12, adjust=False).mean()
    df_merged['EMA_26'] = df_merged['Mid_Price'].ewm(span=26, adjust=False).mean()
    df_merged['MACD'] = df_merged['EMA_12'] - df_merged['EMA_26']
    df_merged['Signal_Line'] = df_merged['MACD'].ewm(span=9, adjust=False).mean()

    # Stochastic Oscillator
    low_min = df_merged['Low_Mid'].rolling(window=14).min()
    high_max = df_merged['High_Mid'].rolling(window=14).max()
    df_merged['%K'] = 100 * ((df_merged['Close_Mid'] - low_min) / (high_max - low_min))
    df_merged['%D'] = df_merged['%K'].rolling(window=3).mean()

    # Commodity Channel Index (CCI)
    tp = (df_merged['High_Mid'] + df_merged['Low_Mid'] + df_merged['Close_Mid']) / 3
    tp_sma = tp.rolling(window=20).mean()
    mad = tp.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df_merged['CCI'] = (tp - tp_sma) / (0.015 * mad)

    # Momentum indicators over different periods
    df_merged['Return_1H'] = df_merged['Mid_Price'].pct_change(1)
    df_merged['Return_3H'] = df_merged['Mid_Price'].pct_change(3)
    df_merged['Return_6H'] = df_merged['Mid_Price'].pct_change(6)

    # Relative Strength Index over shorter and longer periods

    # Relative Strength Index over longer
    window_length = min(14, len(df_merged) - 1)

    # Calculate daily price changes  
    delta = df_merged['Mid_Price'].diff()
        
    # Separate gains and losses
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    # Calculate average gain and average loss over the window
    avg_gain = gain.rolling(window=window_length).mean()
    avg_loss = loss.rolling(window=window_length).mean()

    # Calculate Relative Strength (RS)
    RS = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero

    # Calculate RSI
    RSI = 100 - (100 / (1 + RS))
        
    # Add the RSI
    df_merged['RSI_long'] = RSI

    # RSI over shorter times

    # Calculate price changes
    delta = df_merged['Mid_Price'].diff()

    # Gains and losses
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    # Average gain and loss
    # window already defined above
    avg_gain = gain.rolling(window=window_length).mean()
    avg_loss = loss.rolling(window=window_length).mean()

    # RSI calculation
    rs = avg_gain / (avg_loss + 1e-10)
    df_merged['RSI_short'] = 100 - (100 / (1 + rs))

    # Volatility indicators

    # Bollinger Bands
    df_merged['Middle_Band'] = df_merged['Mid_Price'].rolling(window=20).mean()
    df_merged['Upper_Band'] = df_merged['Middle_Band'] + 2 * df_merged['Mid_Price'].rolling(window=20).std()
    df_merged['Lower_Band'] = df_merged['Middle_Band'] - 2 * df_merged['Mid_Price'].rolling(window=20).std()

    # Average True Range (ATR)
    df_merged['High_Low'] = df_merged['High_Mid'] - df_merged['Low_Mid']
    df_merged['High_Close'] = np.abs(df_merged['High_Mid'] - df_merged['Close_Mid'].shift())
    df_merged['Low_Close'] = np.abs(df_merged['Low_Mid'] - df_merged['Close_Mid'].shift())
    df_merged['True_Range'] = df_merged[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
    df_merged['ATR'] = df_merged['True_Range'].rolling(window=14).mean()

    # Volume Indicators

    # Volume-Based Features
    # Volume Moving Averages
    df_merged['Volume_SMA_3'] = df_merged['Volume'].rolling(window=3).mean()
    df_merged['Volume_SMA_6'] = df_merged['Volume'].rolling(window=6).mean()

    # On-Balance Volume (OBV)
    df_merged['OBV'] = (np.sign(df_merged['Close_Mid'].diff()) * df_merged['Volume']).fillna(0).cumsum()

    # Chaikin Money Flow (CMF)
    df_merged['MF_Multiplier'] = ((df_merged['Close_Mid'] - df_merged['Low_Mid']) - (df_merged['High_Mid'] - df_merged['Close_Mid'])) / (df_merged['High_Mid'] - df_merged['Low_Mid'])
    df_merged['MF_Volume'] = df_merged['MF_Multiplier'] * df_merged['Volume']
    df_merged['CMF'] = df_merged['MF_Volume'].rolling(window=20).sum() / df_merged['Volume'].rolling(window=20).sum()

    # Other features
    df_merged['Volatility'] = df_merged['Mid_Price'].rolling(window=5).std()
    df_merged['Return'] = df_merged['Mid_Price'].pct_change()  # Percent change in closing price
    df_merged['Volume_change'] = df_merged['Volume'].pct_change()  # Percent change in volume

    # Lagged features (for 1 to lag_period hours ago)
    lag_period = 5
    # Lagged Mid_Price
    for lag in range(1, lag_period):
        df_merged[f'Lag_Mid_Price_{lag}'] = df_merged['Mid_Price'].shift(lag)

    # Lagged Returns
    for lag in range(1, lag_period):
        df_merged[f'Lag_Return_{lag}'] = df_merged['Return'].shift(lag)

    # Lagged returns over different periods
    df_merged['Return_1'] = df_merged['Mid_Price'].pct_change(1)
    df_merged['Return_5'] = df_merged['Mid_Price'].pct_change(5)
    df_merged['Return_10'] = df_merged['Mid_Price'].pct_change(10)

    # Rolling Statistics
    df_merged['Rolling_Mean_5'] = df_merged['Mid_Price'].rolling(window=5).mean()
    df_merged['Rolling_Std_5'] = df_merged['Mid_Price'].rolling(window=5).std()

    # Lagged Volatility
    df_merged['Lagged_Volatility'] = df_merged['Volatility'].shift(1)

    # Candlestick Patterns

    # Hammer Pattern
    df_merged['Hammer'] = np.where(
        ((df_merged['High_Mid'] - df_merged['Low_Mid']) > 3 * (df_merged['Open_Mid'] - df_merged['Close_Mid'])) &
        ((df_merged['Close_Mid'] - df_merged['Low_Mid']) / (0.001 + df_merged['High_Mid'] - df_merged['Low_Mid']) > 0.6) &
        ((df_merged['Open_Mid'] - df_merged['Low_Mid']) / (0.001 + df_merged['High_Mid'] - df_merged['Low_Mid']) > 0.6),
        1, 0)

    # Statistical Features

    # Skewness and Kurtosis
    df_merged['Rolling_Skew'] = df_merged['Return'].rolling(window=20).skew()
    df_merged['Rolling_Kurt'] = df_merged['Return'].rolling(window=20).kurt()

    # Z-score of price
    df_merged['Price_Z_Score'] = (df_merged['Mid_Price'] - df_merged['Mid_Price'].rolling(window=20).mean()) / df_merged['Mid_Price'].rolling(window=20).std()

    # Advanced Technical Indicators

    # Ichimoku Cloud Components
    high9 = df_merged['High_Mid'].rolling(window=9).max()
    low9 = df_merged['Low_Mid'].rolling(window=9).min()
    df_merged['Conversion_Line'] = (high9 + low9) / 2

    high26 = df_merged['High_Mid'].rolling(window=26).max()
    low26 = df_merged['Low_Mid'].rolling(window=26).min()
    df_merged['Base_Line'] = (high26 + low26) / 2

    df_merged['Leading_Span_A'] = ((df_merged['Conversion_Line'] + df_merged['Base_Line']) / 2).shift(26)
    high52 = df_merged['High_Mid'].rolling(window=52).max()
    low52 = df_merged['Low_Mid'].rolling(window=52).min()
    df_merged['Leading_Span_B'] = ((high52 + low52) / 2).shift(26)

    # Williams %R
    high_n = df_merged['High_Mid'].rolling(window=14).max()
    low_n = df_merged['Low_Mid'].rolling(window=14).min()
    df_merged['Williams_%R'] = (high_n - df_merged['Close_Mid']) / (high_n - low_n) * -100

    # Price action features

    # High-Low ratio
    df_merged['High_Low_Ratio'] = df_merged['High_Mid'] / df_merged['Low_Mid']

    # Close-Open ratio
    df_merged['Close_Open_Ratio'] = df_merged['Close_Mid'] / df_merged['Open_Mid']

    # Volume and Order Flow Features

    # Order Flow features
    # Order Flow Imbalance
    df_merged['Order_Imbalance'] = df_merged['Volume_BID'] - df_merged['Volume_ASK']

    # Volume Price Trend (VPT)
    df_merged['VPT'] = (df_merged['Volume'] * (df_merged['Close_Mid'] - df_merged['Close_Mid'].shift(1)) / df_merged['Close_Mid'].shift(1)).cumsum()

    # Accumulation/Distribution Line (A/D Line)
    money_flow_multiplier = ((df_merged['Close_Mid'] - df_merged['Low_Mid']) - (df_merged['High_Mid'] - df_merged['Close_Mid'])) / (df_merged['High_Mid'] - df_merged['Low_Mid'])
    money_flow_volume = money_flow_multiplier * df_merged['Volume']
    df_merged['A/D_Line'] = money_flow_volume.cumsum()

    # Support and resistance levels

    df_merged['Support_10'] = df_merged[['Low_ASK', 'Low_BID']].min(axis=1).rolling(window=10).min()
    df_merged['Resistance_10'] = df_merged[['High_ASK', 'High_BID']].max(axis=1).rolling(window=10).max()
    
    # Calculating window=10 using the median value
    df_merged['Support_Close_10'] = df_merged['Mid_Price'].rolling(window=10).min()
    df_merged['Resistance_Close_10'] = df_merged['Mid_Price'].rolling(window=10).max()

    # Calculate support and resistance over the last 20 periods
    df_merged['Support_20'] = df_merged[['Low_ASK', 'Low_BID']].min(axis=1).rolling(window=20).min()
    df_merged['Resistance_20'] = df_merged[['High_ASK', 'High_BID']].max(axis=1).rolling(window=20).max()
    
    # Calculating window=20 using the median value
    df_merged['Support_Close_20'] = df_merged['Mid_Price'].rolling(window=20).min()
    df_merged['Resistance_Close_20'] = df_merged['Mid_Price'].rolling(window=20).max()

    # Feature Interactions

    # Interaction Terms
    df_merged['SMA5_Volume'] = df_merged['SMA_5'] * df_merged['Volume']
    df_merged['RSI_Volatility'] = df_merged['RSI_short'] * df_merged['Volatility']

    # Determining the y label
    if label:
        if y_type == "binary_classifier":
            print("Adding labels for binary classifier (hourly)")
            df_merged['y'] = (df_merged['Mid_Price'].shift(-delta_t) > df_merged['Mid_Price']).astype(int)
            df_merged['y'] = df_merged['y'].astype(np.int64)  # Only change to int if classification problem

        elif y_type == "percentage_change":
            print("Adding percentage change labels (hourly)")
            df_merged['y'] = ((df_merged['Mid_Price'].shift(-delta_t) - df_merged['Mid_Price']) / df_merged['Mid_Price']) * 100
        
        elif y_type == "price_movements_classification":
            print("Adding price movements classification labels (hourly)")
            future_return = ((df_merged['Mid_Price'].shift(-delta_t) - df_merged['Mid_Price']) / df_merged['Mid_Price']) * 100
            positive_threshold = 0.1  # Adjust based on desired sensitivity
            negative_threshold = -0.1
            conditions = [
                (future_return >= positive_threshold),
                (future_return <= negative_threshold)
            ]
            choices = [1, -1]
            df_merged['y'] = np.select(conditions, choices, default=0)

            df_merged['y'] = df_merged['y'].astype(np.int64)
        elif y_type == "implied_volatility":
            print("Implied volatility labeling not implemented yet.")
            return
        else:
            print(f"This type of y_type, {y_type}, is not implemented")
            return

    # Exclude the last delta_t rows where y cannot be computed
    df_merged = df_merged.iloc[:-delta_t]

    # Exclude the first few rows that may have NaN values from feature calculations
    df_merged = df_merged.dropna()

    start_index = int(df_merged.shape[0] - df_merged.shape[0] * PRECENTAGE)  # Include {percentage}% of the dataset for quicker fitting
    df_merged = df_merged.iloc[start_index:] 



    #the rating column can contain 5 values: extreme fear, fear, neutral, greed, extreme greed. I want to normalize this as well.
    if 'rating' in df_merged.columns:
        ordinal_mapping = {
            "extreme fear": 0,
            "fear": 1,
            "neutral": 2,
            "greed": 3,
            "extreme greed": 4
        }
        # Map the ordinal column values to integers
        df_merged['rating'] = df_merged['rating'].map(ordinal_mapping)
    else:
        print("Warning: 'rating' column not found after merging with fear_greed data.")
    
    
    # Saving the unscaled features
    output_path = os.path.join(path, f"features_{y_type}delta_t{delta_t}{PRECENTAGE}{ask_file_name}")
    df_merged.to_csv(output_path, index=False)
    print(f"Successfully added features to {output_path} with {PRECENTAGE * 100}% of the data")    
