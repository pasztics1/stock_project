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

    def get_unix_diff_since_last_earnings(date):
        past_earnings = [ed for ed in earnings_dates if ed < date]
        last_earnings_date = past_earnings[-1] if past_earnings else None
        if not last_earnings_date:
            return None
        return int(time.mktime(date.timetuple())) - int(time.mktime(last_earnings_date.timetuple()))


    # Apply the function to each row in the dataframe
    df['Unix_Diff_Since_Earnings'] = df[date_column].apply(lambda x: get_unix_diff_since_last_earnings(x))

    df['Unix_Diff_Until_Earnings'] = df[date_column].apply(lambda x: get_unix_diff_until_next_earnings(x))
    return df

def get_numerical_columns(df):
    """
    Returns a list of numerical column names in the DataFrame.
    """
    return df.select_dtypes(include=[np.number]).columns.tolist()

# Function to identify problematic columns
def identify_problematic_columns(df):
    # Columns with infinite values
    inf_cols = df.columns[df.isin([np.inf, -np.inf]).any()].tolist()
        
    # Columns with NaN values
    nan_cols = df.columns[df.isna().any()].tolist()
        
    # Columns with excessively large values (e.g., >1e10)
    threshold = 1e10
    large_cols = df.columns[(df.abs() > threshold).any()].tolist()
        
    return inf_cols, nan_cols, large_cols


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
    path = os.path.join(os.getcwd(), 'labeled_data')
    output_path = os.path.join(path, f"features_{y_type}delta_t{delta_t}{PRECENTAGE}{ask_file_name}")
        
    if not os.path.isfile(output_path):
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
        path = os.getcwd()
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

        df_merged['BB_Width'] = df_merged['Upper_Band'] - df_merged['Lower_Band']


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

        # -----------------------------------------
        # New Feature: Support & Resistance Levels
        # -----------------------------------------
        for period in [17, 30, 50]:
            df[f'Support_{period}'] = df['Low_Mid'].rolling(window=period).min()
            df[f'Resistance_{period}'] = df['High_Mid'].rolling(window=period).max()
        # End of Support & Resistance Levels feature addition

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


        #NEW FEATURES#

        #obv rate of change

        # Calculate percentage change safely
        df_merged['OBV_ROC'] = df_merged['OBV'].pct_change().replace([np.inf, -np.inf], np.nan)

        # Fill NaN values resulting from division by zero
        df_merged['OBV_ROC'] = df_merged['OBV_ROC'].fillna(0)

        

        #rate of change in the last 12 hours
        df_merged['ROC'] = df_merged['Mid_Price'].pct_change(12)  # 12-hour ROC

        #Directional moment index

        # Calculate directional movements
        df_merged['UpMove'] = df_merged['High_Mid'] - df_merged['High_Mid'].shift(1)
        df_merged['DownMove'] = df_merged['Low_Mid'].shift(1) - df_merged['Low_Mid']

        df_merged['+DM'] = np.where((df_merged['UpMove'] > df_merged['DownMove']) & (df_merged['UpMove'] > 0), df_merged['UpMove'], 0)
        df_merged['-DM'] = np.where((df_merged['DownMove'] > df_merged['UpMove']) & (df_merged['DownMove'] > 0), df_merged['DownMove'], 0)

        # Calculate ATR
        df_merged['ATR'] = df_merged['True_Range'].rolling(window=14).mean()

        # Calculate +DI and -DI
        df_merged['+DI'] = 100 * (df_merged['+DM'].rolling(window=14).mean() / df_merged['ATR'])
        df_merged['-DI'] = 100 * (df_merged['-DM'].rolling(window=14).mean() / df_merged['ATR'])

        # Calculate ADX
        df_merged['DX'] = (abs(df_merged['+DI'] - df_merged['-DI']) / (df_merged['+DI'] + df_merged['-DI'] + 1e-10)) * 100
        df_merged['ADX'] = df_merged['DX'].rolling(window=14).mean()

        #Parabolic SAR
        from ta.trend import PSARIndicator

        psar = PSARIndicator(high=df_merged['High_Mid'], low=df_merged['Low_Mid'], close=df_merged['Close_Mid'], step=0.02, max_step=0.2)
        df_merged['PSAR'] = psar.psar()

        #Candlestick patterns

        # Example: Engulfing Pattern
        df_merged['Engulfing'] = np.where(
            ((df_merged['Close_Mid'] > df_merged['Open_Mid']) & (df_merged['Close_Mid'].shift(1) < df_merged['Open_Mid'].shift(1)) & 
            (df_merged['Open_Mid'] < df_merged['Close_Mid'].shift(1)) & (df_merged['Close_Mid'] > df_merged['Open_Mid'].shift(1))),
            1, 0)


        #Price channels

        df_merged['Price_Channel_High'] = df_merged['Mid_Price'].rolling(window=20).max()
        df_merged['Price_Channel_Low'] = df_merged['Mid_Price'].rolling(window=20).min()
        df_merged['Price_Channel_Range'] = df_merged['Price_Channel_High'] - df_merged['Price_Channel_Low']


        #Volatility channels 
        df_merged['Hist_Volatility'] = df_merged['Return'].rolling(window=20).std()

        #Volume-weighted average price (VWAP)
        df_merged['Cumulative_Volume'] = df_merged['Volume'].cumsum()


        df_merged['Cumulative_VP'] = (df_merged['Mid_Price'] * df_merged['Volume']).cumsum()
        df_merged['VWAP'] = df_merged['Cumulative_VP'] / df_merged['Cumulative_Volume']

        # Add a small epsilon to avoid log(0)
        epsilon = 1e-10
        df_merged['Log_Cumulative_VP'] = np.log(df_merged['Cumulative_VP'] + epsilon)
        df_merged.drop(columns=['Cumulative_VP'], inplace=True)

        

        #Holiday effects

        from pandas.tseries.holiday import USFederalHolidayCalendar

        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=df_merged['Datetime'].min(), end=df_merged['Datetime'].max())

        df_merged['Is_Holiday'] = df_merged['Datetime'].dt.date.isin(holidays.date).astype(int)
        df_merged['Is_Nearest_Holiday'] = df_merged['Datetime'].dt.date.apply(
            lambda x: 1 if any(abs((x - holiday).days) <= 3 for holiday in holidays.date) else 0
        )

        #Daylight saving time transitions

        # df_merged['Is_DST_Start'] = df_merged['Datetime'].dt.date.isin([datetime(2024, 3, 10).date(), ...])  # Add all DST start dates
        # df_merged['Is_DST_End'] = df_merged['Datetime'].dt.date.isin([datetime(2024, 11, 3).date(), ...])  # Add all DST end dates

        #Interaction and polynomial features

        #feature crosses
        df_merged['RSI_MACD'] = df_merged['RSI_short'] * df_merged['MACD']
        df_merged['ATR_VPT'] = df_merged['ATR'] * df_merged['VPT']



        #time based decay features
        df_merged['Decay_Factor'] = np.exp(-0.1 * (df_merged['Datetime'] - df_merged['Datetime'].min()).dt.days)
        df_merged['Price_Decayed'] = df_merged['Mid_Price'] * df_merged['Decay_Factor']

        
        #Price Momentum Features

        #Momentum indicator
        df_merged['Momentum'] = df_merged['Mid_Price'] - df_merged['Mid_Price'].shift(10)

        #Velocity and acceleration
        df_merged['Velocity'] = df_merged['Mid_Price'].diff()
        df_merged['Acceleration'] = df_merged['Velocity'].diff()


        #Alt price representations
        df_merged['Log_Return'] = np.log(df_merged['Mid_Price'] / df_merged['Mid_Price'].shift(1))
        
        df_merged['Pct_VWAP'] = (df_merged['Mid_Price'] - df_merged['VWAP']) / df_merged['VWAP']



        #Additional time based features
        df_merged['Trading_Session'] = df_merged['Datetime'].dt.hour.apply(
            lambda x: 'Opening' if 9 <= x < 11 else ('Middle' if 11 <= x < 14 else 'Closing')
        )
        # One-hot encode the sessions
        df_merged = pd.get_dummies(df_merged, columns=['Trading_Session'])

        
        #Other
        df_merged['Spread_Change'] = df_merged['Spread'].pct_change()
        df_merged['Spread_Change_SMA_5'] = df_merged['Spread_Change'].rolling(window=5).mean()

        df_merged['Order_Book_Imbalance'] = (df_merged['Volume_BID'] - df_merged['Volume_ASK']) / (df_merged['Volume_BID'] + df_merged['Volume_ASK'] + 1e-10)

        df_merged['Breakout_Support'] = np.where(df_merged['Mid_Price'] < df_merged['Support_Close_10'], 1, 0)
        df_merged['Breakout_Resistance'] = np.where(df_merged['Mid_Price'] > df_merged['Resistance_Close_10'], 1, 0)

        df_merged['Mean_Reversion'] = np.where(
            (df_merged['Mid_Price'] > df_merged['Rolling_Mean_5']) & (df_merged['Return'] < 0), 1, 0)



        #Machine Learning-Based features

        numerical_cols = get_numerical_columns(df_merged)

        # Convert all numerical columns to float
        df_merged[numerical_cols] = df_merged[numerical_cols].apply(pd.to_numeric, errors='coerce')


        inf_cols, nan_cols, large_cols = identify_problematic_columns(df_merged[numerical_cols])

        # Apply imputer to the specified columns

        # Replace infinite values with NaN (if not already done)
        df_merged.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Forward fill and then backward fill
        df_merged[nan_cols] = df_merged[nan_cols].fillna(method='ffill').fillna(method='bfill')

        # For any remaining NaNs, fill with zero or another appropriate value
        df_merged[nan_cols] = df_merged[nan_cols].fillna(0)



        #PCA
        from sklearn.decomposition import PCA

        pca = PCA(n_components=5)
        pca_features = pca.fit_transform(df_merged[numerical_cols].fillna(0))
        for i in range(pca.n_components):
            df_merged[f'PCA_{i+1}'] = pca_features[:, i]

        
        #Autoencoder
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense

        # It's essential to ensure only numerical data is passed to the autoencoder
        input_dim = len(numerical_cols)
        encoding_dim = 10

        input_layer = Input(shape=(input_dim,))
        encoder = Dense(encoding_dim, activation='relu')(input_layer)
        decoder = Dense(input_dim, activation='linear')(encoder)

        autoencoder = Model(inputs=input_layer, outputs=decoder)
        autoencoder.compile(optimizer='adam', loss='mse')

        # Prepare data for autoencoder with explicit float conversion
        autoencoder_input = df_merged[numerical_cols].values.astype(float)

        # Train the autoencoder
        autoencoder.fit(
            autoencoder_input, 
            autoencoder_input,
            epochs=50, 
            batch_size=256, 
            shuffle=True, 
            verbose=0
        )


        # Extract the encoder part
        encoder_model = Model(inputs=input_layer, outputs=encoder)
        encoded_features = encoder_model.predict(autoencoder_input)
        for i in range(encoding_dim):
            df_merged[f'AutoEnc_{i+1}'] = encoded_features[:, i]

            
        #polynomial features
        # Check for NaNs in the specific columns

        # Verify that no NaNs remain
        nan_counts = df_merged[nan_cols].isna().sum()


        # Select only numerical columns for PolynomialFeatures
        numerical_cols_after_impute = get_numerical_columns(df_merged)
        poly_input = df_merged[['RSI_short', 'MACD', 'ATR', 'VPT']].copy()

        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import PolynomialFeatures

        # Scale features before PolynomialFeatures to prevent large polynomial terms
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(poly_input)
        poly_input_scaled = pd.DataFrame(scaled_features, columns=['RSI_short_scaled', 'MACD_scaled', 'ATR_scaled', 'VPT_scaled'], index=df_merged.index)

        # Apply PolynomialFeatures
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        poly_features = poly.fit_transform(poly_input_scaled)
        poly_feature_names = poly.get_feature_names_out(['RSI_short_scaled', 'MACD_scaled', 'ATR_scaled', 'VPT_scaled'])
        df_poly = pd.DataFrame(poly_features, columns=poly_feature_names, index=df_merged.index)
        df_merged = pd.concat([df_merged, df_poly], axis=1)



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
        df_merged = df_merged.iloc[:len(df_merged)-start_index] 


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
        path = os.path.join(os.getcwd(), 'labeled_data')
        output_path = os.path.join(path, f"features_{y_type}delta_t{delta_t}{PRECENTAGE}{ask_file_name}")
        df_merged.to_csv(output_path, index=False)
        print(f"Successfully added features to {output_path} with {PRECENTAGE * 100}% of the data")    


