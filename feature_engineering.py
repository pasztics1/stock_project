#(Datetime), Open, High, Low, Close, Adj Close, Volume, Higher
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
from sklearn.preprocessing import MinMaxScaler

from read_data import correct_format

def normalize_features(df, exclude_columns=None):
    """
    Normalizes numeric columns in the dataframe using Min-Max Scaling.
    Parameters:
    - df: The input dataframe to be normalized
    - exclude_columns: List of columns to exclude from normalization (like target variables, datetime, etc.)
    """
    if exclude_columns is None:
        exclude_columns = []

    # Identify numeric columns excluding the ones in exclude_columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in exclude_columns]

    # Apply Min-Max Scaling to numeric columns
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df


def add_earnings_feature_unix(df, earnings_dates, date_column='Datetime'):
    """
    Adds a feature to the dataframe indicating the difference in Unix timestamps until the next earnings call.
    Parameters:
    - df: The input dataframe containing a datetime column
    - earnings_dates: List of earnings call dates as datetime objects
    - date_column: The name of the column containing the datetime information
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

def add_features(ask_file_name, bid_file_name, PRECENTAGE, label=True, dayfirst = True, predicted=5): #void function, just adds the features to a merged csv based on ask and bid data.
    
    #Merging dataframes with bid and ask data
    df_bid = correct_file(bid_file_name,dayfirst)
    df_ask = correct_file(ask_file_name,dayfirst)
    
    df=pd.merge(df_bid,df_ask, on="Datetime", suffixes=('_BID', '_ASK'))
    
    #Calculating BID and ASK features
    df['Mid_Price'] = (df['Close_BID'] + df['Close_ASK']) / 2
    df['Spread'] = df['Close_ASK'] - df['Close_BID']
    
    #since Volume_BID = Volume_ASK
    df = df.drop(columns="Volume_BID")
    df.rename(columns={'Volume_ASK':'Volume'},inplace=True)



    #Timestamps#
    df['Hour'] = df['Datetime'].dt.hour
    df['Month'] = df['Datetime'].dt.month
    df['Day_Type'] = df['Datetime'].dt.weekday.apply(lambda x: 1 if x == 0 else (2 if x == 4 else 0))  # 1 for Monday, 2 for Friday, 0 otherwise

    #adding a feature that tracks how long untill the next earnings call
    
    #create simplified earnings call dates
    apple_earnings_calls = sorted([
    datetime(2025, 2, 6), datetime(2025, 5, 1), datetime(2025, 7, 31), datetime(2025, 10, 30),
    datetime(2024, 2, 1), datetime(2024, 5, 2), datetime(2024, 8, 1), datetime(2024, 11, 7),
    datetime(2023, 2, 2), datetime(2023, 5, 4), datetime(2023, 8, 3), datetime(2023, 11, 2),
    datetime(2022, 1, 27), datetime(2022, 4, 28), datetime(2022, 7, 28), datetime(2022, 10, 27),
    datetime(2021, 1, 27), datetime(2021, 4, 28), datetime(2021, 7, 27), datetime(2021, 10, 28),
    datetime(2020, 1, 28), datetime(2020, 4, 30), datetime(2020, 7, 30), datetime(2020, 10, 29),
    datetime(2019, 1, 29), datetime(2019, 4, 30), datetime(2019, 7, 30), datetime(2019, 10, 30),
    ])
    df = add_earnings_feature_unix(df,apple_earnings_calls)
    #

    #Market features#

    #Fear / Greed CNN (definetely not from a network pull)
    #read fear/greed
    path = os.path.join(os.getcwd(), "data")
    file_path = os.path.join(path,"fear_greed_historical.csv")
    df_fear_greed = pd.read_csv(file_path, parse_dates=['date'])
    df_fear_greed['date'] = pd.to_datetime(df_fear_greed['date']).dt.date

    #merge it with the df
    df['date'] = df['Datetime'].dt.date #gonna drop it later
    df_merged = pd.merge(df,df_fear_greed,left_on='date', right_on='date',how='left')

    df_merged = df_merged.drop(columns=['date']) #dropped it
    #
    

    #Moving averages
    df_merged['SMA_5'] = df_merged['Mid_Price'].rolling(window=5).mean()
    df_merged['SMA_10'] = df_merged['Mid_Price'].rolling(window=10).mean()
    #

    #Other features
    df_merged['Volatility'] = df_merged['Mid_Price'].rolling(window=5).std()
    df_merged['Return'] = df_merged['Mid_Price'].pct_change() #precent change in closing price
    df_merged['Volume_change'] = df_merged['Volume'].pct_change() #precent change in volume
    #

    #Lagged features (for 1 to 5 mins. ago)
    for lag in range(1, 6): #lagged closing price
        df_merged[f'Lag_Close_{lag}'] = df_merged['Mid_Price'].shift(lag)

    for lag in range(1, 6): #lagged volume
        df_merged[f'Lag_Volume_{lag}'] = df_merged['Volume'].shift(lag)
        
    for lag in range(1, 6): #lagged returns
        df_merged[f'Lag_Return_{lag}'] = df_merged['Return'].shift(lag)
    #

    #Support and resistance levels

    df_merged['Support_10'] = df_merged[['Low_ASK', 'Low_BID']].min(axis=1).rolling(window=10).min()
    df_merged['Resistance_10'] = df_merged[['High_ASK', 'High_BID']].max(axis=1).rolling(window=10).max()
    
    #calculating window=10 using the median value
    df_merged['Support_Close_10'] = df_merged['Mid_Price'].rolling(window=10).min()
    df_merged['Resistance_Close_10'] = df_merged['Mid_Price'].rolling(window=10).max()

    # Calculate support and resistance over the last 20 periods
    df_merged['Support_20'] = df_merged[['Low_ASK', 'Low_BID']].min(axis=1).rolling(window=20).min()
    df_merged['Resistance_20'] = df_merged[['High_ASK', 'High_BID']].max(axis=1).rolling(window=20).max()
    
    #calculating window=20 using the median value
    df_merged['Support_Close_20'] = df_merged['Mid_Price'].rolling(window=20).min()
    df_merged['Resistance_Close_20'] = df_merged['Mid_Price'].rolling(window=20).max()
    #


    #Relative strenght index

    window=14
        #Calculate daily price changes  
    delta = df_merged['Mid_Price'].diff()
        
        #Separate gains and losses
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

        #Calculate average gain and average loss over the window
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

        #Calculate Relative Strength (RS)
    
    RS = avg_gain / avg_loss #we might divide by zero and therefore break the universe but I live a dangerous life
    #Calculate RSI
    RSI = 100 - (100 / (1 + RS))
        
    # Add the RSI
    df_merged['RSI'] = RSI
    
    
    
    #Determining the y label
    if label:
        print(predicted)
        
        df_merged['Higher'] = False 
        for i in range(0, len(df_merged)-(predicted)):
            if df_merged['Mid_Price'].iloc[i+predicted]>df_merged['Mid_Price'].iloc[i]:
                df_merged.loc[i,'Higher'] = True
            
            #this way the last 5 values don't get a "Higher" value
                
            
            df_merged['Higher'] = df_merged['Higher'].astype(bool)
        
    df_merged = df_merged.iloc[int(df.shape[0]-df.shape[0]*PRECENTAGE):-5] # Include {precentage}% of a dataset for quicker fitting

    df_merged = df_merged.dropna()
    df_normalized = normalize_features(df_merged,['Spread','Day_Type','Unix_Diff_Until_Earnings'])
    
    #the rating column can contain 5 values: extreme fear, fear, neutral, greed, extreme greed. I want to normalize this as well.
    ordinal_mapping = {
        "extreme fear": 0,
        "fear": 1,
        "neutral": 2,
        "greed": 3,
        "extreme greed": 4
    }
    
    # Map the ordinal column values to integers
    df_normalized['rating'] = df_normalized['rating'].map(ordinal_mapping)
    

    
    
    output_path = os.path.join(path,(f"features_{PRECENTAGE}{ask_file_name}"))
    df_normalized.to_csv(output_path, index=False)
    print(f"Succesfully added features to {ask_file_name} with {PRECENTAGE*100}% of the data")    
     
