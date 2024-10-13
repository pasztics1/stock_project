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

import pandas as pd
from labeling import correct_file

def add_features(ask_file_name, bid_file_name, dayfirst = True, predicted=5):
    
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
    
    df_merged['Higher'] = False 
    
            
    for i in range(0, len(df_merged)-(predicted)):
        if df_merged['Mid_Price'].iloc[i+predicted]>df_merged['Mid_Price'].iloc[i]:
            df_merged.loc[i,'Higher'] = True
        
        #this way the last 5 values don't get a "Higher" value
            
        
        df_merged['Higher'] = df_merged['Higher'].astype(bool)
        
    df_merged = df_merged.iloc[int(df.shape[0]-df.shape[0]*0.1):-5]

    df_merged = df_merged.dropna()
    print(df_merged.head())
    
    
    output_path = os.path.join(path,("features_"+ask_file_name))
    df_merged.to_csv(output_path, index=False)
    
    return f"Succesfully added features to {ask_file_name}"

    
add_features("AAPL.USUSD_Candlestick_1_M_ASK_11.10.2021-05.10.2024.csv","AAPL.USUSD_Candlestick_1_M_BID_11.10.2021-05.10.2024.csv")