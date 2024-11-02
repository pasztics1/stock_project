import os 
import pandas as pd

#the function just saves the file in the data folder
#idk if the os library works on linux and mac or not...

def correct_format(file_name,dayfirst = False):
    #setting the path to the right folder (stock_project\data)

    #sufnituning
    if (file_name == "AAPL.USUSD_Candlestick_1_Hour_ASK_26.01.2017-31.10.2024.csv") | (file_name == "AAPL.USUSD_Candlestick_1_Hour_BID_26.01.2017-31.10.2024.csv"):
        path = os.getcwd()
    else:
        path = os.path.join(os.getcwd(),'labeled_data')
            
    file_path = os.path.join(path,file_name)
    df=pd.read_csv(file_path) #the orig. is sep=";"
    
    print(f"File at {file_path}")
    if dayfirst:
        df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d.%m.%Y %H:%M:%S.%f', errors='coerce')
    else:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
                        
    return df