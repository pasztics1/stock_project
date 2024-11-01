import os 
import pandas as pd

#the function just saves the file in the data folder
#idk if the os library works on linux and mac or not...

def correct_format(file_name,dayfirst = False):
    #setting the path to the right folder (stock_project\data)
    path = os.path.join(os.getcwd(), "data")
    file_path = os.path.join(path,file_name)

    # if file_name=="AAPL.USUSD_Candlestick_1_Hour_ASK_18.09.2024-30.10.2024":
    #     file_path = r'C:\Users\CsP\Desktop\stock_project-master\data\AAPL.USUSD_Candlestick_1_Hour_ASK_18.09.2024-30.10.2024.csv'
    # else:
    #     file_name = r'C:\Users\CsP\Desktop\stock_project-master\data\AAPL.USUSD_Candlestick_1_Hour_BID_18.09.2024-30.10.2024.csv'

    df=pd.read_csv(file_path) #the orig. is sep=";"
    
    print(f"File at {file_path}")
    if dayfirst:
        df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d.%m.%Y %H:%M:%S.%f', errors='coerce')
    else:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
                        
    return df