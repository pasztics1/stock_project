import os 
import pandas as pd

#the function just saves the file in the data folder
#idk if the os library works on linux and mac or not...

def correct_format(file_name,dayfirst = False, period=60,separator=',',predicted = 5):
    #setting the path to the right folder (stock_project\data)
    path = os.path.join(os.getcwd(), "data")
    file_path = os.path.join(path,file_name)
     
    df=pd.read_csv(file_path, sep=separator) #the orig. is sep=";"
    
    print(f"File at {file_path}")
    if dayfirst:
        df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d.%m.%Y %H:%M:%S.%f', errors='coerce')
    else:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
                        
    return df

