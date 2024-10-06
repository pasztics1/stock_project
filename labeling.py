import os 
import pandas as pd

#this fuc. labels the df's entries in a new column key='Higher' at every 60 minutes and ignores the rest as we don't care about those points (yet?)
#the function just saves the file in the data folder
#idk if the os library works on linux and mac or not...

def correct_file(file_name,period,separator=','): # period is in minutes for now
    path = os.path.join(os.getcwd(), "data")
    file_path = os.path.join(path,file_name)
    output_path = os.path.join(path,("labeled_"+file_name))
     
    df=pd.read_csv(file_path, sep=separator) #the orig. corrected.csv is sep=";"
    #tf is adj close?
    
    if os.path.exists(output_path):
        print(f"File at {output_path} already exists")
        df = pd.read_csv(output_path,sep=separator)

            
    else:
        df['Higher'] = False #as it turns out this is how you add a column to a pd df
                             #https://pandas.pydata.org/pandas-docs/version/1.3/reference/api/pandas.DataFrame.append.html
        
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        
        print(df.head())

        jumps = (60//period) #60//period = 60/period (using to get int.)
            
        for i in range(0, len(df)-(jumps),jumps):
            if df['Close'].iloc[i+jumps]>df['Close'].iloc[i]:
                df.loc[i,'Higher'] = True
        
        df['Higher'] = df['Higher'].astype(bool)
                
        df.to_csv(output_path, index=False)
        print(f"File saved to {output_path}")

        
    return df

