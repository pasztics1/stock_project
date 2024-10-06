import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from labeling import correct_file 
from main import RF

#df struct: Datetime, Open, High, Low, Close, Adj Close, Volume, Higher
#Higher is a boole value, which is true when the stock's closing value increased in the next hour. That's what the model is trying to guess.





df_test= correct_file("AAPL_1m_5min.csv",period=5)

df = correct_file("AAPL_5y_60min.csv", period=60)
X_train = df.iloc[:, 1:-1].values #not using date currently
y_train = df.iloc[:, -1].values


X_test = [] #only because of the 5 min period
y_test = []
for i in range(0, len(df_test), 12):
    X_test.append(df_test.iloc[i, 1:-1].values)  # Append features (excluding first and last column)
    y_test.append(df_test.iloc[i, -1])   
    
X_test = np.array(X_test)
y_test = np.array(y_test)


#initial plotting
fig, ax = plt.subplots()


for i in range(1,len(X_test)):
    color = "green" if (y_test[i-1]) else 'red'
    ax.plot(df_test['Datetime'].iloc[(i-1)*12:(i+1)*12], df_test['Close'].iloc[(i-1)*12:(i+1)*12],color=color, linewidth = 2)
    
    ax.axvline(df_test['Datetime'].iloc[i*12], color='gray', linestyle='--', linewidth=0.5)
    
    #print(df_test["Close"].iloc[i-12],df_test["Close"].iloc[i], df_test["Higher"].iloc[i-12])
    

 
ax.set_xlabel('Time')
ax.set_ylabel('Price')
ax.set_title('Stock Price Over Time')

ax.set_xticks([])
plt.tight_layout()

plt.show()


clf = RF()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)


#plotting results

fig, ax = plt.subplots()


for i in range(1,len(predictions)):
    color = "green" if (predictions[i]==y_test[i]) else 'red'
    
    ax.plot(df_test['Datetime'][(i-1)*12:(i+1)*12], df_test['Close'].iloc[(i-1)*12:(i+1)*12],color=color, linewidth = 2)
    ax.axvline(df_test['Datetime'].iloc[i*12], color='gray', linestyle='--', linewidth=0.5)
    
    #print(df_test["Close"].iloc[i-12],df_test["Close"].iloc[i], df_test["Higher"].iloc[i-12])
    

 
ax.set_xlabel('Time')
ax.set_ylabel('Price')
ax.set_title('Stock Price Over Time')

ax.set_xticks([])
plt.tight_layout()

plt.show()



print(f"Accuracy: {np.sum(y_test == predictions) / len(y_test)}")
print(f"Orig. True/False: {np.sum(y_test==True)}/{np.sum(y_test==False)} Pred. True/False: {np.sum(predictions==True)}/{np.sum(predictions==False)}")

results_df = pd.DataFrame({
    'Original': y_test,
    'Predicted': predictions
})

path = os.path.join(os.getcwd(), "data")
results_df.to_csv(os.path.join(path,"result.csv"),index=False)