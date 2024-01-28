# LSTM model that tends to capture up and down trend of a stock
# Code based on Deep Learning A-Z Udemy Course

from datetime import date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

def get_data(stock:str, start_date:date, end_date:date) -> pd.DataFrame:
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    df_stock = yf.Ticker(stock).history(start=start_date_str, end=end_date_str)
    return df_stock


df_stock=get_data('MSFT',date(2015,1,1),date(2023,9,1))

arr_training = df_stock['Open'].values

arr_training = arr_training.reshape(-1,1) #transforma num vetor coluna

sc = MinMaxScaler(feature_range=(0,1))

arr_training_scaled = sc.fit_transform(arr_training)

X_train = []
y_train = []

for i in range(60, len(arr_training_scaled)):
    X_train.append(arr_training_scaled[i-60:i,0])
    y_train.append(arr_training_scaled[i])

X_train = np.array(X_train)
y_train = np.array(y_train)

print(X_train.shape)
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
print(X_train.shape)

# %% Rede Neural Recorrente

regressor = Sequential()

#first layer
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(60,1)))
regressor.add(Dropout(0.2))

#second layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

#third layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

#fourth layer
regressor.add(LSTM(units=50, return_sequences=False))
regressor.add(Dropout(0.2))

#output layer
regressor.add(Dense(units=1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 50, batch_size = 32)

# %% Avaliando

df_stock_test = get_data('MSFT',date(2023,9,1),date(2023,12,1))

df_total = pd.concat((df_stock['Open'], df_stock_test['Open']), axis = 0)

inputs = df_total[len(df_total) - len(df_stock_test) - 60:].values #remove actual values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []

for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, 0])
    
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


plt.plot(df_stock_test['Open'], color = 'red', label = 'Real')
plt.plot(df_stock_test.index, predicted_stock_price, color = 'blue', label = 'Predicted')
plt.title('MSFT Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('MSFT Stock Price')
plt.legend()
plt.show()

