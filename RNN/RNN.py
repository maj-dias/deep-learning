##LSTM model that tends to capture up and down trend of a stock
from datetime import date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

def get_data(stock:str, start_date:date, end_date:date) -> pd.DataFrame:
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    df_stock = yf.Ticker(stock).history(start=start_date_str, end=end_date_str)
    return df_stock

def main() -> None:
    df_stock=get_data('MSFT',date(2015,1,1),date(2023,9,1))

    arr_training = df_stock['Open'].values

    sc = MinMaxScaler(feature_range=(0,1))

    arr_training_scaled = sc.fit(arr_training)

    print(arr_training_scaled)

if __name__ == '__main__':
    main()
    
## new cell
print('new_cell')