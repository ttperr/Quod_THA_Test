import pandas as pd
import numpy as np

def format_date_column(df, column_name):
    """Ensures date columns are correctly formatted."""
    df[column_name] = pd.to_datetime(df[column_name]).dt.tz_localize(None)
    return df

def create_lags(df, lags=[1, 2, 3]):
    """Creates lag features for a time series dataset."""
    for lag in lags:
        df[f'transactions_lag_{lag}'] = df.groupby('customer_id')['transaction_count'].shift(lag)
    return df

def create_sequences(X, y, time_steps=3):
    """Creates sequences for a time series dataset."""
    Xs, ys = [], []
    X_filled = pd.DataFrame(X).fillna(0)
    y_filled = pd.DataFrame(y).fillna(0)
    for i in range(len(X_filled) - time_steps):
        Xs.append(X_filled[i:(i + time_steps)])
        ys.append(y_filled.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)