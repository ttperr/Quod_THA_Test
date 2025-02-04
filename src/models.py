import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from xgboost import XGBRegressor

from utils import create_lags, create_sequences


def prepare_data(df, separation_date, lstm=False):
    """Prepares time series data with lag features and splits into train/test."""
    df_grouped = df.groupby(['customer_id', 'month']).size().reset_index(name='transaction_count')

    # Create lag features
    for lag in [1, 2, 3]:
        df_grouped[f'transactions_lag_{lag}'] = df_grouped.groupby('customer_id')['transaction_count'].shift(lag)

    df_grouped = create_lags(df_grouped)
    df_grouped.dropna(inplace=True)

    df_grouped['rolling_mean'] = df_grouped.groupby('customer_id')['transaction_count'].transform(lambda x: x.rolling(window=3).mean())
    df_grouped['rolling_std'] = df_grouped.groupby('customer_id')['transaction_count'].transform(lambda x: x.rolling(window=3).std())

    # Create seasonal features
    df_grouped['month_num'] = df_grouped['month'].dt.month
    df_grouped['quarter'] = df_grouped['month'].dt.quarter
    df_grouped['year'] = df_grouped['month'].dt.year

    # One-hot encode the month and quarter
    df_grouped = pd.get_dummies(df_grouped, columns=['month_num', 'quarter'], drop_first=True)

    separation_date = pd.to_datetime(separation_date).to_period('M')
    train = df_grouped[df_grouped['month'] < separation_date]
    end_prediction_date = separation_date + 3
    test = df_grouped[(df_grouped['month'] >= separation_date) & (df_grouped['month'] < end_prediction_date)]


    X_train = train.drop(columns=['customer_id', 'month', 'transaction_count'])
    y_train = train['transaction_count']
    X_test = test.drop(columns=['customer_id', 'month', 'transaction_count'])
    y_test = test['transaction_count']

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Reset the index of y_train and y_test to ensure proper alignment
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    if lstm:
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test)
        return X_train_seq, X_test_seq, y_train_seq, y_test_seq, X_train.columns

    return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns

def train_model(model_name, X_train, y_train):
    """Trains selected model and returns it."""
    if model_name == 'Random Forest':
        model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    elif model_name == 'XGBoost':
        model = XGBRegressor(n_estimators=400, learning_rate=0.1, max_depth=5, random_state=42, n_jobs=-1)
    else:
        raise ValueError("Invalid model name")
    
    model.fit(X_train, y_train)
    return model

def train_model_lstm(model, X_train_seq, y_train_seq, epcohs=100, batch_size=32, verbose=0):
    """Trains LSTM model and returns it."""
    lstm_model = Sequential()
    lstm_model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(64, return_sequences=False, activation='relu'))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(32, activation='relu'))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mse')
    
    lstm_model.fit(X_train_seq, y_train_seq, epochs=epcohs, batch_size=batch_size, verbose=verbose)
    return lstm_model

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and returns performance metrics."""
    y_pred = model.predict(X_test)
    return {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred)
    }

def plot_feature_importance(model, X_train, features_name):
    """Plots feature importance for tree-based models."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots()
    ax.barh(range(X_train.shape[1]), importances[indices])
    ax.set_yticks(range(X_train.shape[1]))
    ax.set_yticklabels(features_name[indices])
    ax.set_xlabel('Feature Importance')
    ax.set_ylabel('Feature')
    return fig

def plot_predictions(model, X_test, y_test):
    """Plots actual vs. predicted values."""
    y_pred = model.predict(X_test)
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax.set_xlabel('Actual Transactions')
    ax.set_ylabel('Predicted Transactions')
    ax.set_title('Actual vs Predicted Transactions')
    return fig