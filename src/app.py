import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# -------------------------- LOAD DATA -------------------------- #
DATA_DIR = '..' + os.sep + 'data'

# Load transactions
transactions_1 = pd.read_csv(DATA_DIR + os.sep + 'transactions_1.csv', index_col=0, parse_dates=['date'])
transactions_2 = pd.read_csv(DATA_DIR + os.sep + 'transactions_2.csv', index_col=0, parse_dates=['date'])

# Combine transactions
all_transactions = pd.concat([transactions_1, transactions_2])
all_transactions = all_transactions.drop_duplicates()

# Ensure date is in datetime format and remove timezone if needed
all_transactions['date'] = pd.to_datetime(all_transactions['date']).dt.tz_localize(None)


# Create a 'month' column for monthly aggregations
all_transactions['month'] = all_transactions['date'].dt.to_period('M')

# -------------------------- STREAMLIT APP -------------------------- #

st.set_page_config(page_title="Transaction Data Analysis", layout="wide")

st.title("Transaction Data Visualization - Quod Financial Test")
st.subheader("Analyze and Visualize Transaction Data")

# Sidebar for navigation
st.sidebar.title("Navigation")
selected_tab = st.sidebar.radio("Go to:", ["Visualizations", "Prediction Models"])

# Transactions per Customer
if selected_tab == "Visualizations":
    
    visualization_select = st.selectbox("Select Visualization:", [
        "Transactions per Customer", "Transactions per Product", "Monthly Sales Trends", "Top 5 Products", "Seasonality Analysis"
    ])
    
    if visualization_select == "Transactions per Customer":
        st.subheader("Total Transactions per Customer (Descending)")
        
        # Compute transactions per customer (descending)
        # Compute transactions per customer (descending)
        customer_transactions = all_transactions.groupby('customer_id').size().reset_index(name='num_transactions')
        customer_transactions_sorted = customer_transactions.sort_values(by='num_transactions', ascending=False)

        # Store original customer IDs
        customer_transactions_sorted['original_customer_id'] = customer_transactions_sorted['customer_id']

        # Replace customer_id with an index (1 to N) for plotting
        customer_transactions_sorted['customer_id'] = [i for i in range(1, len(customer_transactions_sorted) + 1)]

        # Plot bar chart with manually sorted order
        fig = px.bar(
            customer_transactions_sorted,
            x='customer_id',
            y='num_transactions',
            title="Total Transactions per Customer",
            labels={'customer_id': "Customer ID", 'num_transactions': "Number of Transactions", 'original_customer_id': "Original Customer ID"},
            text_auto=True,
            hover_data={'original_customer_id': True, 'customer_id': False},
            color='num_transactions'
        )

        # Remove x-axis ticks to avoid showing anonymized IDs
        fig.update_xaxes(showticklabels=False)

        st.plotly_chart(fig, use_container_width=True)

    # Transactions per Product
    elif visualization_select == "Transactions per Product":
        st.subheader("Total Transactions per Product (Descending)")

        # Compute transactions per product (descending)
        product_transactions = all_transactions.groupby('product_id').size().reset_index(name='num_transactions')
        product_transactions_sorted = product_transactions.sort_values(by='num_transactions', ascending=False)

        # Store original product IDs
        product_transactions_sorted['original_product_id'] = product_transactions_sorted['product_id']

        # Replace product_id with an index (1 to N) for plotting
        product_transactions_sorted['product_id'] = [i for i in range(1, len(product_transactions_sorted) + 1)]

        # Plot bar chart with manually sorted order
        fig = px.bar(
            product_transactions_sorted,
            x='product_id',
            y='num_transactions',
            title="Total Transactions per Product",
            labels={'product_id': "Product ID", 'num_transactions': "Number of Transactions", 'original_product_id': "Original Product ID"},
            text_auto=True,
            hover_data={'original_product_id': True, 'product_id': False},
            color='num_transactions'
        )

        st.plotly_chart(fig, use_container_width=True)

    # Monthly Sales Trends
    elif visualization_select == "Monthly Sales Trends":
        st.subheader("Transaction Frequency per Month")
        
        product_id = st.selectbox("Select Product:", all_transactions['product_id'].unique())

        # Filter transactions
        transactions_2018 = all_transactions[
            (all_transactions['date'].dt.year == 2018) & (all_transactions['product_id'] == product_id)
        ]

        transactions_2018_per_month = transactions_2018.groupby(transactions_2018['date'].dt.month).size().reset_index()
        transactions_2018_per_month.columns = ['Month', 'Number of Transactions']
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        transactions_2018_per_month['Month'] = transactions_2018_per_month['Month'].apply(lambda x: month_names[x-1])

        fig = px.bar(
            transactions_2018_per_month,
            x="Month",
            y="Number of Transactions",
            title=f"Transactions per Month for {product_id} in 2018",
            text="Number of Transactions"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Top 5 Products (Last 6 Months)
    elif visualization_select == "Top 5 Products":
        st.subheader("Top 5 Products (every 6 months)")

        # Resample data to 6-month periods
        all_transactions['6_month_period'] = all_transactions['date'].dt.to_period('6M')

        # Compute top 5 products for each 6-month period
        top_5_products_per_period = all_transactions.groupby(['6_month_period', 'product_id']).size().reset_index(name='num_transactions')
        top_5_products_per_period['rank'] = top_5_products_per_period.groupby('6_month_period')['num_transactions'].rank(method='first', ascending=False)
        top_5_products = top_5_products_per_period[top_5_products_per_period['rank'] <= 5]['product_id'].unique()

        # Filter transactions to include only top 5 products
        top_5_transactions = all_transactions[all_transactions['product_id'].isin(top_5_products)]

        # Group by product and 6-month period
        top_5_6_month_sales = top_5_transactions.groupby(['6_month_period', 'product_id']).size().reset_index(name='num_transactions')
        
        # Convert '6_month_period' to string for better readability in the plot
        top_5_6_month_sales['6_month_period'] = top_5_6_month_sales['6_month_period'].astype(str)

        # Plot
        fig = px.line(
            top_5_6_month_sales,
            x='6_month_period',
            y='num_transactions',
            color='product_id',
            title="Top 5 Products (every 6 months)",
            labels={'6_month_period': "6-Month Period", 'num_transactions': "Number of Transactions", 'product_id': "Product ID"}
        )
        
        st.plotly_chart(fig, use_container_width=True)

    # Seasonality Analysis
    elif visualization_select == "Seasonality Analysis":
        st.subheader("Seasonality Analysis")

        product_id = st.selectbox("Select Product for Seasonality Analysis:", all_transactions['product_id'].unique())

        product_sales = all_transactions[all_transactions['product_id'] == product_id].groupby('month').size()
        product_sales.index = product_sales.index.to_timestamp()

        decomposition = seasonal_decompose(product_sales, model='additive', period=12)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=product_sales.index, y=decomposition.observed, mode='lines', name='Observed'))
        fig.add_trace(go.Scatter(x=product_sales.index, y=decomposition.trend, mode='lines', name='Trend'))
        fig.add_trace(go.Scatter(x=product_sales.index, y=decomposition.seasonal, mode='lines', name='Seasonality'))
        fig.add_trace(go.Scatter(x=product_sales.index, y=decomposition.resid, mode='lines', name='Residual'))

        fig.update_layout(
            title=f"Seasonality Decomposition for {product_id}",
            xaxis_title="Month",
            yaxis_title="Transactions",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)
        
        st.write("We can clearly see the seasonality pattern in the data. There is a higher demand in October and November. The trend is also decreasing over time.")

# Prediction Models
elif selected_tab == "Prediction Models":
    st.subheader("Prediction Models")

    model_choice = st.selectbox("Select Prediction Model:", ['Random Forest', 'XGBoost'])
    
    if st.button("Train Model"):
        st.write("Training model...")

        # Feature Engineering Example
        df_features = all_transactions.groupby(['customer_id', 'month']).size().reset_index(name='num_transactions')
        df_features['month'] = df_features['month'].dt.to_timestamp()

        X = df_features[['customer_id', 'num_transactions']]
        y = df_features['num_transactions'].shift(-1).fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_choice == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            st.write("XGBoost model implementation needed.")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        st.metric("Mean Absolute Error (MAE)", mae)
        st.metric("Mean Squared Error (MSE)", mse)

        fig = px.scatter(x=y_test, y=y_pred, labels={'x': "Actual", 'y': "Predicted"}, title="Actual vs Predicted")
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.write("Built with Streamlit | Author: Tristan PERROT")