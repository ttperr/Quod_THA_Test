import os

import pandas as pd
import streamlit as st

from data_loader import load_transactions
from models import (evaluate_model, plot_feature_importance, plot_predictions,
                    prepare_data, train_model, train_model_lstm)
from visualization import (monthly_sales_trend_plot, seasonality_analysis_plot,
                           top_5_products_plot, transactions_per_customer_plot,
                           transactions_per_product_plot)

# -------------------------- LOAD DATA -------------------------- #
DATA_DIR = '..' + os.sep + 'data'

# Load transactions
all_transactions = load_transactions()

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
        st.plotly_chart(transactions_per_customer_plot(all_transactions), use_container_width=True)

    elif visualization_select == "Transactions per Product":
        st.subheader("Total Transactions per Product (Descending)")
        st.plotly_chart(transactions_per_product_plot(all_transactions), use_container_width=True)

    elif visualization_select == "Monthly Sales Trends":
        st.subheader("Transaction Frequency per Month")
        product_id = st.selectbox("Select Product:", all_transactions['product_id'].unique())
        st.plotly_chart(monthly_sales_trend_plot(all_transactions, product_id), use_container_width=True)

    elif visualization_select == "Top 5 Products":
        st.subheader("Top 5 Products (every 6 months)")
        st.plotly_chart(top_5_products_plot(all_transactions), use_container_width=True)

    elif visualization_select == "Seasonality Analysis":
        st.subheader("Seasonality Analysis")
        product_id = st.selectbox("Select Product for Seasonality Analysis:", all_transactions['product_id'].unique())
        st.plotly_chart(seasonality_analysis_plot(all_transactions, product_id), use_container_width=True)
        
        st.write("We can clearly see the seasonality pattern in the data. There is a higher demand in October and November. The trend is also decreasing over time.")

# Prediction Models
elif selected_tab == "Prediction Models":
    st.subheader("Prediction Models")

    model_choice = st.selectbox("Select Prediction Model:", ['XGBoost', 'Random Forest', 'LSTM'], index=0)
    
    separation_date = st.date_input("Separation Date for Training and Test Set:", pd.to_datetime('2019-02-01'))
    
    if st.button("Train Model"):
        st.write("Training model...")

        # Prepare data
        X_train, X_test, y_train, y_test, features_name = prepare_data(all_transactions, separation_date, lstm=(model_choice == 'LSTM'))

        # Train model
        if model_choice == 'LSTM':
            model = train_model_lstm(model_choice, X_train, y_train, epcohs=1)
        else:
            model = train_model(model_choice, X_train, y_train)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        for metric, value in metrics.items():
            st.write(f"{metric}: {value}")
            
        # Plot feature importance
        if model_choice in ['Random Forest', 'XGBoost']:
            st.subheader("Feature Importance")
            fig = plot_feature_importance(model, X_train, features_name)
            st.pyplot(fig)
            
        # Plot predictions
        st.subheader("Predictions")
        fig = plot_predictions(model, X_test, y_test)
        st.pyplot(fig)

# Footer
st.write("Built with Streamlit | Author: Tristan PERROT")