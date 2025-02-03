"""
Dash Visualization App for Transaction Analysis
---------------------------------------------------
This script creates an interactive Plotly Dash app to visualize:
1. Top 5 products driving sales over the last 6 months (Dynamic).
2. Monthly sales trends for all products.
3. Seasonality analysis using decomposition.
4. Ordered transactions per customer (Descending).

Author: Tristan
"""

import os

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# -------------------------- LOAD DATA -------------------------- #
DATA_DIR = '..' + os.sep + 'data'

# Load transactions
transactions_1 = pd.read_csv(DATA_DIR + os.sep + 'transactions_1.csv', index_col=0, parse_dates=['date'])
transactions_2 = pd.read_csv(DATA_DIR + os.sep + 'transactions_2.csv', index_col=0, parse_dates=['date'])

# Combine transactions
all_transactions = pd.concat([transactions_1, transactions_2])

# Ensure date is in datetime format and remove timezone if needed
all_transactions['date'] = pd.to_datetime(all_transactions['date']).dt.tz_localize(None)

# Create a 'month' column for monthly aggregations
all_transactions['month'] = all_transactions['date'].dt.to_period('M')

# Compute transactions per customer (descending)
customer_transactions = all_transactions.groupby('customer_id').size().reset_index(name='num_transactions')
customer_transactions = customer_transactions.sort_values(by='num_transactions', ascending=False)

# -------------------------- DASH APP -------------------------- #
app = dash.Dash(__name__)

app.layout = html.Div([
    
    # Apply Google Fonts
    html.Link(
        href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap",
        rel="stylesheet"
    ),
    
    html.H1("Transaction Data Visualization - Quod Financial Test", style={'textAlign': 'center', 'fontFamily': 'Poppins'}),
    html.H2("Analyze and Visualize Transaction Data", style={'textAlign': 'center', 'fontFamily': 'Poppins'}),
    
    dcc.Tabs([
        # Transactions per Customer (Descending)
        dcc.Tab(label="Transactions per Customer", children=[
            html.Br(),
            dcc.Graph(id='customer-transactions-graph')
        ]),
        
        # Monthly Sales Trends
        dcc.Tab(label="Monthly Sales Trends", children=[
            html.Br(),
            html.Label("Select Product for Analysis:", style={'fontFamily': 'Poppins', 'fontWeight': '600'}),
            dcc.Dropdown(
                id='product-dropdown',
                options=[{'label': p, 'value': p} for p in all_transactions['product_id'].unique()],
                value=all_transactions['product_id'].unique()[0],
                clearable=False
            ),
            dcc.Graph(id='product-transaction-plot')
        ]),
        
        # Top 5 Products (Last 6 Months)
        dcc.Tab(label="Top 5 Products (Last 6 Months)", children=[
            html.Br(),
            html.Label("📅 Select Reference Date:", style={'fontFamily': 'Poppins', 'fontWeight': '600'}),
            dcc.DatePickerSingle(
                id='date-picker',
                min_date_allowed=all_transactions['date'].min(),
                max_date_allowed=all_transactions['date'].max(),
                date=all_transactions['date'].max()
            ),
            dcc.Graph(id='top-products-graph')
        ]),
        
        # Seasonality Analysis
        dcc.Tab(label="Seasonality Analysis", children=[
            html.Br(),
            html.Label("Select Product for Analysis:", style={'fontFamily': 'Poppins', 'fontWeight': '600'}),
            dcc.Dropdown(
                id='seasonality-product-dropdown',
                options=[{'label': p, 'value': p} for p in all_transactions['product_id'].unique()],
                value=all_transactions['product_id'].value_counts().idxmax(),
                clearable=False
            ),
            dcc.Graph(id='seasonality-graph'),
            html.P("We can clearly see the seasonality pattern in the data. We observe that during the months of october and november, the sales are at their peak. This could be due to the holiday season and the need of cars for travel. The trend is also decreasing over time, which could be due to the increase in the number of cars available in the market.", style={'fontFamily': 'Poppins'})
        ]),

    ]),
    
    html.Hr(),
    
    html.H2("Prediction models", style={'textAlign': 'center', 'fontFamily': 'Poppins'}),
    html.Div([
        # Model selection dropdown
        html.Label("Select Prediction Model:", style={'fontFamily': 'Poppins', 'fontWeight': '600'}),
        dcc.Dropdown(
            id='model-selector',
            options=[
                {'label': 'Random Forest', 'value': 'RF'},
                {'label': 'XGBoost', 'value': 'XGB'}
                # You can add more options as needed
            ],
            value='RF',
            clearable=False,
            style={'width': '50%', 'margin': 'auto'}
        ),
        html.Br(),
        
        # Train Model Button
        html.Button("Train Model", id="train-button", n_clicks=0, style={'fontFamily': 'Poppins'}),
        html.Br(), html.Br(),

        # Display a table with performance metrics
        html.Div(id='performance-table', style={'width': '70%', 'margin': 'auto'}),

        # Automatic plot for actual vs. predicted values
        dcc.Graph(id='prediction-plot'),
        
        # A markdown/text block to explain the feature engineering
        html.Div([
            html.H4("Feature Engineering Explanation", style={'fontFamily': 'Poppins'}),
            dcc.Markdown("""
                The feature engineering pipeline includes:
                - **Lag Features:** Transaction counts from the previous 1 to 3 months.
                - **Rolling Aggregates:** A 3-month rolling average and sum to capture trends.
                - **Seasonality Dummies:** Month or quarter indicators to capture seasonal variations.
                - **Recency Features:** Time since the last transaction.
                These features help the model understand temporal trends and seasonality, 
                improving its ability to forecast the total transactions in the next three months.
            """, style={'fontFamily': 'Poppins'})
        ], style={'width': '70%', 'margin': 'auto', 'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px'})
    ], style={'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '10px', 'margin': '20px auto'} ),
    
    html.Hr(),
    html.P("This dashboard was created using Plotly Dash, a Python framework for building interactive web applications. The data used in this dashboard is a sample of transaction data from a car rental company. The data was preprocessed and analyzed to create the visualizations above.", style={'fontFamily': 'Poppins'}),
    html.P("Author: Tristan PERROT", style={'fontFamily': 'Poppins', 'textAlign': 'center'}),
    
], style={'fontFamily': 'Poppins'})

# -------------------------- CALLBACKS -------------------------- #

# **Update Transactions per Customer Graph (Descending)**
@app.callback(
    Output('customer-transactions-graph', 'figure'),
    Input('customer-transactions-graph', 'id')
)
def update_customer_transactions(_):
    # Ensure the dataframe is not empty
    if customer_transactions.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No transactions available",
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=20)
        )
        return fig

    # Ensure correct data types
    customer_transactions['customer_id'] = customer_transactions['customer_id'].astype(str)
    customer_transactions['num_transactions'] = pd.to_numeric(customer_transactions['num_transactions'], errors='coerce')

    # Create the bar chart
    fig = px.bar(
        customer_transactions,
        x='customer_id',
        y='num_transactions',
        title="Total Transactions per Customer (Descending)",
        labels={'customer_id': "Customer ID", 'num_transactions': "Number of Transactions"},
        text_auto=True,
        color='num_transactions'
    )

    # Update bar color and background color for readability
    fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)', marker_line_width=1.5, opacity=0.6)

    # Hide x-axis labels for readability
    fig.update_xaxes(showticklabels=False)

    return fig

# **Update Monthly Sales Trends Graph**
@app.callback(
    Output('product-transaction-plot', 'figure'),
    Input('product-dropdown', 'value')
)
def update_graph(product_id):
    # Filter transactions for the selected product
    transactions_2018 = all_transactions[
        (all_transactions['date'].dt.year == 2018) & (all_transactions['product_id'] == product_id)
    ]

    # Group by month
    transactions_2018_per_month = transactions_2018.groupby(transactions_2018['date'].dt.month).size().reset_index()
    transactions_2018_per_month.columns = ['Month', 'Number of Transactions']

    # Map month numbers to names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    transactions_2018_per_month['Month'] = transactions_2018_per_month['Month'].apply(lambda x: month_names[x-1])

    # Create the figure
    fig = px.bar(
        transactions_2018_per_month,
        x="Month",
        y="Number of Transactions",
        title=f"Transaction Frequency per Month for {product_id} in 2018",
        labels={"Month": "Month", "Number of Transactions": "Transaction Count"},
        text="Number of Transactions"
    )
    
    return fig

# **Update Top 5 Products based on selected date**
@app.callback(
    Output('top-products-graph', 'figure'),
    Input('date-picker', 'date')
)
def update_top_products(reference_date):
    reference_date = pd.to_datetime(reference_date)
    start_date = reference_date - pd.DateOffset(months=6)
    
    recent_transactions = all_transactions[
        (all_transactions['date'] >= start_date) & (all_transactions['date'] <= reference_date)
    ]
    
    product_sales = recent_transactions.groupby('product_id').size().reset_index(name='sales')
    top_5_products = product_sales.nlargest(5, 'sales')
    
    fig = px.bar(
        top_5_products, x='product_id', y='sales', 
        title="Top 5 Products (Last 6 Months)",
        labels={'product_id': "Product", 'sales': "Number of Transactions"},
        text_auto=True
    )
    
    return fig

# **Update Seasonality Graph based on selected product**
@app.callback(
    Output('seasonality-graph', 'figure'),
    Input('seasonality-product-dropdown', 'value')
)
def update_seasonality_graph(product_id):
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
    
    return fig



# -------------------------- RUN APP -------------------------- #
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)