import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose


# -------------------------- Transactions per Customer -------------------------- #
def transactions_per_customer_plot(df):
    """Plots transactions per customer in descending order."""
    customer_transactions = df.groupby('customer_id').size().reset_index(name='num_transactions')
    customer_transactions_sorted = customer_transactions.sort_values(by='num_transactions', ascending=False)
    
    # Store original customer IDs
    customer_transactions_sorted['original_customer_id'] = customer_transactions_sorted['customer_id']
    
    # Replace customer_id with an index (1 to N) for plotting
    customer_transactions_sorted['customer_id'] = range(1, len(customer_transactions_sorted) + 1)

    fig = px.bar(
        customer_transactions_sorted,
        x='customer_id',
        y='num_transactions',
        title="Total Transactions per Customer",
        labels={'customer_id': "Customer ID", 'num_transactions': "Number of Transactions"},
        text_auto=True,
        hover_data={'original_customer_id': True, 'customer_id': False},
        color='num_transactions'
    )
    fig.update_xaxes(showticklabels=False)  # Hide anonymized IDs
    return fig

# -------------------------- Transactions per Product -------------------------- #
def transactions_per_product_plot(df):
    """Plots transactions per product in descending order."""
    product_transactions = df.groupby('product_id').size().reset_index(name='num_transactions')
    product_transactions_sorted = product_transactions.sort_values(by='num_transactions', ascending=False)

    # Store original product IDs
    product_transactions_sorted['original_product_id'] = product_transactions_sorted['product_id']

    # Replace product_id with an index (1 to N) for plotting
    product_transactions_sorted['product_id'] = range(1, len(product_transactions_sorted) + 1)

    fig = px.bar(
        product_transactions_sorted,
        x='product_id',
        y='num_transactions',
        title="Total Transactions per Product",
        labels={'product_id': "Product ID", 'num_transactions': "Number of Transactions"},
        text_auto=True,
        hover_data={'original_product_id': True, 'product_id': False},
        color='num_transactions'
    )
    return fig

# -------------------------- Monthly Sales Trends -------------------------- #
def monthly_sales_trend_plot(df, product_id):
    """Plots monthly transaction trends for a selected product."""
    transactions_2018 = df[(df['date'].dt.year == 2018) & (df['product_id'] == product_id)]
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
    return fig

# -------------------------- Top 5 Products (Last 6 Months) -------------------------- #
def top_5_products_plot(df):
    """Plots the top 5 best-selling products every 6 months."""
    df['6_month_period'] = df['date'].dt.to_period('6M')

    # Compute top 5 products for each 6-month period
    top_5_products_per_period = df.groupby(['6_month_period', 'product_id']).size().reset_index(name='num_transactions')
    top_5_products_per_period['rank'] = top_5_products_per_period.groupby('6_month_period')['num_transactions'].rank(method='first', ascending=False)
    top_5_products = top_5_products_per_period[top_5_products_per_period['rank'] <= 5]['product_id'].unique()

    # Filter transactions to include only top 5 products
    top_5_transactions = df[df['product_id'].isin(top_5_products)]

    # Group by product and 6-month period
    top_5_6_month_sales = top_5_transactions.groupby(['6_month_period', 'product_id']).size().reset_index(name='num_transactions')
    top_5_6_month_sales['6_month_period'] = top_5_6_month_sales['6_month_period'].astype(str)

    fig = px.line(
        top_5_6_month_sales,
        x='6_month_period',
        y='num_transactions',
        color='product_id',
        title="Top 5 Products (every 6 months)",
        labels={'6_month_period': "6-Month Period", 'num_transactions': "Number of Transactions", 'product_id': "Product ID"}
    )
    return fig

# -------------------------- Seasonality Analysis -------------------------- #
def seasonality_analysis_plot(df, product_id):
    """Performs seasonal decomposition and plots observed, trend, seasonality, and residual components."""
    product_sales = df[df['product_id'] == product_id].groupby('month').size()
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