import os

import pandas as pd

from utils import format_date_column

DATA_DIR = '..' + os.sep + 'data'

def load_transactions():
    """Loads and merges transaction data."""
    transactions_1 = pd.read_csv(DATA_DIR + os.sep + 'transactions_1.csv', index_col=0, parse_dates=['date'])
    transactions_2 = pd.read_csv(DATA_DIR + os.sep + 'transactions_2.csv', index_col=0, parse_dates=['date'])

    all_transactions = pd.concat([transactions_1, transactions_2]).drop_duplicates()
    all_transactions = format_date_column(all_transactions, 'date')
    all_transactions['month'] = all_transactions['date'].dt.to_period('M')

    return all_transactions