# input.py

import pandas as pd

def load_data(order_path, transaction_path):
    """Read CSV files from given paths."""
    order_df = pd.read_csv(order_path)
    transaction_df = pd.read_csv(transaction_path)
    return order_df, transaction_df

def convert_data_types(order_df, transaction_df):
    """Convert dates and numeric fields to proper formats."""
    transaction_df['Purchase Date'] = pd.to_datetime(transaction_df['Purchase Date'], dayfirst=True, errors='coerce')
    order_df['Return Date'] = pd.to_datetime(order_df['Return Date'], dayfirst=True, errors='coerce')

    numeric_columns = ['Unit Price', 'Total Amount', 'Discount Value', 'Shipping Cost', 'Total Payable']
    order_df[numeric_columns] = order_df[numeric_columns].astype(float, errors='ignore')
    return order_df, transaction_df

def load_and_process_data(order_path, transaction_path):
    """End-to-end wrapper to load and clean data."""
    order_df, transaction_df = load_data(order_path, transaction_path)
    return convert_data_types(order_df, transaction_df)
