import pandas as pd

def input_processing(order_path,transaction_path):
    order_df = pd.read_csv(order_path)
    transaction_df = pd.read_csv(transaction_path)

    return order_df,transaction_df

def data_type(order_df, transaction_df):

    transaction_df['Purchase Date'] = pd.to_datetime(transaction_df['Purchase Date'], dayfirst=True)
    order_df['Return Date'] = pd.to_datetime(order_df['Return Date'], dayfirst=True)

    order_df[['Unit Price','Total Amount', 'Discount Value', 'Shipping Cost', 'Total Payable']] = order_df[[ 'Unit Price','Total Amount', 'Discount Value', 'Shipping Cost', 'Total Payable']].astype('float')
    return order_df, transaction_df


