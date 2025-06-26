import streamlit as st
import pandas as pd

def handle_file_uploads():
    orders_file = st.file_uploader("Upload Orders CSV", type=["csv"], key="orders_upload")
    transactions_file = st.file_uploader("Upload Transactions CSV", type=["csv"], key="transactions_upload")
    if orders_file and transactions_file:
        return pd.read_csv(orders_file), pd.read_csv(transactions_file)
    return None, None