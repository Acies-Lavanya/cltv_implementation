# streamlit_ui.py

import streamlit as st
import pandas as pd
from input import convert_data_types
from operations import CustomerAnalytics
from mapping import auto_map_columns, expected_orders_cols, expected_transaction_cols

def run_streamlit_app():
    st.set_page_config(page_title="CLTV Dashboard", layout="wide")
    st.title("Customer Lifetime Value Dashboard")

    orders_file = st.file_uploader("Upload Orders CSV", type=["csv"])
    transactions_file = st.file_uploader("Upload Transactions CSV", type=["csv"])

    if orders_file and transactions_file:
        df_orders = pd.read_csv(orders_file)
        df_transactions = pd.read_csv(transactions_file)

        if has_duplicate_columns(df_orders, df_transactions):
            st.error("Duplicate column names detected.")
            return

        # Auto mapping
        orders_mapping = auto_map_columns(df_orders, expected_orders_cols)
        trans_mapping = auto_map_columns(df_transactions, expected_transaction_cols)

        # Rename & convert
        df_orders = df_orders.rename(columns={v: k for k, v in orders_mapping.items()})
        df_transactions = df_transactions.rename(columns={v: k for k, v in trans_mapping.items()})
        df_orders, df_transactions = convert_data_types(df_orders, df_transactions)

        # Analytics pipeline
        analytics = CustomerAnalytics(df_transactions)
        customer_level = analytics.compute_customer_level()
        rfm_segmented = analytics.rfm_segmentation(customer_level)
        rfm_segmented = analytics.calculate_cltv(rfm_segmented)
        at_risk = analytics.customers_at_risk(customer_level)

        # ---- Summary Page ----
        st.subheader("Key Insights")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", len(rfm_segmented))
        col2.metric("High Value Customers", (rfm_segmented['segment'] == 'High').sum())
        col3.metric("Customers at Risk", len(at_risk))

        st.subheader("Top 5 Customers by CLTV")
        st.dataframe(rfm_segmented[['User ID', 'CLTV']].sort_values(by='CLTV', ascending=False).head(5))

        # Additional Insights
        st.subheader("Customer Segment Distribution")
        seg_counts = rfm_segmented['segment'].value_counts()
        st.bar_chart(seg_counts)

        st.subheader("Average Order Value by Segment")
        aov_by_segment = rfm_segmented.groupby('segment')['aov'].mean()
        st.bar_chart(aov_by_segment)

        st.subheader("Revenue Contribution by Segment")
        revenue_by_segment = rfm_segmented.groupby('segment')['monetary'].sum()
        st.bar_chart(revenue_by_segment)

        st.subheader("Top Products Bought by High-Value Customers")
        high_value_users = rfm_segmented[rfm_segmented['segment'] == 'High']['User ID']
        top_products = df_orders[df_orders['Transaction ID'].isin(
            df_transactions[df_transactions['User ID'].isin(high_value_users)]['Transaction ID']
        )].groupby('Product ID')['Quantity'].sum().sort_values(ascending=False).head(5)

        st.dataframe(top_products.reset_index())

        if st.button("View Detailed Process"):
            show_detailed_view(rfm_segmented, at_risk)

def show_detailed_view(rfm_segmented, at_risk):
    st.subheader("Full RFM Segmented Data with CLTV")
    st.dataframe(rfm_segmented)

    st.subheader("Customers at Risk (Recency > 90 days)")
    st.dataframe(at_risk)

def has_duplicate_columns(df1, df2):
    return df1.columns.duplicated().any() or df2.columns.duplicated().any()
