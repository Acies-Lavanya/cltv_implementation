# Consolidated CLTV Dashboard App with Predictive Modeling Integration
import os
import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.colors import qualitative
from input import convert_data_types
from operations import CustomerAnalytics
from mapping import auto_map_columns, expected_orders_cols, expected_transaction_cols
from cltv_model import fit_bgf_ggf  # Predictive model

# Sample file paths
BASE_DIR = os.path.dirname(__file__)
SAMPLE_ORDER_PATH = os.path.join(BASE_DIR, "sample_data", "Orders.csv")
SAMPLE_TRANS_PATH = os.path.join(BASE_DIR, "sample_data", "Transactional.csv")

def run_streamlit_app():
    st.set_page_config(page_title="CLTV Dashboard", layout="wide")
    st.title("Customer Lifetime Value Dashboard")

    tab1, tab2, tab3 = st.tabs(["Upload / Load Data", "Insights", "Detailed View"])

    with tab1:
        handle_data_upload()

    if data_ready():
        with tab2:
            show_insights()
        with tab3:
            show_detailed_view(
                st.session_state['rfm_segmented'],
                st.session_state['at_risk']
            )
    else:
        with tab2:
            st.warning("⚠ Please upload or load data first.")
        with tab3:
            st.warning("⚠ Please upload or load data first.")

def handle_data_upload():
    orders_file = st.file_uploader("Upload Orders CSV", type=["csv"])
    transactions_file = st.file_uploader("Upload Transactions CSV", type=["csv"])

    if orders_file and transactions_file:
        st.session_state['orders_file'] = orders_file
        st.session_state['transactions_file'] = transactions_file
        st.session_state['use_sample'] = False
        process_data()
    elif st.button("🚀 Use Sample Data Instead"):
        st.session_state['use_sample'] = True
        process_data()

def process_data():
    try:
        if st.session_state.get('use_sample'):
            df_orders = pd.read_csv(SAMPLE_ORDER_PATH)
            df_transactions = pd.read_csv(SAMPLE_TRANS_PATH)
        else:
            df_orders = pd.read_csv(st.session_state['orders_file'])
            df_transactions = pd.read_csv(st.session_state['transactions_file'])

        if has_duplicate_columns(df_orders, df_transactions):
            st.error("❌ Duplicate column names detected.")
            return

        orders_mapping = auto_map_columns(df_orders, expected_orders_cols)
        trans_mapping = auto_map_columns(df_transactions, expected_transaction_cols)

        df_orders = df_orders.rename(columns={v: k for k, v in orders_mapping.items()})
        df_transactions = df_transactions.rename(columns={v: k for k, v in trans_mapping.items()})
        df_orders, df_transactions = convert_data_types(df_orders, df_transactions)

        analytics = CustomerAnalytics(df_transactions)
        customer_level = analytics.compute_customer_level()
        rfm_segmented = analytics.rfm_segmentation(customer_level)
        rfm_segmented = analytics.calculate_cltv(rfm_segmented)
        at_risk = analytics.customers_at_risk(customer_level)

        predicted_cltv = fit_bgf_ggf(df_transactions)
        rfm_segmented = rfm_segmented.merge(predicted_cltv, on='User ID', how='left')
        rfm_segmented['predicted_cltv_3m'] = rfm_segmented['predicted_cltv_3m'].fillna(0)

        st.session_state['df_orders'] = df_orders
        st.session_state['df_transactions'] = df_transactions
        st.session_state['rfm_segmented'] = rfm_segmented
        st.session_state['at_risk'] = at_risk

        st.success("✅ Data processed successfully!")

    except Exception as e:
        st.error(f"❌ Error during processing: {e}")

def data_ready():
    keys = ['df_orders', 'df_transactions', 'rfm_segmented', 'at_risk']
    return all(k in st.session_state and st.session_state[k] is not None for k in keys)

def show_insights():
    rfm_segmented = st.session_state['rfm_segmented']
    at_risk = st.session_state['at_risk']
    df_orders = st.session_state['df_orders']
    df_transactions = st.session_state['df_transactions']

    st.subheader("📊 Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(rfm_segmented))
    col2.metric("High Value Customers", (rfm_segmented['segment'] == 'High').sum())
    col3.metric("Customers at Risk*", len(at_risk))
    st.caption("📌 *Customers at Risk* refers to users whose **Recency > 90 days**, indicating potential churn risk.")

    st.divider()
    st.subheader("📈 Visual Insights")

    # Prepare data
    segment_counts = rfm_segmented['segment'].value_counts().reset_index()
    segment_counts.columns = ['Segment', 'Count']
    aov_by_segment = rfm_segmented.groupby('segment')['aov'].mean().reset_index()
    revenue_by_segment = rfm_segmented.groupby('segment')['monetary'].sum().reset_index()
    custom_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']

    # 2x2 layout
    viz_col1, viz_col2 = st.columns(2)
    with viz_col1:
        st.markdown("#### 🎯 Customer Segment Distribution")
        fig1 = px.pie(
            segment_counts,
            values='Count',
            names='Segment',
            hole=0.45,
            color_discrete_sequence=custom_colors
        )
        fig1.update_traces(textinfo='percent+label', textposition='inside')
        st.plotly_chart(fig1, use_container_width=True)

    with viz_col2:
        st.markdown("#### 🧾 Average Order Value by Segment")
        fig2 = px.bar(
            aov_by_segment.sort_values(by='aov'),
            x='aov',
            y='segment',
            orientation='h',
            color='segment',
            color_discrete_sequence=custom_colors
        )
        st.plotly_chart(fig2, use_container_width=True)

    viz_col3, viz_col4 = st.columns(2)
    with viz_col3:
        st.markdown("#### 💸 Revenue Contribution by Segment")
        fig3 = px.bar(
            revenue_by_segment.sort_values(by='monetary'),
            x='monetary',
            y='segment',
            orientation='h',
            color='segment',
            color_discrete_sequence=custom_colors
        )
        st.plotly_chart(fig3, use_container_width=True)

    with viz_col4:
        st.markdown("#### 🔮 Predicted CLTV Distribution (3-Month)")
        fig4 = px.histogram(
            rfm_segmented,
            x='predicted_cltv_3m',
            nbins=30,
            title='CLTV Prediction Distribution',
            color_discrete_sequence=['#636efa']
        )
        fig4.update_layout(xaxis_title="Predicted CLTV", yaxis_title="Customer Count")
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()
    st.subheader("📄 Tabular Insights")

    st.markdown("#### 🥇 Top 5 Customers by CLTV")
    st.dataframe(rfm_segmented[['User ID', 'CLTV']].sort_values(by='CLTV', ascending=False).head(5), use_container_width=True)

    st.markdown("#### 🔥 Top Products Bought by High-Value Customers")
    try:
        high_value_users = rfm_segmented[rfm_segmented['segment'] == 'High']['User ID']
        top_products = df_orders[df_orders['Transaction ID'].isin(
            df_transactions[df_transactions['User ID'].isin(high_value_users)]['Transaction ID']
        )].groupby('Product ID')['Quantity'].sum().sort_values(ascending=False).head(5)

        if not top_products.empty:
            st.dataframe(top_products.reset_index(), use_container_width=True)
        else:
            st.info("✅ No top products found for high-value customers.")
    except Exception as e:
        st.warning(f"⚠️ Could not compute top products: {e}")

    st.markdown("#### 📉 Predicted CLTV (Next 3 Months - BG/NBD + Gamma-Gamma)")
    st.dataframe(
        rfm_segmented[['User ID', 'predicted_cltv_3m']]
        .sort_values(by='predicted_cltv_3m', ascending=False)
        .reset_index(drop=True)
        .style.format({'predicted_cltv_3m': '₹{:,.2f}'}),
        use_container_width=True
    )

def show_detailed_view(rfm_segmented, at_risk):
    st.subheader("📋 Full RFM Segmented Data with CLTV")
    st.dataframe(rfm_segmented)

    st.subheader("⚠️ Customers at Risk (Recency > 90 days)")
    st.caption("These are customers whose last purchase was over 90 days ago and may be at risk of churning.")
    st.dataframe(at_risk)

def has_duplicate_columns(df1, df2):
    return df1.columns.duplicated().any() or df2.columns.duplicated().any()

# To run the app
if __name__ == "__main__":
    run_streamlit_app()
