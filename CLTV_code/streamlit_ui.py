# Consolidated CLTV Dashboard App with Predictive Modeling Integration
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

    tab1, tab2, tab3, tab4 = st.tabs(["Upload / Load Data", "Insights", "Detailed View", "Predictions"])

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
        with tab4:
            show_prediction_tab(st.session_state['rfm_segmented'])
    else:
        for tab in [tab2, tab3, tab4]:
            with tab:
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
    custom_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']

    # First row: Segment Distribution and CLTV Prediction
    viz_col1, viz_col2 = st.columns(2)
    with viz_col1:
        st.markdown("#### 🎯 Customer Segment Distribution")
        fig1 = px.pie(segment_counts, values='Count', names='Segment', hole=0.45, color_discrete_sequence=custom_colors)
        fig1.update_traces(textinfo='percent+label', textposition='inside')
        st.plotly_chart(fig1, use_container_width=True)

    with viz_col2:
        st.markdown("#### 🔮 Predicted CLTV Distribution (3-Month)")
        fig4 = px.histogram(
            rfm_segmented, x='predicted_cltv_3m', nbins=30,
            title='CLTV Prediction Distribution',
            color_discrete_sequence=['#636efa']
        )
        fig4.update_layout(xaxis_title="Predicted CLTV", yaxis_title="Customer Count")
        st.plotly_chart(fig4, use_container_width=True)

    # Full-width: Segment-wise Average Metrics
    st.markdown("#### 📊 Segment-wise Average Metrics")
    metric_option = st.selectbox(
        "Choose Metric to Display",
        options=["AOV", "Average CLTV"],
        index=0,
        key="segment_metric_selector"
    )

    if metric_option == "AOV":
        metric_data = rfm_segmented.groupby("segment")['aov'].mean().reset_index().rename(columns={"aov": "value"})
        y_title = "Average Order Value"
    else:
        metric_data = rfm_segmented.groupby("segment")['CLTV'].mean().reset_index().rename(columns={"CLTV": "value"})
        y_title = "Average CLTV"

    fig2 = px.bar(
        metric_data.sort_values(by='value'),
        x='value',
        y='segment',
        orientation='h',
        labels={'value': y_title},
        color='segment',
        color_discrete_sequence=custom_colors,
        text='value'
    )
    fig2.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig2.update_layout(title=f"{y_title} by Segment", xaxis_title=y_title)
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    st.markdown("#### Top Products Bought by Segment Customers")

    try:
        selected_segment = st.selectbox("Choose a Customer Segment", options=['High', 'Medium', 'Low'], index=0)
        segment_users = rfm_segmented[rfm_segmented['segment'] == selected_segment]['User ID']
        segment_transaction_ids = df_transactions[df_transactions['User ID'].isin(segment_users)]['Transaction ID']

        orders = df_orders.rename(columns=lambda x: x.strip().lower().replace(" ", "_"))
        orders.rename(columns={'unitprice': 'unit_price'}, inplace=True)

        required_cols = ['transaction_id', 'product_id', 'quantity', 'unit_price']
        missing_cols = [col for col in required_cols if col not in orders.columns]
        if missing_cols:
            st.warning(f"⚠️ Required column(s) missing in orders data: {', '.join(missing_cols)}")
        else:
            filtered_orders = orders[orders['transaction_id'].isin(segment_transaction_ids)].copy()
            filtered_orders['revenue'] = filtered_orders['quantity'] * filtered_orders['unit_price']

            top_products = filtered_orders.groupby('product_id').agg(
                Total_Quantity=('quantity', 'sum'),
                Total_Revenue=('revenue', 'sum')
            ).sort_values(by='Total_Revenue', ascending=False).head(5).reset_index()

            if not top_products.empty:
                st.markdown(f"#### 📦 Top 5 Products by Revenue for '{selected_segment}' Segment")
                fig_products = px.bar(
                    top_products,
                    x='product_id',
                    y='Total_Revenue',
                    text='Total_Revenue',
                    labels={'product_id': 'Product ID', 'Total_Revenue': 'Revenue'},
                    color='product_id',
                    color_discrete_sequence=custom_colors[:5]
                )
                fig_products.update_traces(texttemplate='₹%{text:.2f}', textposition='outside')
                fig_products.update_layout(yaxis_title="Total Revenue", xaxis_title="Product ID")
                st.plotly_chart(fig_products, use_container_width=True)
            else:
                st.info("✅ No products found for this segment.")
    except Exception as e:
        st.warning(f"⚠️ Could not compute product stats: {e}")

def show_prediction_tab(rfm_segmented):
    st.subheader("🔮 Predicted CLTV (Next 3 Months)")
    st.caption("Forecasted Customer Lifetime Value using BG/NBD + Gamma-Gamma model.")
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

# Run the app
if __name__ == "__main__":
    run_streamlit_app()
