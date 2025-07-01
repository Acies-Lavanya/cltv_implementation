import os
import streamlit as st
import pandas as pd
import plotly.express as px
from input import convert_data_types
from operations import CustomerAnalytics
from mapping import auto_map_columns, expected_orders_cols, expected_transaction_cols
from cltv_model import fit_bgf_ggf

BASE_DIR = os.path.dirname(__file__)
SAMPLE_ORDER_PATH = os.path.join(BASE_DIR, "sample_data", "Orders_v2.csv")
SAMPLE_TRANS_PATH = os.path.join(BASE_DIR, "sample_data", "Transactional_v2.csv")

def run_streamlit_app():
    st.set_page_config(page_title="CLTV Dashboard", layout="wide")
    st.title("Customer Lifetime Value Dashboard")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Upload / Load Data", "Insights", "Detailed View", "Predictions", "Realization Curve"
    ])

    with tab1:
        handle_data_upload()

    if data_ready():
        with tab2:
            show_insights()
        with tab3:
            show_detailed_view(st.session_state['rfm_segmented'], st.session_state['at_risk'])
        with tab4:
            show_prediction_tab(st.session_state['rfm_segmented'])
        with tab5:
            show_realization_curve(st.session_state['df_orders'], st.session_state['rfm_segmented'])
    else:
        for tab in [tab2, tab3, tab4, tab5]:
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

        # Merge user_id into df_orders based on transaction_id
        if 'Transaction ID' in df_transactions.columns and 'Transaction ID' in df_orders.columns:
            df_orders = df_orders.merge(
                df_transactions[['Transaction ID', 'User ID']],
                on='Transaction ID',
                how='left'
            )
        else:
            st.warning("Transaction ID not found in both orders and transactions.")

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

    segment_counts = rfm_segmented['segment'].value_counts().reset_index()
    segment_counts.columns = ['Segment', 'Count']
    custom_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']

    viz_col1, viz_col2 = st.columns(2)
    with viz_col1:
        st.markdown("#### 🎯 Customer Segment Distribution")
        fig1 = px.pie(segment_counts, values='Count', names='Segment', hole=0.45, color_discrete_sequence=custom_colors)
        fig1.update_traces(textinfo='percent+label', textposition='inside')
        st.plotly_chart(fig1, use_container_width=True)

    with viz_col2:
        st.markdown("#### 🔮 Predicted CLTV Distribution (3-Month)")
        fig4 = px.histogram(rfm_segmented, x='predicted_cltv_3m', nbins=30, color_discrete_sequence=['#636efa'])
        fig4.update_layout(xaxis_title="Predicted CLTV", yaxis_title="Customer Count")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("#### 📊 Segment-wise Average Metrics")
    metric_option = st.selectbox("Choose Metric to Display", options=["AOV", "Average CLTV"], index=0)
    if metric_option == "AOV":
        metric_data = rfm_segmented.groupby("segment")['aov'].mean().reset_index().rename(columns={"aov": "value"})
        y_title = "Average Order Value"
    else:
        metric_data = rfm_segmented.groupby("segment")['CLTV'].mean().reset_index().rename(columns={"CLTV": "value"})
        y_title = "Average CLTV"

    fig2 = px.bar(metric_data.sort_values(by='value'), x='value', y='segment', orientation='h',
                  labels={'value': y_title}, color='segment', color_discrete_sequence=custom_colors, text='value')
    fig2.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    st.plotly_chart(fig2, use_container_width=True)

    # 🛍️ Top Products by Segment
    st.divider()
    st.markdown("#### 🛍️ Top Products Bought by Segment Customers")
    try:
        selected_segment = st.selectbox("Choose a Customer Segment", options=['High', 'Medium', 'Low'], index=0)
        segment_users = rfm_segmented[rfm_segmented['segment'] == selected_segment]['User ID']
        segment_transaction_ids = df_transactions[df_transactions['User ID'].isin(segment_users)]['Transaction ID']

        orders = df_orders.rename(columns=lambda x: x.strip().lower().replace(" ", "_"))
        if 'unit_price' in orders.columns:
            orders.rename(columns={'unit_price': 'unitprice'}, inplace=True)

        required_cols = {'transaction_id', 'product_id', 'quantity', 'unitprice'}
        if not required_cols.issubset(set(orders.columns)):
            st.warning(f"⚠️ Required columns not found: {required_cols}")
            st.write("Found columns:", orders.columns.tolist())
        else:
            filtered_orders = orders[orders['transaction_id'].isin(segment_transaction_ids)].copy()
            filtered_orders['revenue'] = filtered_orders['quantity'] * filtered_orders['unitprice']

            top_products = (
                filtered_orders.groupby('product_id')
                .agg(Total_Quantity=('quantity', 'sum'), Total_Revenue=('revenue', 'sum'))
                .sort_values(by='Total_Revenue', ascending=False)
                .head(5)
                .reset_index()
            )

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
                st.plotly_chart(fig_products, use_container_width=True)
            else:
                st.info("✅ No products found for this segment.")
    except Exception as e:
        st.warning(f"⚠️ Could not compute top products: {e}")

def show_prediction_tab(rfm_segmented):
    st.subheader("🔮 Predicted CLTV (Next 3 Months)")
    st.caption("Forecasted Customer Lifetime Value using BG/NBD + Gamma-Gamma model.")
    st.dataframe(
        rfm_segmented[['User ID', 'predicted_cltv_3m']]
        .sort_values(by='predicted_cltv_3m', ascending=False)
        .reset_index(drop=True)
        .style.format({'predicted_cltv_3m': '₹{:,.2f}'}), use_container_width=True
    )

def show_detailed_view(rfm_segmented, at_risk):
    st.subheader("📋 Full RFM Segmented Data with CLTV")
    st.dataframe(rfm_segmented)

    st.subheader("⚠️ Customers at Risk (Recency > 90 days)")
    st.caption("These are customers whose last purchase was over 90 days ago and may be at risk of churning.")
    st.dataframe(at_risk)

def show_realization_curve(df_orders, rfm_segmented):
    st.subheader("📈 Realization Curve of CLTV Over Time")
    try:
        df = df_orders.copy()
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        if 'unit_price' in df.columns:
            df.rename(columns={'unit_price': 'unitprice'}, inplace=True)
        if 'user_id' not in df.columns and 'user id' in df.columns:
            df.rename(columns={'user id': 'user_id'}, inplace=True)

        required_cols = {'order_date', 'quantity', 'unitprice', 'user_id'}
        if not required_cols.issubset(set(df.columns)):
            st.warning(f"⚠️ Required columns not found: {required_cols}")
            st.write("Found columns:", df.columns.tolist())
            return

        df['order_date'] = pd.to_datetime(df['order_date'])
        df['revenue'] = df['quantity'] * df['unitprice']

        segment_option = st.selectbox("Select Customer Group for CLTV Curve",
                                      options=["Overall", "High CLTV Users", "Mid CLTV Users", "Low CLTV Users"])

        if segment_option == "High CLTV Users":
            selected_users = rfm_segmented[rfm_segmented['segment'] == 'High']['User ID']
        elif segment_option == "Mid CLTV Users":
            selected_users = rfm_segmented[rfm_segmented['segment'] == 'Medium']['User ID']
        elif segment_option == "Low CLTV Users":
            selected_users = rfm_segmented[rfm_segmented['segment'] == 'Low']['User ID']
        else:
            selected_users = df['user_id'].unique()

        filtered_df = df[df['user_id'].isin(selected_users)]
        user_count = filtered_df['user_id'].nunique()

        if user_count == 0:
            st.warning("⚠️ No users found in this segment.")
            return

        start_date = filtered_df['order_date'].min()
        intervals = [15, 30, 45, 60, 90]
        cltv_values = []

        for days in intervals:
            cutoff = start_date + pd.Timedelta(days=days)
            revenue = filtered_df[filtered_df['order_date'] <= cutoff]['revenue'].sum()
            avg_cltv = revenue / user_count
            cltv_values.append(round(avg_cltv, 2))

        chart_df = pd.DataFrame({
            "Period (Days)": intervals,
            "Avg CLTV per User": cltv_values
        })

        fig = px.line(chart_df, x="Period (Days)", y="Avg CLTV per User", markers=True)
        fig.update_layout(title=f"CLTV Realization Curve - {segment_option}", xaxis_title="Days", yaxis_title="Avg CLTV")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Could not generate CLTV curve: {e}")

def has_duplicate_columns(df1, df2):
    return df1.columns.duplicated().any() or df2.columns.duplicated().any()

if __name__ == "__main__":
    run_streamlit_app()
