import os
import streamlit as st
import pandas as pd
import plotly.express as px
from input import convert_data_types
from operations import CustomerAnalytics
from mapping import auto_map_columns, expected_orders_cols, expected_transaction_cols
from cltv_model import fit_bgf_ggf
from churn_model import train_churn_model
from cox_model import train_cox_model

# Sample file paths
BASE_DIR = os.path.dirname(__file__)
SAMPLE_ORDER_PATH = os.path.join(BASE_DIR, "sample_data", "Orders.csv")
SAMPLE_TRANS_PATH = os.path.join(BASE_DIR, "sample_data", "Transactional.csv")

def run_streamlit_app():
    st.set_page_config(page_title="CLTV Dashboard", layout="wide")
    st.title("Customer Lifetime Value Dashboard")


    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Upload / Load Data", "Insights", "Detailed View", "Predictions", "Realization Curve", "Churn" 
    ])

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
        with tab5:
            show_churn_tab()
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

        analytics = CustomerAnalytics(df_transactions)
        customer_level = analytics.compute_customer_level()
        rfm_segmented = analytics.rfm_segmentation(customer_level)
        rfm_segmented = analytics.calculate_cltv(rfm_segmented)
        rfm_segmented = analytics.label_churned_customers(rfm_segmented)
        rfm_segmented = analytics.prepare_survival_data(rfm_segmented)

        X, y = analytics.get_churn_features(rfm_segmented)

        model, report, importances, X_test, y_test = train_churn_model(X, y)

        rfm_segmented['predicted_churn_prob'] = model.predict_proba(X)[:, 1]
        rfm_segmented['predicted_churn'] = (rfm_segmented['predicted_churn_prob'] >= 0.5).astype(int)

        predicted_cltv = fit_bgf_ggf(df_transactions)
        rfm_segmented = rfm_segmented.merge(predicted_cltv, on='User ID', how='left')
        rfm_segmented['predicted_cltv_3m'] = rfm_segmented['predicted_cltv_3m'].fillna(0)

        at_risk = analytics.customers_at_risk(rfm_segmented)

        cox_features = ['recency', 'frequency', 'monetary', 'aov', 'avg_days_between_orders',
                'CLTV_30d', 'CLTV_60d', 'CLTV_90d']
        cox_model, rfm_segmented = train_cox_model(rfm_segmented, cox_features)
        st.session_state['cox_model'] = cox_model

        st.session_state['df_orders'] = df_orders
        st.session_state['df_transactions'] = df_transactions
        st.session_state['rfm_segmented'] = rfm_segmented
        st.session_state['at_risk'] = at_risk
        st.session_state['churn_model'] = model
        st.session_state['churn_report'] = report
        st.session_state['churn_importance'] = importances

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

    st.subheader("📌 Key KPIs")

    # Safety check: required columns must exist in already-standardized format
    required_order_cols = {'Quantity', 'Unit Price', 'Order Date', 'User ID'}
    required_trans_cols = {'User ID', 'Transaction ID'}

    if not required_order_cols.issubset(df_orders.columns):
        st.warning(f"⚠ Required columns missing from orders: {required_order_cols - set(df_orders.columns)}")
        return
    if not required_trans_cols.issubset(df_transactions.columns):
        st.warning(f"⚠ Required columns missing from transactions: {required_trans_cols - set(df_transactions.columns)}")
        return

    # KPI Calculations
    df_orders['Revenue'] = df_orders['Quantity'] * df_orders['Unit Price']
    total_revenue = df_orders['Revenue'].sum()
    aov = df_orders.groupby('User ID')['Revenue'].sum().mean()
    avg_cltv = rfm_segmented['CLTV'].mean()
    avg_txns_per_user = df_transactions.groupby('User ID')['Transaction ID'].nunique().mean()

    start_date = pd.to_datetime(df_orders['Order Date']).min().strftime("%d-%m-%Y")
    end_date = pd.to_datetime(df_orders['Order Date']).max().strftime("%d-%m-%Y")
    total_customers = len(rfm_segmented)
    high_value_customers = (rfm_segmented['segment'] == 'High').sum()
    customers_at_risk = len(at_risk)

    # 3 x 3 KPI layout
    row1 = st.columns(3)
    row1[0].metric("🛒 Avg Order Value", f"₹{aov:,.2f}")
    row1[1].metric("💰 Avg Customer CLTV", f"₹{avg_cltv:,.2f}")
    row1[2].metric("📦 Avg Transactions/User", f"{avg_txns_per_user:.2f}")

    row2 = st.columns(3)
    row2[0].metric("📈 Total Revenue", f"₹{total_revenue:,.0f}")
    row2[1].metric("📆 Data Timeframe", f"{start_date} → {end_date}")
    row2[2].metric("👥 Total Customers", total_customers)

    row3 = st.columns(3)
    row3[0].metric("🌟 High Value Customers", high_value_customers)
    row3[1].metric("⚠️ Customers at Risk*", customers_at_risk)
    row3[2].empty()

    st.caption("📌 *Customers at Risk* refers to users whose **Recency > 90 days**")
    st.divider()
    st.subheader("📈 Visual Insights")


    # Color palette
    segment_colors = {
        'High': '#1f77b4',     
        'Medium': "#5fa2dd",   
        'Low': "#cfe2f3"       
    }

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
        st.markdown("#### 📊 Segment-wise Average Metrics")

        metric_option = st.selectbox(
            "Choose Metric",
            options=[
                "Average Order Value",
                "Average CLTV",
                "Avg Transactions per User",
                "Avg Days Between Orders",
                "Avg Recency",
                "Avg Monetary Value",
                "Predicted Churn Probability"
            ],
            index=0,
            key="segment_metric_option"
        )

        if metric_option == "Average Order Value":
            metric_data = rfm_segmented.groupby("segment")["aov"].mean().reset_index().rename(columns={"aov": "value"})
            y_title = "Average Order Value"
        elif metric_option == "Average CLTV":
            metric_data = rfm_segmented.groupby("segment")["CLTV"].mean().reset_index().rename(columns={"CLTV": "value"})
            y_title = "Average CLTV"
        elif metric_option == "Avg Transactions per User":
            metric_data = rfm_segmented.groupby("segment")["frequency"].mean().reset_index().rename(columns={"frequency": "value"})
            y_title = "Avg Transactions/User"
        elif metric_option == "Avg Days Between Orders":
            metric_data = rfm_segmented.groupby("segment")["avg_days_between_orders"].mean().reset_index().rename(columns={"avg_days_between_orders": "value"})
            y_title = "Avg Days Between Orders"
        elif metric_option == "Avg Recency":
            metric_data = rfm_segmented.groupby("segment")["recency"].mean().reset_index().rename(columns={"recency": "value"})
            y_title = "Average Recency (days)"
        elif metric_option == "Avg Monetary Value":
            metric_data = rfm_segmented.groupby("segment")["monetary"].mean().reset_index().rename(columns={"monetary": "value"})
            y_title = "Avg Monetary Value"
        elif metric_option == "Predicted Churn Probability":
            metric_data = rfm_segmented.groupby("segment")["predicted_churn_prob"].mean().reset_index().rename(columns={"predicted_churn_prob": "value"})
            y_title = "Predicted Churn Probability"

        metric_data['Color'] = metric_data['segment'].map(segment_colors)
        fig2 = px.bar(
            metric_data.sort_values(by='value'),
            x='value',
            y='segment',
            orientation='h',
            labels={'value': y_title},
            color='segment',
            color_discrete_map=segment_colors,
            text='value'
        )

        if "₹" in y_title:
            fig2.update_traces(texttemplate='₹%{text:.2f}')
        elif "Probability" in y_title:
            fig2.update_traces(texttemplate='%{text:.1%}')
        else:
            fig2.update_traces(texttemplate='%{text:.2f}')
        fig2.update_layout(title=f"{y_title} by Segment", xaxis_title=y_title)
        st.plotly_chart(fig2, use_container_width=True)

    # st.markdown("#### 📊 Segment-wise Average Metrics")
    # metric_option = st.selectbox("Choose Metric to Display", options=["AOV", "Average CLTV"], index=0)
    # if metric_option == "AOV":
    #     metric_data = rfm_segmented.groupby("segment")['aov'].mean().reset_index().rename(columns={"aov": "value"})
    #     y_title = "Average Order Value"
    # else:
    #     metric_data = rfm_segmented.groupby("segment")['CLTV'].mean().reset_index().rename(columns={"CLTV": "value"})
    #     y_title = "Average CLTV"

    # metric_data['Color'] = metric_data['segment'].map(segment_colors)
    # fig2 = px.bar(
    #     metric_data.sort_values(by='value'),
    #     x='value',
    #     y='segment',
    #     orientation='h',
    #     labels={'value': y_title},
    #     color='segment',
    #     color_discrete_map=segment_colors,
    #     text='value'
    # )
    # fig2.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    # fig2.update_layout(title=f"{y_title} by Segment", xaxis_title=y_title)
    # st.plotly_chart(fig2, use_container_width=True)

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
                    color_discrete_sequence = [
                            '#08306b',  # Very Dark Blue
                            '#2171b5',  # Mid Blue
                            '#4292c6',  # Light Blue
                            '#6baed6',  # Softer Blue
                            '#9ecae1'   # Pale Blue
                        ]
                        
                )
                fig_products.update_traces(texttemplate='₹%{text:.2f}', textposition='outside')
                fig_products.update_layout(yaxis_title="Total Revenue", xaxis_title="Product ID")
                st.plotly_chart(fig_products, use_container_width=True)
            else:
                st.info("✅ No products found for this segment.")
    except Exception as e:
        st.warning(f"⚠️ Could not compute top products: {e}")

def show_prediction_tab(rfm_segmented):
    st.subheader("🔮 Predicted CLTV (Next 3 Months)")
    st.caption("Forecasted CLTV using BG/NBD + Gamma-Gamma model")
    st.dataframe(
        rfm_segmented[['User ID', 'predicted_cltv_3m']]
        .sort_values(by='predicted_cltv_3m', ascending=False)
        .reset_index(drop=True)
        .style.format({'predicted_cltv_3m': '₹{:,.2f}'}),
        use_container_width=True
    )

    st.markdown("---")

    # Graph Filter BELOW the table
    graph_segment = st.selectbox(
        "📈 Graph Filter by Segment", ["All", "High", "Medium", "Low"],
        index=0, key="graph_segment_filter"
    )

    if graph_segment != "All":
        graph_data = rfm_segmented[rfm_segmented['segment'] == graph_segment].copy()
    else:
        graph_data = rfm_segmented.copy()

    # Sort by actual CLTV for visual clarity
    graph_data = graph_data.sort_values(by='CLTV', ascending=False).reset_index(drop=True)
    graph_data['Index'] = graph_data.index + 1  # For x-axis

    # Create line chart with both actual and predicted
    fig = px.line(
        graph_data, x='Index', y='CLTV', markers=True,
        labels={'Index': 'Customer Index', 'CLTV': 'Historical CLTV'},
        title=f"📊 Historical vs Predicted CLTV ({graph_segment} Segment)"
    )

    fig.add_scatter(
        x=graph_data['Index'],
        y=graph_data['predicted_cltv_3m'],
        mode='markers+lines',
        name='Predicted CLTV (3M)',
        marker=dict(color='green'),
        line=dict(color='green')
    )

    fig.update_layout(
        xaxis_title="Customer Index (sorted by Historical CLTV)",
        yaxis_title="CLTV Value (₹)",
        height=500,
        legend=dict(title="Type")
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🔮 Predicted CLTV Distribution (3-Month)")

    fig_dist = px.histogram(
        graph_data,
        x='predicted_cltv_3m',
        nbins=30,
        color_discrete_sequence=['#5fa2dd']  # solid green
    )
    fig_dist.update_layout(
        xaxis_title="Predicted CLTV",
        yaxis_title="Customer Count",
        title="Distribution of Predicted CLTV (Next 3 Months)"
    )
    st.plotly_chart(fig_dist, use_container_width=True)



def show_detailed_view(rfm_segmented, at_risk):
    st.subheader("📋 Full RFM Segmented Data with CLTV")
    st.dataframe(rfm_segmented)

    st.subheader("⚠️ Customers at Risk (Recency > 90 days)")
    st.dataframe(at_risk)

def show_churn_tab():
    st.subheader("📉 Churn Prediction")

    rfm_segmented = st.session_state['rfm_segmented']
    churned = rfm_segmented[rfm_segmented['predicted_churn'] == 1]

    st.metric("Predicted Churned Customers", len(churned))
    st.metric("Churn Rate (%)", f"{(len(churned) / len(rfm_segmented) * 100):.2f}")

    """st.divider()
    st.markdown("### 🧠 Model Classification Report")
    if 'churn_report' in st.session_state:
        st.dataframe(pd.DataFrame(st.session_state['churn_report']).T.style.format(precision=2), use_container_width=True)"""

    """st.divider()
    st.markdown("### 🧪 Feature Importance")
    if 'churn_importance' in st.session_state:
        feature_cols = ['frequency', 'monetary', 'aov', 'avg_days_between_orders', 'CLTV_30d', 'CLTV_60d', 'CLTV_90d']
        importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': st.session_state['churn_importance']}).sort_values(by='Importance')
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title="Feature Importance")
        st.plotly_chart(fig, use_container_width=True)"""

    st.divider()
    st.markdown("### 🔍 All Customers with Churn Prediction")
    st.dataframe(
        rfm_segmented[['User ID', 'segment', 'frequency', 'aov', 'predicted_cltv_3m',
                       'predicted_churn_prob', 'predicted_churn']]
        .sort_values(by='predicted_churn_prob', ascending=False)
        .style.format({'predicted_churn_prob': '{:.2%}', 'predicted_cltv_3m': '₹{:,.2f}'}),
        use_container_width=True
    )
    st.divider()
    st.markdown("### ⏳ Predicted Days Until Churn")
    st.dataframe(
        rfm_segmented[['User ID', 'segment', 'expected_churn_days', 'predicted_churn_prob']]
        .sort_values(by='expected_churn_days')
        .style.format({'expected_churn_days': '{:.0f}', 'predicted_churn_prob': '{:.2%}'})
    )

def has_duplicate_columns(df1, df2):
    return df1.columns.duplicated().any() or df2.columns.duplicated().any()

if __name__ == "__main__":
    run_streamlit_app()
