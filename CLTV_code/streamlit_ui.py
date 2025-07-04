import os
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import plotly.express as px
from input import convert_data_types
from operations import CustomerAnalytics
from mapping import auto_map_columns, expected_orders_cols, expected_transaction_cols
from cltv_model import fit_bgf_ggf
from churn_model import train_churn_model
from cox_model import train_cox_model

BASE_DIR = os.path.dirname(__file__)
SAMPLE_ORDER_PATH = os.path.join(BASE_DIR, "sample_data", "Orders_v2.csv")
SAMPLE_TRANS_PATH = os.path.join(BASE_DIR, "sample_data", "Transactional_v2.csv")

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
            show_detailed_view(st.session_state['rfm_segmented'], st.session_state['at_risk'])
        with tab4:
            show_prediction_tab(st.session_state['rfm_segmented'])
        with tab5:
            show_realization_curve(st.session_state['df_orders'], st.session_state['rfm_segmented'])
        with tab6:
            show_churn_tab()

    else:
        for tab in [tab2, tab3, tab4, tab5, tab6]:
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
        rfm_segmented = analytics.label_churned_customers(rfm_segmented)
        rfm_segmented = analytics.prepare_survival_data(rfm_segmented)

        X, y = analytics.get_churn_features(rfm_segmented)

        model, report, importances, X_test, y_test = train_churn_model(X, y)

        rfm_segmented['predicted_churn_prob'] = model.predict_proba(X)[:, 1]
        rfm_segmented['predicted_churn'] = (rfm_segmented['predicted_churn_prob'] >= 0.5).astype(int)

        predicted_cltv = fit_bgf_ggf(df_transactions)
        rfm_segmented = rfm_segmented.merge(predicted_cltv, on='User ID', how='left')
        rfm_segmented['predicted_cltv_3m'] = rfm_segmented['predicted_cltv_3m'].fillna(0)

        cox_features = ['recency', 'frequency', 'monetary', 'aov', 'avg_days_between_orders',
                'CLTV_30d', 'CLTV_60d', 'CLTV_90d']
        cox_model, rfm_segmented = train_cox_model(rfm_segmented, cox_features)
        st.session_state['cox_model'] = cox_model

        at_risk = analytics.customers_at_risk(rfm_segmented)

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

    start_dt = pd.to_datetime(df_orders['Order Date']).min()
    end_dt = pd.to_datetime(df_orders['Order Date']).max()

    start_date = format_date_with_ordinal(start_dt)
    end_date = format_date_with_ordinal(end_dt)

    total_customers = len(rfm_segmented)
    high_value_customers = (rfm_segmented['segment'] == 'High').sum()
    mid_value_customers = (rfm_segmented['segment'] == "Medium").sum()
    low_value_customers = (rfm_segmented['segment'] == "Low").sum()
    customers_at_risk = len(at_risk)

    # Redesigned 3x3 KPI layout with styled markdown (Card-like)
    def kpi_card(title, value, color="black"):
        st.markdown(f"""
            <div style="background-color:#aee2fd;
                        padding:18px 12px 14px 12px;
                        border-radius:10px;
                        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
                        min-height:100px;
                        color:black;
                        font-family: 'Segoe UI', sans-serif;
                        text-align:center">
                <div style="font-size:16px; font-weight:600; margin-bottom:6px;">{title}</div>
                <div style="font-size:24px; font-weight:bold; color:{color};">{value}</div>
            </div>
        """, unsafe_allow_html=True)

    row1 = st.columns(3, gap="small")
    with row1[0]: kpi_card("📈 Total Revenue", f"₹{total_revenue:,.0f}", color="black")
    with row1[1]: kpi_card("💰 CLTV", f"₹{avg_cltv:,.0f}")
    with row1[2]: kpi_card("🛒 Avg Order Value", f"₹{aov:.0f}")
    row2 = st.columns(3, gap="small")
    with row2[0]: st.text('')
    with row2[1]: st.text('')
    with row2[2]: st.text('')

    row3 = st.columns(3, gap="small")
    with row3[0]: kpi_card("📦 Avg Transactions/User", f"{avg_txns_per_user:.0f}")
    with row3[1]: kpi_card("📆 Data Timeframe", f"{start_date} to {end_date}", color="black")
    with row3[2]: kpi_card("👥 Total Customers", total_customers, color="black")

    #st.caption("📌 *Customers at Risk* refers to users whose **Recency > 70 days**")
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
    segment_counts['Color'] = segment_counts['Segment'].map(segment_colors)

    viz_col1, viz_col2 = st.columns([1, 1.2])
    with viz_col1:
        st.markdown("#### 🎯 Customer Segment Distribution")
        fig1 = px.pie(
            segment_counts,
            values='Count',
            names='Segment',
            hole=0.45,
            color='Segment',
            color_discrete_map=segment_colors
        )
        fig1.update_traces(textinfo='percent+label', textposition='inside')
        st.plotly_chart(fig1, use_container_width=True)
        high_pct = (high_value_customers / total_customers) * 100
        metrics_cols = st.columns(3)
        metrics_cols[0].metric("High Value*", high_value_customers) #f"{high_pct:.1f}%")
        # metrics_cols[1].metric("⚠️ Customers at Risk*", customers_at_risk)
        metrics_cols[1].metric("Medium Value", mid_value_customers)
        metrics_cols[2].metric("Low Value", low_value_customers)
        st.caption("📌 *High Value Customers refers to users whose **RFM Score is in the top 33%.**")
    
    with viz_col2:
        st.markdown("#### 📊 Segment-wise Summary Metrics")

        # Aggregate required metrics
        segment_summary = rfm_segmented.groupby("segment").agg({
            "aov": "mean",
            "CLTV": "mean",
            "frequency": "mean",
            "avg_days_between_orders": "mean",
            "recency": "mean",
            "monetary": "mean"
        }).round(2)

        # Order: High → Medium → Low
        segment_order = ["High", "Medium", "Low"]
        colors = {"High": "#1f77b4", "Medium": "#5fa2dd", "Low": "#9dcbf3"}

        cards = st.columns(3)
        for i, segment in enumerate(segment_order):
            metrics = segment_summary.loc[segment]
            with cards[i]:
                st.markdown(f"""
                    <div style="
                        background-color: {colors[segment]};
                        padding: 20px 15px;
                        border-radius: 12px;
                        color: white;
                        min-height: 250px;
                        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
                        font-family: 'Segoe UI', sans-serif;
                    ">
                        <h4 style="text-align: center; margin-bottom: 15px; font-size: 20px; font-weight: 700;">
                            {segment} Segment
                        </h4>
                        <ul style="list-style: none; padding: 0; font-size: 16px; font-weight: 500; line-height: 1.8;">
                            <li><b>Avg Order Value:</b> ₹{metrics['aov']:,.2f}</li>
                            <li><b>Avg CLTV:</b> ₹{metrics['CLTV']:,.2f}</li>
                            <li><b>Avg Txns/User:</b> {metrics['frequency']:,.2f}</li>
                            <li><b>Days Between Orders:</b> {metrics['avg_days_between_orders']:,.2f}</li>
                            <li><b>Avg Recency:</b> {metrics['recency']:,.0f} days</li>
                            <li><b>Monetary Value:</b> ₹{metrics['monetary']:,.2f}</li>
                        </ul>
                    </div>
            """, unsafe_allow_html=True)
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
                            "#9dcce6"   # Pale Blue
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
    st.caption("Forecasted Customer Lifetime Value using BG/NBD + Gamma-Gamma model.")

    # Table Filter
    table_segment = st.selectbox(
        "📋 Table Filter by Segment", ["All", "High", "Medium", "Low"],
        index=0, key="table_segment_filter"
    )

    if table_segment != "All":
        filtered_df = rfm_segmented[rfm_segmented['segment'] == table_segment].copy()
    else:
        filtered_df = rfm_segmented.copy()

    st.dataframe(
        filtered_df[['User ID', 'segment', 'CLTV', 'predicted_cltv_3m']]
        .sort_values(by='predicted_cltv_3m', ascending=False)
        .reset_index(drop=True)
        .style.format({'CLTV': '₹{:,.2f}', 'predicted_cltv_3m': '₹{:,.2f}'}),
        use_container_width=True
    )

    st.markdown("---")

    # 📊 New Bar Chart: Segment-Wise Avg CLTV Comparison
    st.markdown("#### 📊 Average Historical vs Predicted CLTV per Segment")

    # Prepare segment-wise averages
    segment_comparison = rfm_segmented.groupby('segment')[['CLTV', 'predicted_cltv_3m']].mean().reset_index()

    # Melt the data for grouped bar chart
    segment_melted = segment_comparison.melt(
        id_vars='segment',
        value_vars=['CLTV', 'predicted_cltv_3m'],
        var_name='CLTV Type',
        value_name='Average CLTV'
    )

    # Sort segment order
    segment_order = ['Low', 'Medium', 'High']
    segment_melted['segment'] = pd.Categorical(segment_melted['segment'], categories=segment_order, ordered=True)
    segment_melted = segment_melted.sort_values(by='segment')

    # Create the grouped bar chart
    fig_bar = px.bar(
        segment_melted,
        x='segment',
        y='Average CLTV',
        color='CLTV Type',
        barmode='group',
        labels={'segment': 'Customer Segment', 'Average CLTV': 'Avg CLTV (₹)'},
        color_discrete_map={'CLTV': "#32a2f1", 'predicted_cltv_3m': "#3fd33f"},
        title='Average Historical vs Predicted CLTV per Segment'
    )

    st.plotly_chart(fig_bar, use_container_width=True)


def show_detailed_view(rfm_segmented, at_risk):
    st.subheader("📋 Full RFM Segmented Data with CLTV")
    st.dataframe(rfm_segmented)

    st.subheader("⚠️ Customers at Risk (Recency > 70 days)")
    st.caption("These are customers whose last purchase was over 90 days ago and may be at risk of churning.")
    st.dataframe(at_risk)
def show_churn_tab():
    st.subheader("📉 Churn Prediction Overview")

    rfm_segmented = st.session_state['rfm_segmented']
    churned = rfm_segmented[rfm_segmented['predicted_churn'] == 1]

    st.metric("Predicted Churned Customers", len(churned))
    st.metric("Churn Rate (%)", f"{(len(churned) / len(rfm_segmented) * 100):.2f}")

    st.divider()
    st.markdown("### 📊 Churn Summary by Segment")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔴 Avg Churn Probability")
        churn_by_segment = (
            rfm_segmented
            .groupby("segment")['predicted_churn_prob']
            .mean()
            .reset_index()
            .rename(columns={'predicted_churn_prob': 'Avg Churn Probability'})
        )
        fig_churn = px.bar(
            churn_by_segment.sort_values(by='Avg Churn Probability'),
            x='Avg Churn Probability',
            y='segment',
            orientation='h',
            color='segment',
            color_discrete_map={'High': '#1f77b4', 'Medium': '#5fa2dd', 'Low': '#cfe2f3'},
            text='Avg Churn Probability'
        )
        fig_churn.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig_churn.update_layout(height=350, margin=dict(t=30))
        st.plotly_chart(fig_churn, use_container_width=True)

    with col2:
        st.markdown("#### ⏳ Avg Expected Active Days")
        active_days = (
            rfm_segmented
            .groupby("segment")['expected_active_days']
            .mean()
            .reset_index()
            .rename(columns={'expected_active_days': 'Avg Expected Active Days'})
        )
        fig_days = px.bar(
            active_days.sort_values(by='Avg Expected Active Days'),
            x='Avg Expected Active Days',
            y='segment',
            orientation='h',
            color='segment',
            color_discrete_map={'High': '#1f77b4', 'Medium': '#5fa2dd', 'Low': '#cfe2f3'},
            text='Avg Expected Active Days'
        )
        fig_days.update_traces(texttemplate='%{text:.0f}', textposition='outside')
        fig_days.update_layout(height=350, margin=dict(t=30))
        st.plotly_chart(fig_days, use_container_width=True)

    
    st.divider()
    st.markdown("### 🔍 All Customers at a Glance")

    if st.toggle("🕵️ Detailed View of Churn Analysis"):
         st.dataframe(
        rfm_segmented[['User ID', 'segment', 'predicted_cltv_3m',
                       'predicted_churn_prob', 'predicted_churn','expected_active_days']]
        .sort_values(by='predicted_churn_prob', ascending=False)
        .style.format({'predicted_churn_prob': '{:.2%}', 'predicted_cltv_3m': '₹{:,.2f}'}),
        use_container_width=True
    )

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
   
    """st.divider()
    st.markdown("### ⏳ Predicted Days Until Churn")
    st.dataframe(
        rfm_segmented[['User ID', 'segment', 'expected_churn_days', 'predicted_churn_prob']]
        .sort_values(by='expected_churn_days')
        .style.format({'expected_churn_days': '{:.0f}', 'predicted_churn_prob': '{:.2%}'})
    )"""

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

        fig = px.line(
            chart_df,
            x="Period (Days)",
            y="Avg CLTV per User",
            text="Avg CLTV per User",
            markers=True
        )
        
        fig.update_traces(
            texttemplate='₹%{text:.2f}',
            textposition='top center',
            textfont=dict(size=14, color='black'),  # 👈 Bigger, darker labels
            marker=dict(size=8)
        )
        
        fig.update_layout(
            title={
                'text': f"CLTV Realization Curve - {segment_option}",
                'x': 0.5,
                'xanchor': 'center',
                'font': dict(size=20, color='black')  # Title styling
            },
            xaxis=dict(
                title=dict(text="Days", font=dict(size=16, color='black')),
                tickfont=dict(size=14, color='black')
            ),
            yaxis=dict(
                title=dict(text="Avg CLTV", font=dict(size=16, color='black')),
                tickfont=dict(size=14, color='black')
            ),
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Could not generate CLTV curve: {e}")


def has_duplicate_columns(df1, df2):
    return df1.columns.duplicated().any() or df2.columns.duplicated().any()

# Helper function to add ordinal suffix
def format_date_with_ordinal(date):
    day = int(date.strftime('%d'))
    suffix = 'th' if 11 <= day <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
    return f"{day}{suffix} {date.strftime('%B %Y')}"

if __name__ == "__main__":
    run_streamlit_app()
