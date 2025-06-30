import streamlit as st
import pandas as pd
from utils.insight_metrics import generate_insights

st.set_page_config(page_title="📊 Insights Dashboard", layout="wide")
st.title("📊 Customer & Order Insights")

# ------------------------------------------
# Check if required data exists in session
# ------------------------------------------
if "customer_df" not in st.session_state or "orders_df" not in st.session_state:
    st.error("❌ Please upload and process data from the main page first.")
else:
    customer_df = st.session_state["customer_df"]
    orders_df = st.session_state["orders_df"]

    # Get insights
    insights = generate_insights(customer_df, orders_df)

    # ------------------------------------------
    # Display Top Metrics
    # ------------------------------------------
    st.markdown("## 📈 Overview Metrics")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🧍‍♂️ Total Customers", insights["Total Customers"])

    with col2:
        st.metric("💸 Average CLTV", f"${insights['Average CLTV']:,.2f}")

    with col3:
        st.metric("🛒 Total Orders", insights["Total Orders"])

    # ------------------------------------------
    # Segment-wise Display
    # ------------------------------------------
    st.markdown("## 🧩 Segment-wise Summary")
    seg_cols = st.columns(3)

    segment_colors = {
        "High": "#D4EDDA",      # light green
        "Medium": "#FFF3CD",    # light yellow
        "Low": "#F8D7DA"        # light red
    }

    for idx, segment in enumerate(["High", "Medium", "Low"]):
        with seg_cols[idx]:
            st.markdown(
                f"""
                <div style="background-color:{segment_colors[segment]}; padding:20px; border-radius:10px;">
                    <h4 style='margin-bottom:10px;'>🔹 {segment} Segment</h4>
                    <p><b>Customers:</b> {insights['Segment Customers'].get(segment, 0)}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # ------------------------------------------
    # RFM Table
    # ------------------------------------------
    st.markdown("---")
    st.subheader("📋 Full RFM Customer Table")
    st.dataframe(customer_df, use_container_width=True)
