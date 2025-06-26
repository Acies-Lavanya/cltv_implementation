import streamlit as st
import pandas as pd
from operations import Customer_level, Rfm_segment
from utils.file_input import handle_file_uploads
from utils.column_mapping import auto_map_columns
from utils.display_helpers import display_file_summary
from pipeline.processor import run_data_pipeline

# Streamlit App Configuration
st.set_page_config(page_title="Upload Transactional and Order Data", layout="wide")
st.title("\U0001F4E5 Upload Transactional and Order Data")

# Expected Column Definitions
target_orders_cols = {
    "Transaction ID": ["Transaction ID", "transaction_id"],
    "Order ID": ["Order ID", "order_id"],
    "Product ID": ["Product ID", "product_id", "SKU", "Item Code"],
    "Quantity": ["Quantity", "Qty", "order_quantity"],
    "Total Amount": ["Total Amount", "total_amount", "amount"],
    "Unit Price": ["unit_price", "price"],
    "Discount Code Used": ["Discount Code Used", "discount_code_used", "promo_code"],
    "Discount Value": ["Discount Value", "discount_value", "discount_amount"],
    "Shipping Cost": ["Shipping Cost", "shipping_cost", "freight"],
    "Total Payable": ["Total Payable", "total_payable", "amount_payable"],
    "Return Status": ["Return Status", "return_stat", "is_returned"],
    "Return Date": ["Return Date", "return_date"]
}

target_transaction_cols = {
    "Transaction ID": ["Transaction ID", "transaction_id"],
    "Visit ID": ["Visit ID", "visit_id"],
    "User ID": ["User ID", "user_id", "Customer ID"],
    "Order ID": ["Order ID", "order_id"],
    "Purchase Date": ["Purchase Date", "purchase_date", "Transaction Date"],
    "Payment Method": ["Payment Method", "payment_method", "Mode of Payment"],
    "Total Amount": ["Total Payable", "total_payable", "amount_payable", "Total_amount"]
}

# Upload Files
orders_df, transactions_df = handle_file_uploads()

if orders_df is not None and transactions_df is not None:
    st.markdown("### \U0001F4CA Uploaded File Overview")
    display_file_summary(orders_df, "Orders")
    display_file_summary(transactions_df, "Transactions")

    if orders_df.columns.duplicated().any() or transactions_df.columns.duplicated().any():
        st.error("❌ Duplicate column names detected. Please fix and re-upload.")
    else:
        orders_mappings = auto_map_columns(orders_df, target_orders_cols)
        trans_mappings = auto_map_columns(transactions_df, target_transaction_cols)

        st.markdown("### ✅ Auto-Mapped Columns")
        mapping_display = pd.DataFrame({
            "Standard Name": list(orders_mappings.keys()) + list(trans_mappings.keys()),
            "Mapped Column": list(orders_mappings.values()) + list(trans_mappings.values()),
            "Sheet": ["Orders"] * len(orders_mappings) + ["Transactions"] * len(trans_mappings)
        })
        st.dataframe(mapping_display, use_container_width=True)

        st.divider()
        st.subheader("✏️ Edit Mappings (Optional)")

        with st.expander("📝 Edit Orders Column Mapping"):
            for k in target_orders_cols:
                orders_mappings[k] = st.selectbox(
                    f"Map for: {k}",
                    orders_df.columns,
                    index=orders_df.columns.get_loc(orders_mappings[k]) if orders_mappings[k] in orders_df.columns else 0,
                    key=f"orders_{k}"
                )

        with st.expander("🧾 Edit Transaction Column Mapping"):
            for k in target_transaction_cols:
                trans_mappings[k] = st.selectbox(
                    f"Map for: {k}",
                    transactions_df.columns,
                    index=transactions_df.columns.get_loc(trans_mappings[k]) if trans_mappings[k] in transactions_df.columns else 0,
                    key=f"transactions_{k}"
                )

        if st.button("✅ Confirm and Process"):
            try:
                processed_orders, processed_trans, customer_segmented = run_data_pipeline(
                    orders_df, transactions_df, orders_mappings, trans_mappings
                )

                st.session_state["orders_df"] = processed_orders
                st.session_state["transactions_df"] = processed_trans
                st.session_state["customer_df"] = customer_segmented

                st.success("🎉 Data processed and mapped successfully!")
                st.subheader("🧍‍♂️ Customer-Level Features with RFM Segmentation")
                st.dataframe(customer_segmented.head(), use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error during processing: {e}")
