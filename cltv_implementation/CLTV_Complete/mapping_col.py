import streamlit as st
import pandas as pd
import difflib

# ---------------------------------
# Streamlit App Configuration
# ---------------------------------
st.set_page_config(page_title="Upload Transactional and Order Data", layout="wide")
st.title("📥 Upload Transactional and Order Data")

# ---------------------------------
# Expected Column Definitions (Complete Mapping)
# ---------------------------------
expected_orders_cols = {
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

expected_transaction_cols = {
    "Transaction ID": ["Transaction ID", "transaction_id"],
    "Visit ID": ["Visit ID", "visit_id"],
    "User ID": ["User ID", "user_id", "Customer ID"],
    "Order ID": ["Order ID", "order_id"],
    "Purchase Date": ["Purchase Date", "purchase_date", "Transaction Date"],
    "Payment Method": ["Payment Method", "payment_method", "Mode of Payment"],
    "Total Amount": ["Total Payable", "total_payable", "amount_payable", "Total_amount"],
}

# ---------------------------------
# Helper Function: Auto Map Columns
# ---------------------------------
def auto_map(col_list, candidates):
    for name in candidates:
        match = difflib.get_close_matches(name, col_list, n=1, cutoff=0.6)
        if match:
            return match[0]
    return None

# ---------------------------------
# Upload Section
# ---------------------------------
orders_file = st.file_uploader("Upload Orders CSV", type=["csv"], key="orders_upload")
transactions_file = st.file_uploader("Upload Transactions CSV", type=["csv"], key="transactions_upload")

if orders_file and transactions_file:
    df_orders = pd.read_csv(orders_file)
    df_transactions = pd.read_csv(transactions_file)
    
    # ---------------------------------
    # Show Dataset Info Before Mapping
    # ---------------------------------
    st.markdown("### 📊 Uploaded File Overview")

    # Orders Info
    with st.expander("📦 Orders CSV Info", expanded=True):
        st.write(f"**Shape:** {df_orders.shape[0]} rows × {df_orders.shape[1]} columns")
        
        st.markdown("**🔤 Columns and Data Types:**")
        st.dataframe(pd.DataFrame({
            "Column": df_orders.columns,
            "Data Type": [str(dtype) for dtype in df_orders.dtypes]
        }))

        st.markdown("**❗ Missing Values:**")
        st.dataframe(
            df_orders.isnull().sum().reset_index().rename(columns={0: "Missing Values", "index": "Column"})
        )

        st.markdown("**👁️ Preview (First 5 Rows):**")
        st.dataframe(df_orders.head(), use_container_width=True)

        st.markdown("**📈 Summary Statistics (Numeric Columns):**")
        st.dataframe(df_orders.describe().T, use_container_width=True)

    # Transactions Info
    with st.expander("🧾 Transactions CSV Info", expanded=True):
        st.write(f"**Shape:** {df_transactions.shape[0]} rows × {df_transactions.shape[1]} columns")
        
        st.markdown("**🔤 Columns and Data Types:**")
        st.dataframe(pd.DataFrame({
            "Column": df_transactions.columns,
            "Data Type": [str(dtype) for dtype in df_transactions.dtypes]
        }))

        st.markdown("**❗ Missing Values:**")
        st.dataframe(
            df_transactions.isnull().sum().reset_index().rename(columns={0: "Missing Values", "index": "Column"})
        )

        st.markdown("**👁️ Preview (First 5 Rows):**")
        st.dataframe(df_transactions.head(), use_container_width=True)

        st.markdown("**📈 Summary Statistics (Numeric Columns):**")
        st.dataframe(df_transactions.describe().T, use_container_width=True)

    # Check for duplicate columns
    if df_orders.columns.duplicated().any() or df_transactions.columns.duplicated().any():
        st.error("❌ Duplicate column names detected. Please fix and re-upload.")
        
    else:
        # Auto Mapping Columns
        orders_mappings = {
            k: auto_map(df_orders.columns.tolist(), v) or ""
            for k, v in expected_orders_cols.items()
        }
        trans_mappings = {
            k: auto_map(df_transactions.columns.tolist(), v) or ""
            for k, v in expected_transaction_cols.items()
        }

        # Display auto-mapping
        st.markdown("### ✅ Auto-Mapped Columns")
        mapping_display = pd.DataFrame({
            "Standard Name": list(orders_mappings.keys()) + list(trans_mappings.keys()),
            "Mapped Column": list(orders_mappings.values()) + list(trans_mappings.values()),
            "Sheet": ["Orders"] * len(orders_mappings) + ["Transactions"] * len(trans_mappings)
        })
        st.dataframe(mapping_display, use_container_width=True)

        st.divider()
        st.subheader("✏️ Edit Mappings (Optional)")

        # Editable Mappings
        with st.expander("📝 Edit Orders Column Mapping"):
            for k in expected_orders_cols:
                orders_mappings[k] = st.selectbox(
                    f"Map for: {k}",
                    df_orders.columns,
                    index=df_orders.columns.get_loc(orders_mappings[k]) if orders_mappings[k] in df_orders.columns else 0,
                    key=f"orders_{k}"
                )

        with st.expander("🧾 Edit Transaction Column Mapping"):
            for k in expected_transaction_cols:
                trans_mappings[k] = st.selectbox(
                    f"Map for: {k}",
                    df_transactions.columns,
                    index=df_transactions.columns.get_loc(trans_mappings[k]) if trans_mappings[k] in df_transactions.columns else 0,
                    key=f"transactions_{k}"
                )

        # Confirm Button
        if st.button("✅ Confirm and Process"):
            try:
                # Process Orders Data
                orders_df = df_orders[
                    [orders_mappings[k] for k in expected_orders_cols]
                ].rename(columns={orders_mappings[k]: k for k in expected_orders_cols})

                # Process Transactions Data
                trans_df = df_transactions[
                    [trans_mappings[k] for k in expected_transaction_cols]
                ].rename(columns={trans_mappings[k]: k for k in expected_transaction_cols})

                # Convert dates
                if "Return Date" in orders_df.columns:
                    orders_df["Return Date"] = pd.to_datetime(orders_df["Return Date"], errors="coerce")
                if "Purchase Date" in trans_df.columns:
                    trans_df["Purchase Date"] = pd.to_datetime(trans_df["Purchase Date"], errors="coerce")

                # Output
                st.success("🎉 Data processed and mapped successfully!")
                st.subheader("📋 Orders Sample")
                st.dataframe(orders_df.head(), use_container_width=True)
                st.subheader("📋 Transactions Sample")
                st.dataframe(trans_df.head(), use_container_width=True)

                # Save in session
                st.session_state["orders_df"] = orders_df
                st.session_state["transactions_df"] = trans_df

            except Exception as e:
                st.error(f"❌ Error during processing: {e}")