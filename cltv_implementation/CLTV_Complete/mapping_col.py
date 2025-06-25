import streamlit as st
import pandas as pd
import difflib

# ---------------------------------
# Streamlit App Configuration
# ---------------------------------
st.set_page_config(page_title="Upload Inventory Data", layout="wide")
st.title("📥 Upload Inventory Data")

# ---------------------------------
# Expected Column Definitions
# ---------------------------------
expected_orders_cols = {
    "Order Date": ["Order Date", "Date", "Order_Date", "OrderDate"],
    "SKU ID": ["SKU ID", "SKU", "Product Code", "Item Code"],
    "Order Quantity": ["Order Quantity", "Quantity", "Qty", "Order Qty"]
}

expected_stock_cols = {
    "SKU ID": ["SKU ID", "SKU", "Product Code"],
    "Current Stock Quantity": ["Current Stock Quantity", "Stock", "Available Stock"],
    "Units (Nos/Kg)": ["Units", "Nos", "Kg", "Unit Type", "Units (Nos/Kg)"],
    "Average Lead Time (days)": ["Average Lead Time", "Avg Lead Time", "Lead Time"],
    "Maximum Lead Time (days)": ["Maximum Lead Time", "Max Lead Time"],
    "Unit Price": ["Unit Price", "Price"],
    "Safety Stock": ["Safety Stock", "Buffer Stock"]
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
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    st.session_state["uploaded_file"] = uploaded_file

if "uploaded_file" in st.session_state:
    uploaded_file = st.session_state["uploaded_file"]
    try:
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names
        st.success(f"✅ Uploaded. Sheets found: {sheet_names}")

        order_sheet = st.selectbox("Select Past Orders Sheet", sheet_names, key="order_sheet")
        stock_sheet = st.selectbox("Select Stock Sheet", sheet_names, key="stock_sheet")

        df_orders = pd.read_excel(uploaded_file, sheet_name=order_sheet)
        df_stock = pd.read_excel(uploaded_file, sheet_name=stock_sheet)

        # Check for duplicate columns
        if df_orders.columns.duplicated().any() or df_stock.columns.duplicated().any():
            st.error("❌ Duplicate column names detected in your file. Please fix them and re-upload.")
        else:
            # Auto Mapping Columns
            order_mappings = {
                k: auto_map(df_orders.columns.tolist(), v) or ""
                for k, v in expected_orders_cols.items()
            }
            stock_mappings = {
                k: auto_map(df_stock.columns.tolist(), v) or ""
                for k, v in expected_stock_cols.items()
            }

            # Display auto-mapping
            st.markdown("### ✅ Auto-Mapped Columns")
            mapping_display = pd.DataFrame({
                "Standard Name": list(order_mappings.keys()) + list(stock_mappings.keys()),
                "Mapped Column": list(order_mappings.values()) + list(stock_mappings.values()),
                "Sheet": ["Past Orders"] * len(order_mappings) + ["Stock"] * len(stock_mappings)
            })
            st.dataframe(mapping_display, use_container_width=True)

            st.divider()
            st.subheader("✏️ Edit Mappings (Optional)")

            # Editable Mappings
            with st.expander("📝 Edit Past Orders Column Mapping"):
                for k in expected_orders_cols:
                    order_mappings[k] = st.selectbox(
                        f"Map for: {k}",
                        df_orders.columns,
                        index=df_orders.columns.get_loc(order_mappings[k]) if order_mappings[k] in df_orders.columns else 0,
                        key=f"order_{k}"
                    )

            with st.expander("📦 Edit Stock Column Mapping"):
                for k in expected_stock_cols:
                    stock_mappings[k] = st.selectbox(
                        f"Map for: {k}",
                        df_stock.columns,
                        index=df_stock.columns.get_loc(stock_mappings[k]) if stock_mappings[k] in df_stock.columns else 0,
                        key=f"stock_{k}"
                    )

            # Confirm Button
            if st.button("✅ Confirm and Merge Data"):
                try:
                    # Select and rename relevant columns for orders
                    orders_df = df_orders[
                        [order_mappings["Order Date"], order_mappings["SKU ID"], order_mappings["Order Quantity"]]
                    ].rename(columns={
                        order_mappings["Order Date"]: "Order Date",
                        order_mappings["SKU ID"]: "SKU ID",
                        order_mappings["Order Quantity"]: "Order Quantity"
                    })

                    # Ensure datetime
                    orders_df["Order Date"] = pd.to_datetime(orders_df["Order Date"], errors="coerce")

                    # Select and rename relevant columns for stock
                    stock_df = df_stock[
                        [stock_mappings["SKU ID"], stock_mappings["Current Stock Quantity"], stock_mappings["Units (Nos/Kg)"],
                         stock_mappings["Average Lead Time (days)"], stock_mappings["Maximum Lead Time (days)"],
                         stock_mappings["Unit Price"], stock_mappings["Safety Stock"]]
                    ].rename(columns={
                        stock_mappings["SKU ID"]: "SKU ID",
                        stock_mappings["Current Stock Quantity"]: "Current Stock Quantity",
                        stock_mappings["Units (Nos/Kg)"]: "Units (Nos/Kg)",
                        stock_mappings["Average Lead Time (days)"]: "Average Lead Time (days)",
                        stock_mappings["Maximum Lead Time (days)"]: "Maximum Lead Time (days)",
                        stock_mappings["Unit Price"]: "Unit Price",
                        stock_mappings["Safety Stock"]: "Safety Stock"
                    })

                    # Remove duplicate columns if any
                    stock_df = stock_df.loc[:, ~stock_df.columns.duplicated()]

                    # ---- Aggregations ----
                    agg_orders = orders_df.groupby("SKU ID").agg({
                        "Order Quantity": ["sum", "mean", "std"]
                    }).reset_index()
                    agg_orders.columns = ["SKU ID", "Order Quantity sum", "Order Quantity mean", "Order Quantity std"]

                    # Last Order Date
                    last_order_df = orders_df.groupby("SKU ID")["Order Date"].max().reset_index(name="Last Order Date")

                    # Median Days Between Orders
                    def compute_median_days(group):
                        group = group.sort_values("Order Date")
                        group["Days Between Orders"] = group["Order Date"].diff().dt.days
                        return pd.Series({
                            "Median Days Between Orders": group["Days Between Orders"].median()
                        })

                    median_days_df = orders_df.groupby("SKU ID").apply(compute_median_days).reset_index()

                    # Merge all
                    merged_df = pd.merge(stock_df, agg_orders, on="SKU ID", how="left")
                    merged_df = pd.merge(merged_df, last_order_df, on="SKU ID", how="left")
                    merged_df = pd.merge(merged_df, median_days_df, on="SKU ID", how="left")

                    # Drop duplicate columns again if needed
                    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

                    # Fill missing values
                    merged_df.fillna(0, inplace=True)

                    # Output
                    st.success("🎉 Data merged and enriched successfully!")
                    st.dataframe(merged_df, use_container_width=True)

                    # Store in session
                    st.session_state["orders_df"] = orders_df
                    st.session_state["stock_df"] = stock_df
                    st.session_state["merged_df"] = merged_df

                except Exception as e:
                    st.error(f"❌ Error during processing: {e}")

    except Exception as e:
        st.error(f"❌ Failed to read Excel file: {e}")