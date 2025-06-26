from operations import Customer_level, Rfm_segment
import pandas as pd

def run_data_pipeline(orders_df, trans_df, orders_mapping, trans_mapping):
    # Select and rename columns
    orders_df = orders_df[[orders_mapping[k] for k in orders_mapping]].rename(columns={orders_mapping[k]: k for k in orders_mapping})
    trans_df = trans_df[[trans_mapping[k] for k in trans_mapping]].rename(columns={trans_mapping[k]: k for k in trans_mapping})

    # Convert dates
    if "Return Date" in orders_df.columns:
        orders_df["Return Date"] = pd.to_datetime(orders_df["Return Date"], errors="coerce")
    if "Purchase Date" in trans_df.columns:
        trans_df["Purchase Date"] = pd.to_datetime(trans_df["Purchase Date"], errors="coerce")

    # RFM pipeline
    customer_df = Customer_level().cl_for_transaction(trans_df)
    segmented_df = Rfm_segment().rfm_segment(customer_df)

    return orders_df, trans_df, segmented_df
