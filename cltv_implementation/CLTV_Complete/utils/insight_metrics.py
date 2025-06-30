# File: utils/insight_metrics.py
import pandas as pd

def generate_insights(customer_df: pd.DataFrame, orders_df: pd.DataFrame):
    insights = {}

    # Total number of customers
    insights["Total Customers"] = customer_df["User ID"].nunique()

    # Average CLTV
    insights["Average CLTV"] = round(customer_df["CLTV"].mean(), 2)

    # Total number of orders
    insights["Total Orders"] = orders_df.shape[0]

    # Segment-level breakdowns
    segment_counts = customer_df.groupby("segment")["User ID"].nunique().to_dict()
    insights["Segment Customers"] = segment_counts

    # Merge segment into orders
    if "User ID" in orders_df.columns:
        merged = orders_df.merge(customer_df[["User ID", "segment"]], on="User ID", how="left")
        segment_order_counts = merged.groupby("segment")["Order ID"].count().to_dict()
        insights["Segment Orders"] = segment_order_counts
    else:
        insights["Segment Orders"] = {seg: "User ID missing in orders_df" for seg in segment_counts.keys()}

    return insights
