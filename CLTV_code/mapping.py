# mapping.py

import difflib

# Expected standard column mappings for orders
expected_orders_cols = {
    "Transaction ID": ["Transaction ID", "transaction_id"],
    "Order ID": ["Order ID", "order_id"],
    "Product ID": ["Product ID", "product_id", "SKU", "Item Code"],
    "Quantity": ["Quantity", "Qty", "order_quantity"],
    "Total Amount": ["Total Amount", "total_amount", "amount"],
    "Unit Price": ["unit_price", "price"],
    "Order Date": ["Order Date", "order_date", "Order_date"],
    "Discount Code Used": ["Discount Code Used", "discount_code_used", "promo_code"],
    "Discount Value": ["Discount Value", "discount_value", "discount_amount"],
    "Shipping Cost": ["Shipping Cost", "shipping_cost", "freight"],
    "Total Payable": ["Total Payable", "total_payable", "amount_payable"],
    "Return Status": ["Return Status", "return_stat", "is_returned"],
    "Return Date": ["Return Date", "return_date"]
}

# Expected standard column mappings for transactions
expected_transaction_cols = {
    "Transaction ID": ["Transaction ID", "transaction_id"],
    "Visit ID": ["Visit ID", "visit_id"],
    "User ID": ["User ID", "user_id", "Customer ID"],
    "Order ID": ["Order ID", "order_id"],
    "Purchase Date": ["Purchase Date", "purchase_date", "Transaction Date"],
    "Payment Method": ["Payment Method", "payment_method", "Mode of Payment"],
    "Total Amount": ["Total Payable", "total_payable", "amount_payable", "Total_amount"]
}

def auto_map_column(column_list, candidate_names):
    for name in candidate_names:
        match = difflib.get_close_matches(name, column_list, n=1, cutoff=0.6)
        if match:
            return match[0]
    return None

def auto_map_columns(df, mapping_dict):
    return {
        standard_name: auto_map_column(df.columns.tolist(), candidates) or ""
        for standard_name, candidates in mapping_dict.items()
    }
