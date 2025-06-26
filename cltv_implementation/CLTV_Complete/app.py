import pandas as pd
from config import order,transaction
from input import input_processing, data_type
from operations import Customer_level, Rfm_segment


order_df,transaction_df = input_processing(order,transaction)
# print(order_df.columns)

order_df, transaction_df = data_type(order_df, transaction_df)
master_data = Customer_level()
customer_level = master_data.cl_for_transaction(transaction_df)
print(customer_level)

master_data = Rfm_segment()
customer = master_data.rfm_segment(customer_level)
print(customer)