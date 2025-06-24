import pandas as pd

def input_processing(order_path,transaction_path):
    order_df = pd.read_csv(order_path)
    transaction_df = pd.read_csv(transaction_path)

    return order_df,transaction_df

def data_type(order_df, transaction_df):
    transaction_df['purchase_date'] = pd.to_datetime(transaction_df['purchase_date'])
    order_df['return_date'] = pd.to_datetime(order_df['return_date'])
    order_df[['unit_price', 'total_amount', 'discount_value', 'shipping_cost', 'total_payable']] = order_df[['unit_price', 'total_amount', 'discount_value', 'shipping_cost', 'total_payable']].astype('float')
    return order_df, transaction_df


def rfm(order_df, transaction_df):
    today = transaction_df['purchase_date'].max() + pd.Timedelta(days=1)
    transaction_df['total_order_amount'] = order_df.groupby('transaction_id')['total_payable'].sum()
    customer_level = transaction_df.groupby('user_id').agg(
    recency=('purchase_date', lambda x: (today - x.max()).days),
    frequency=('purchase_date', 'count'),
    monetary=('total_order_amount', 'sum'),
    last_purchase=('purchase_date', 'max'),
    first_purchase=('purchase_date', 'min')
).reset_index()

    customer_level['aov'] = round(customer_level['monetary'] / customer_level['frequency'],2)
    customer_level['avg_days_between_orders'] = round((customer_level['last_purchase'] - customer_level['first_purchase']).dt.days / (customer_level['frequency'] - 1),0)
    customer_level['avg_days_between_orders'] = customer_level['avg_days_between_orders'].fillna(-1).astype(int)

    return customer_level

def rfm_segment(customer_level):
    #quantile binning
    customer_level['R_score'] = pd.qcut(customer_level['recency'], 5, labels=[5,4,3,2,1]).astype(int)  # Lower recency = better
    customer_level['F_score'] = pd.qcut(customer_level['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
    customer_level['M_score'] = pd.qcut(customer_level['monetary'], 5, labels=[1,2,3,4,5]).astype(int)
    customer_level['RFM_score'] = customer_level['R_score'] + customer_level['F_score']+ customer_level['M_score']
    customer_level['RFM'] = customer_level['R_score'].astype(str) + customer_level['F_score'].astype(str) + customer_level['M_score'].astype(str)
    q1 = customer_level['RFM_score'].quantile(0.33) # 6.5>=
    q2 = customer_level['RFM_score'].quantile(0.66) # 10.5>=

# Apply segmentation
    def rfm_level(score):
        if score <= q1:
            return 'Low'
        elif score <= q2:
            return 'Medium'
        else:
            return 'High'
    customer_level['segment'] = customer_level['RFM_score'].apply(rfm_level)

