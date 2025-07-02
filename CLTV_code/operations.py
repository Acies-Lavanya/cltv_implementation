import pandas as pd

class CustomerAnalytics:
    def __init__(self, transaction_df):
        self.df = transaction_df.copy()
        self.today = self.df['Purchase Date'].max() + pd.Timedelta(days=1)

    def compute_customer_level(self):
        customer_level = self.df.groupby('User ID').agg(
            recency=('Purchase Date', lambda x: (self.today - x.max()).days),
            frequency=('Purchase Date', 'count'),
            monetary=('Total Amount', 'sum'),
            last_purchase=('Purchase Date', 'max'),
            first_purchase=('Purchase Date', 'min')
        ).reset_index()

        customer_level['aov'] = round(customer_level['monetary'] / customer_level['frequency'], 2)
        customer_level['avg_days_between_orders'] = (
    (customer_level['last_purchase'] - customer_level['first_purchase']).dt.days / 
    (customer_level['frequency'] - 1)
)

        # Replace infinite or NaN values (from frequency = 1) with median
        valid_avg = customer_level['avg_days_between_orders'][customer_level['avg_days_between_orders'].notna() & (customer_level['avg_days_between_orders'] != float('inf'))]
        median_gap = valid_avg.median()
        customer_level['avg_days_between_orders'] = customer_level['avg_days_between_orders'].replace([float('inf'), -float('inf')], None)
        customer_level['avg_days_between_orders'] = customer_level['avg_days_between_orders'].fillna(median_gap).round(0).astype(int)

        # Lifespan metrics
        customer_level['lifespan_1d'] = (customer_level['last_purchase'] - customer_level['first_purchase']).dt.days + 1
        customer_level['lifespan_7d'] = round(customer_level['lifespan_1d'] / 7, 2)
        customer_level['lifespan_15d'] = round(customer_level['lifespan_1d'] / 15, 2)
        customer_level['lifespan_30d'] = round(customer_level['lifespan_1d'] / 30, 2)
        customer_level['lifespan_60d'] = round(customer_level['lifespan_1d'] / 60, 2)
        customer_level['lifespan_90d'] = round(customer_level['lifespan_1d'] / 90, 2)
        

        # Time-normalized CLTV
        customer_level['CLTV_1d'] = round(customer_level['monetary'] / customer_level['lifespan_1d'].replace(0, 1), 2)
        customer_level['CLTV_7d'] = round(customer_level['monetary'] / customer_level['lifespan_7d'].replace(0, 0.1), 2)
        customer_level['CLTV_15d'] = round(customer_level['monetary'] / customer_level['lifespan_15d'].replace(0, 0.1), 2)
        customer_level['CLTV_30d'] = round(customer_level['monetary'] / customer_level['lifespan_30d'].replace(0, 0.1), 2)
        customer_level['CLTV_60d'] = round(customer_level['monetary'] / customer_level['lifespan_60d'].replace(0, 0.1), 2)
        customer_level['CLTV_90d'] = round(customer_level['monetary'] / customer_level['lifespan_90d'].replace(0, 0.1), 2)
        customer_level['CLTV_total'] = customer_level['monetary']

        return customer_level

    def rfm_segmentation(self, customer_level):
        customer_level['R_score'] = pd.qcut(customer_level['recency'], 5, labels=[5, 4, 3, 2, 1]).astype(int)
        customer_level['F_score'] = pd.qcut(customer_level['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        customer_level['M_score'] = pd.qcut(customer_level['monetary'], 5, labels=[1, 2, 3, 4, 5]).astype(int)

        customer_level['RFM_score'] = customer_level['R_score'] + customer_level['F_score'] + customer_level['M_score']
        customer_level['RFM'] = customer_level['R_score'].astype(str) + customer_level['F_score'].astype(str) + customer_level['M_score'].astype(str)

        q1 = customer_level['RFM_score'].quantile(0.33)
        q2 = customer_level['RFM_score'].quantile(0.66)

        def assign_segment(score):
            if score <= q1:
                return 'Low'
            elif score <= q2:
                return 'Medium'
            else:
                return 'High'

        customer_level['segment'] = customer_level['RFM_score'].apply(assign_segment)
        return customer_level

    def calculate_cltv(self, df):
        df['CLTV'] = df['aov'] * df['frequency']
        return df

    def customers_at_risk(self, customer_level, threshold_days=90):
        return customer_level[customer_level['recency'] > threshold_days]
    
    def label_churned_customers(self, customer_df, inactive_days_threshold=30):
     customer_df['is_churned'] = (customer_df['recency'] > inactive_days_threshold).astype(int)
     return customer_df

    def get_churn_features(self, customer_df):
        feature_cols = [
            'frequency', 'monetary', 'aov',
            'avg_days_between_orders', 'CLTV_30d', 'CLTV_60d', 'CLTV_90d'
        ]
        X = customer_df[feature_cols]
        y = customer_df['is_churned']
        return X, y
