from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter, GammaGammaFitter
import pandas as pd

def fit_bgf_ggf(transactions_df):
    # Ensure date column is datetime
    transactions_df = transactions_df.copy()
    
    if 'Purchase Date' not in transactions_df.columns:
        raise KeyError("Missing required column 'Purchase Date' in the transaction dataset.")

    transactions_df['Purchase Date'] = pd.to_datetime(transactions_df['Purchase Date'])

    # Create summary from raw transaction log
    summary_df = summary_data_from_transaction_data(
        transactions_df,
        customer_id_col='User ID',
        datetime_col='Purchase Date',
        monetary_value_col='Total Amount',
        observation_period_end=transactions_df['Purchase Date'].max()
    )

    # Filter out non-repeaters and invalid monetary values
    summary_df = summary_df[(summary_df['frequency'] > 0) & (summary_df['monetary_value'] > 0)]

    if summary_df.empty:
        raise ValueError("No valid data for CLTV prediction.")

    # Fit BG/NBD model
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(summary_df['frequency'], summary_df['recency'], summary_df['T'])

    # Fit Gamma-Gamma model
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(summary_df['frequency'], summary_df['monetary_value'])

    # Predict 3-month (90-day) CLTV
    summary_df['predicted_cltv_3m'] = ggf.customer_lifetime_value(
        bgf,
        summary_df['frequency'],
        summary_df['recency'],
        summary_df['T'],
        summary_df['monetary_value'],
        time=3,  # months
        freq='D',  # frequency of transaction is in days
        discount_rate=0.01  # optional
    )

    summary_df = summary_df.reset_index()
    return summary_df[['User ID', 'predicted_cltv_3m']]
