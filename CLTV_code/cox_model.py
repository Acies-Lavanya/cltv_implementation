# cox_model.py
from lifelines import CoxPHFitter
import pandas as pd

def train_cox_model(df, feature_cols):
    """
    Trains Cox Proportional Hazards model for churn time prediction.
    Returns trained model and predicted expected churn times.
    """
    df = df.copy()
    cph = CoxPHFitter()

    survival_df = df[feature_cols + ['duration', 'event']]
    cph.fit(survival_df, duration_col='duration', event_col='event')

    # Predict expected survival times (expected days until churn)
    df['expected_active_days'] = cph.predict_expectation(survival_df)
    return cph, df
