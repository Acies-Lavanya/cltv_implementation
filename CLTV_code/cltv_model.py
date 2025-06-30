from lifetimes import BetaGeoFitter, GammaGammaFitter

def fit_bgf_ggf(df):
    model_df = df[df['frequency'] > 1][['User ID', 'frequency', 'recency', 'monetary']]
    model_df = model_df.copy()
    model_df.columns = ['customer_id', 'frequency', 'recency', 'monetary_value']
    model_df['T'] = df['recency'].max()  # Assuming max recency as observation period

    bgf = BetaGeoFitter()
    bgf.fit(model_df['frequency'], model_df['recency'], model_df['T'])

    ggf = GammaGammaFitter()
    ggf.fit(model_df['frequency'], model_df['monetary_value'])

    model_df['predicted_purchases_3m'] = bgf.conditional_expected_number_of_purchases_up_to_time(
        90, model_df['frequency'], model_df['recency'], model_df['T'])

    model_df['expected_avg_profit'] = ggf.conditional_expected_average_profit(
        model_df['frequency'], model_df['monetary_value'])

    model_df['predicted_cltv_3m'] = model_df['predicted_purchases_3m'] * model_df['expected_avg_profit']

    return model_df[['customer_id', 'predicted_cltv_3m']]