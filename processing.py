import numpy as np

def apply_transformations(df):
    df_transformed = df.copy()
    if "NOx (ppm)" in df_transformed.columns:
        df_transformed["NOx (ppm)_sqrt"] = np.sqrt(df_transformed["NOx (ppm)"])
    right_skew = ["HC (ppm)", "Fuel temperature (℃)", "Ignition Advance Angle (°BTDC)",
                  "Fuel consumption (g/s)", "Specific fuel consumption (g/kWh)"]
    for col in right_skew:
        if col in df_transformed.columns:
            df_transformed[col] = df_transformed[col].apply(lambda x: np.log1p(x) if x > 0 else 0)
    left_skew = ["Torque (N m)"]
    for col in left_skew:
        if col in df_transformed.columns:
            max_val = df_transformed[col].max() + 1
            df_transformed[col] = df_transformed[col].apply(lambda x: np.log1p(max_val - x) if x < max_val else 0)
    return df_transformed

def prepare_features(input_df, features_em, features_q):
    df_transformed = apply_transformations(input_df)
    for feat in features_em + features_q:
        if feat not in df_transformed.columns:
            df_transformed[feat] = 0
    X_em = df_transformed[features_em]
    X_q = df_transformed[features_q]
    return X_em, X_q
