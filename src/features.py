import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder

import global_vars

def engineer_features(
    df: pd.DataFrame,
    categorical_features
):
    """
        Returns a joint dataframe and the encoder of both numerical values and categorical after one hot encoding values like can be see in src/global_vars/ - CATEGORICAL_FEATURES_GLOBAL, CATEGORICAL_FEATURES_PER_MAKE.
        Creates efficiency score to remove the stress on training on two features, now can be done with one, also better for later clustering like 'Efficient', 'Performance' etc.
    """
    df = df.copy()
    
    # Efficiency score, better for the model, instead of trying to race over which feature decides on the price, just create the efficiency score to find that out per make / model of vehicle
    # Will onl create if engineSize is higher than 0, making sure we do not create bad data for model to be trained, so we just set as nan if logic fails
    if {'mpg', 'engineSize'}.issubset(df.columns):
        df['efficiency_score'] = np.where(
            df['engineSize'] > 0,
            df['mpg'] / df['engineSize'],
            np.nan
        )

    encoder = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse_output=False
    )
    encoder.fit(df[categorical_features])

    # Build features manually (no globals)
    X_numeric = df[global_vars.NUMERIC_FEATURES].reset_index(drop=True)

    X_categorical = encoder.transform(df[categorical_features])
    categorical_cols = encoder.get_feature_names_out(
        categorical_features
    )
    
    X_Concatenated = pd.concat(
        [X_numeric, pd.DataFrame(X_categorical, columns=categorical_cols)],
        axis=1
    )

    # Returns a joint dataframe of both numeric X-Axis and categorical X-Axis
    return X_Concatenated, encoder

def get_feature_effects(model):
    """
        Returns (values, kind) for feature effect visualization.
        - Tree models -> feature_importances_
        - Linear models -> abs(coef_)
    """
    # RandomForest, GradientBoosting, etc.
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        return values, "importance"

    # LinearRegression, Ridge, Lasso, etc.
    if hasattr(model, "coef_"):
        values = np.abs(model.coef_)
        return values, "coefficient"

    # Unsupported model
    return None, None