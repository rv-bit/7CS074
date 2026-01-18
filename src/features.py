import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder

from global_vars import *

_ENCODER = None
_CATEGORICAL_COLS = None

def fit_feature_encoder(
    df, 
    categorical_cols
):
    """
        Uses OneHotEncoder to create a global encoder for the categorical_columns parsed in the function, 
        Doesn't copy the current dataframe in order to get the categorical columns data, as there are no changes to dataframe this is allowed anything else always copy the dataframe before any changes.
    """
    global _ENCODER, _CATEGORICAL_COLS

    _CATEGORICAL_COLS = categorical_cols

    encoder = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse_output=False
    )

    encoder.fit(df[categorical_cols])
    _ENCODER = encoder

def engineer_features(
    df: pd.DataFrame,
    create_efficiency_score: bool = True
) -> pd.DataFrame:
    """
        Returns a joint dataframe of both numerical values and categorical after one hot encoding values like can be see in src/global_vars/ - CATEGORICAL_FEATURES_GLOBAL, CATEGORICAL_FEATURES_PER_MAKE.
        Creates efficiency score to remove the stress on training on two features, now can be done with one, also better for later clustering like 'Efficient', 'Performance' etc.
    """
    if _ENCODER is None:
        raise RuntimeError("Feature encoder not fitted. Call fit_feature_encoder first.")
    
    df = df.copy()

    # Efficiency score, better for the model, instead of trying to race over which feature decides on the price, just create the efficiency score to find that out per make / model of vehicle
    if create_efficiency_score and {'mpg', 'engineSize'}.issubset(df.columns):
        df['efficiency_score'] = df['mpg'] / df['engineSize']
        df['efficiency_score'].replace([np.inf, -np.inf], np.nan, inplace=True)

    # Numeric
    X_numeric = df[NUMERIC_FEATURES].reset_index(drop=True)

    # Categorical
    X_categorical = _ENCODER.transform(df[_CATEGORICAL_COLS])
    categorical_cols = _ENCODER.get_feature_names_out(_CATEGORICAL_COLS)

    X_categorical = pd.DataFrame(X_categorical, columns=categorical_cols)

    # Returns a joint dataframe of both numeric X-Axis and categorical X-Axis
    return pd.concat([X_numeric, X_categorical], axis=1)

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

def get_feature_names():
    """
        Returns a list for feature names for the categorical columns, so it already gets transformed to 'model_Tiguan', for a VW Car.
    """
    if _ENCODER is None:
        raise RuntimeError("Encoder not fitted")

    return list(NUMERIC_FEATURES) + list(
        _ENCODER.get_feature_names_out(_CATEGORICAL_COLS)
    )