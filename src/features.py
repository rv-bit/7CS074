from datetime import datetime
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder

import global_vars

def engineer_features(
    df: pd.DataFrame,
    y: pd.Series,
    low_card_categorical_features,
    high_card_categorical_features,
):
    """
        Returns a joint dataframe and the encoder of both numerical values and categorical after one hot encoding and target encoding values like can be see in src/global_vars/ - HIGH_CARD_CATEGORICAL_FEATURES, LOW_CARD_CATEGORICAL_FEATURES.
        Creates efficiency score to remove the stress on training on two features (mpg and engine size) to determine price, instead just create efficiency score to let model decide on that.
        Creates vehicle age to better determine the age of vehicle instead of year of manufacture.
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
    
    # We will create vehicle_age as the primary column to create a better relationship between price and age of vehicle
    if {'year'}.issubset(df.columns):
        current_year = datetime.now().year
        df['vehicle_age'] = np.where(
            df['year'] > 0,
            current_year - df['year'],
            np.nan 
        )

    X_numeric = df[global_vars.NUMERIC_FEATURES]

    low_card_encoder = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse_output=False
    )
    low_card_encoder.fit(df[low_card_categorical_features])
    X_train_low_card = low_card_encoder.transform(df[low_card_categorical_features])
    low_categorical_cols = low_card_encoder.get_feature_names_out(
        low_card_categorical_features
    )
    X_low_df = pd.DataFrame(
        X_train_low_card,
        columns=low_categorical_cols,
        index=df.index
    )
    
    
    high_card_encoder = TargetEncoder(
        cols=high_card_categorical_features,
        smoothing=10
    )
    X_train_high_card = high_card_encoder.fit_transform(df[high_card_categorical_features], y)
    high_categorical_cols = high_card_encoder.get_feature_names_out(
        high_card_categorical_features
    )
    X_high_df = pd.DataFrame(
        X_train_high_card,
        columns=high_categorical_cols,
        index=df.index
    )
    
    X_Concatenated = pd.concat(
        [
            X_numeric, 
            X_low_df,
            X_high_df
        ],
        axis=1
    )
    
    X_Concatenated.drop(['year', 'mpg', 'engineSize'], axis=1, inplace=True)

    # Returns a joint dataframe of both numeric X-Axis and categorical X-Axis
    return X_Concatenated

def get_feature_effects(model):
    """
        Returns (values, kind) for feature effect visualization. Since we are using different models, we need to extract feature importance or coefficients based on model type.
        Supported models:
        - Tree models -> feature_importances_ - https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
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