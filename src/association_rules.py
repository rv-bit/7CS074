import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

def prepare_transactions_from_cars(df: pd.DataFrame, features: list = None) -> pd.DataFrame:
    """
    Prepare car data for association rule mining by creating transactions.
    Each transaction represents a car with its categorical features.
    
    Args:
        df: DataFrame with car data
        features: List of features to include in transactions. 
                 If None, uses default categorical features.
    
    Returns:
        DataFrame in transaction format suitable for apriori algorithm
    """
    if features is None:
        # Default features for association rules
        features = ['make', 'fuelType', 'transmission']
    
    df_copy = df[features].copy()
    
    # Create bins for continuous variables if included
    if 'price' in df_copy.columns:
        df_copy['price_category'] = pd.cut(
            df['price'], 
            bins=[0, 10000, 20000, 30000, np.inf],
            labels=['Budget', 'Mid-Range', 'Premium', 'Luxury']
        )
        df_copy.drop('price', axis=1, inplace=True)
    
    if 'mileage' in df_copy.columns:
        df_copy['mileage_category'] = pd.cut(
            df['mileage'],
            bins=[0, 30000, 60000, 100000, np.inf],
            labels=['Low-Mileage', 'Medium-Mileage', 'High-Mileage', 'Very-High-Mileage']
        )
        df_copy.drop('mileage', axis=1, inplace=True)
    
    if 'engineSize' in df_copy.columns:
        df_copy['engine_category'] = pd.cut(
            df['engineSize'],
            bins=[0, 1.5, 2.5, 4.0, np.inf],
            labels=['Small-Engine', 'Medium-Engine', 'Large-Engine', 'Very-Large-Engine']
        )
        df_copy.drop('engineSize', axis=1, inplace=True)
    
    if 'mpg' in df_copy.columns:
        df_copy['efficiency_category'] = pd.cut(
            df['mpg'],
            bins=[0, 30, 45, 60, np.inf],
            labels=['Low-Efficiency', 'Medium-Efficiency', 'High-Efficiency', 'Very-High-Efficiency']
        )
        df_copy.drop('mpg', axis=1, inplace=True)
    
    # Convert to list of transactions
    transactions = []
    for _, row in df_copy.iterrows():
        transaction = [f"{col}:{val}" for col, val in row.items() if pd.notna(val)]
        transactions.append(transaction)
    
    # Use TransactionEncoder to create one-hot encoded DataFrame
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    return df_encoded

def mine_association_rules(
    df: pd.DataFrame,
    features: list = None,
    min_support: float = 0.01,
    min_confidence: float = 0.1,
    min_lift: float = 1.0,
    metric: str = 'lift'
) -> pd.DataFrame:
    """
    Mine association rules from car dataset.
    
    Args:
        df: DataFrame with car data
        features: List of features to include
        min_support: Minimum support threshold
        min_confidence: Minimum confidence threshold
        min_lift: Minimum lift threshold
        metric: Metric to use for filtering rules
    
    Returns:
        DataFrame with association rules
    """
    # Prepare transactions
    df_transactions = prepare_transactions_from_cars(df, features)
    
    # Find frequent itemsets
    frequent_itemsets = apriori(
        df_transactions, 
        min_support=min_support, 
        use_colnames=True
    )
    
    if frequent_itemsets.empty:
        print(f"No frequent itemsets found with min_support={min_support}")
        return pd.DataFrame()
    
    # Generate association rules
    rules = association_rules(
        frequent_itemsets, 
        metric=metric,
        min_threshold=min_lift if metric == 'lift' else min_confidence
    )
    
    if rules.empty:
        print("No association rules found with given thresholds")
        return pd.DataFrame()
    
    # Filter by confidence if using lift as primary metric
    if metric == 'lift':
        rules = rules[rules['confidence'] >= min_confidence]
    
    # Sort by lift descending
    rules = rules.sort_values('lift', ascending=False)
    
    return rules