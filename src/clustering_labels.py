import pandas as pd

def assign_cluster_labels(df: pd.DataFrame, cluster_col: str = 'cluster') -> pd.DataFrame:
    """
    Assigns meaningful labels to clusters based on car characteristics.
    
    Labels:
    - 'Luxury': High price, low mileage, premium features
    - 'Performance': High engine size, moderate-high price
    - 'Efficient': High MPG, smaller engine size
    - 'Family SUV': Moderate price, good MPG, practical
    - 'Budget': Low price, higher mileage
    - 'Electric/Hybrid': Electric or Hybrid fuel type
    """
    df = df.copy()
    
    # Calculate cluster statistics
    cluster_stats = df.groupby(cluster_col).agg({
        'price': 'mean',
        'mileage': 'mean',
        'mpg': 'mean',
        'engineSize': 'mean',
        'fuelType': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
    }).reset_index()
    
    # Create label mapping
    label_map = {}
    
    for _, row in cluster_stats.iterrows():
        cluster_id = row[cluster_col]
        
        # Check for Electric/Hybrid first
        if row['fuelType'] in ['Electric', 'Hybrid']:
            label_map[cluster_id] = 'Electric/Hybrid'
        # Luxury: high price, low mileage
        elif row['price'] > df['price'].quantile(0.75) and row['mileage'] < df['mileage'].quantile(0.25):
            label_map[cluster_id] = 'Luxury'
        # Performance: large engine, high price
        elif row['engineSize'] > df['engineSize'].quantile(0.75) and row['price'] > df['price'].median():
            label_map[cluster_id] = 'Performance'
        # Efficient: high MPG, smaller engine
        elif row['mpg'] > df['mpg'].quantile(0.75) and row['engineSize'] < df['engineSize'].median():
            label_map[cluster_id] = 'Efficient'
        # Budget: low price, high mileage
        elif row['price'] < df['price'].quantile(0.25):
            label_map[cluster_id] = 'Budget'
        # Family SUV: everything else (moderate across board)
        else:
            label_map[cluster_id] = 'Family SUV'
    
    df['cluster_label'] = df[cluster_col].map(label_map)
    
    return df, label_map