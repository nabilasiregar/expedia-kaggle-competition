
import numpy as np
import polars as pl
import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
    def load_data(filepath):
        data = pl.read_csv(filepath)
        data = data.sort(['srch_id', 'booking_bool', 'click_bool'], descending=[False, True, True])
        data_pd = data.to_pandas()
        data_pd = data_pd.replace('NULL', np.nan)
        
        ranking_pd = data_pd[['srch_id', 'prop_id']]

        # Convert object columns to appropriate data types
        object_columns = data_pd.select_dtypes(include=['object']).columns
        data_pd[object_columns] = data_pd[object_columns].apply(pd.to_numeric, errors='coerce')

        return data_pd, ranking_pd
    
    def get_features_and_target(data_pd, ranking_pd):
        # features (X), target (y)
        X = data_pd.drop(['srch_id'], axis=1)
        y = ranking_pd['prop_id']

        return X, y

    def get_train_test_data(data_pd, ranking_pd):
        # Split the data into training and testing sets based on srch_id
        srch_ids = data_pd['srch_id'].unique()
        train_srch_ids, test_srch_ids = train_test_split(srch_ids, test_size=0.2, random_state=42)

        # Create training and testing DataFrames
        train_data = data_pd[data_pd['srch_id'].isin(train_srch_ids)]
        test_data = data_pd[data_pd['srch_id'].isin(test_srch_ids)]

        # Create training and testing ranking DataFrames
        train_ranking = ranking_pd[ranking_pd['srch_id'].isin(train_srch_ids)]
        test_ranking = ranking_pd[ranking_pd['srch_id'].isin(test_srch_ids)]

        return train_data, test_data, train_ranking, test_ranking
