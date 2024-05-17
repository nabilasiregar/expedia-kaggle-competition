
import xgboost as xg
import polars as pl
import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
import pdb

model = xg.Booster()
model.load_model('models/fixed_model.json')

test_set = pl.read_csv('../data/preprocessed/engineered_test_set.csv')
test_set = test_set.to_pandas()
test_set = test_set.replace('NULL', np.nan)

object_columns = test_set.select_dtypes(include=['object']).columns
test_set[object_columns] = test_set[object_columns].apply(pd.to_numeric, errors='coerce')

test_set_dmatrix = xg.DMatrix(test_set.drop(['srch_id', 'date_time', 'orig_destination_distance', 'prop_id'], axis=1), group=test_set['srch_id'].value_counts().sort_index().values)
test_set['pred'] = model.predict(test_set_dmatrix)

# same as earlier, without need for calculating the ndcg, so less steps
submission = test_set.sort_values(['srch_id', 'pred'], ascending=[True, False])[['srch_id', 'prop_id']]
submission.to_csv('../data/submit/new_submission.csv', index=False)