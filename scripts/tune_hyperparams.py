import pandas as pd
import xgboost as xgb
import numpy as np
import optuna
from sklearn.metrics import ndcg_score

data = pd.read_csv('../data/preprocessed/engineered_train_set.csv')

data['relevance_grade'] = 5 * data['booking_bool'] + 1 * (data['click_bool'] & ~data['booking_bool'])

features = [col for col in data.columns if col not in ['srch_id', 'prop_id', 'booking_bool', 'click_bool', 'gross_bookings_usd', 'position', 'relevance_grade', 'orig_destination_distance', 'date_time']]

# Split data into training and validation sets based on srch_id % 10
train_data = data[data['srch_id'] % 10 != 1]
valid_data = data[data['srch_id'] % 10 == 1]

# Prepare group information for XGBoost
train_groups = train_data.groupby('srch_id').size().tolist()
valid_groups = valid_data.groupby('srch_id').size().tolist()

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(train_data[features], label=train_data['relevance_grade'])
dtrain.set_group(train_groups)
dvalid = xgb.DMatrix(valid_data[features], label=valid_data['relevance_grade'])
dvalid.set_group(valid_groups)

def objective(trial):
    param = {
        'objective': 'rank:ndcg',
        'eval_metric': 'ndcg@5',
        'eta': trial.suggest_loguniform('eta', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'seed': 42
    }
    
    model = xgb.train(param, dtrain, num_boost_round=130, evals=[(dvalid, 'validate')], early_stopping_rounds=10, verbose_eval=False)
    
    valid_data['predicted_relevance'] = model.predict(dvalid)
    sorted_valid_data = valid_data.sort_values(by=['srch_id', 'predicted_relevance'], ascending=[True, False])
    
    ndcg_value = calculate_ndcg(sorted_valid_data, k=5)
    
    return ndcg_value

def dcg_at_k(r, k):
    """Discounted Cumulative Gain at K."""
    r = np.asfarray(r)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    return 0.

def ndcg_at_k(r, k):
    """Normalized Discounted Cumulative Gain at K."""
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

def calculate_ndcg(df, k=5):
    """Calculate NDCG@k averaged over all queries."""
    ndcg_scores = df.groupby('srch_id')['relevance_grade'].apply(lambda x: ndcg_at_k(x.tolist(), k))
    return ndcg_scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

trials_df = study.trials_dataframe()
trials_df.to_csv('../data/tuning_results.csv', index=False)
print(f"Best trial:\n{study.best_trial}")
