import pandas as pd
import xgboost as xgb
from sklearn.metrics import ndcg_score
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 20,
    "axes.titlesize": 26,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "figure.figsize": (8, 6),
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "savefig.format": "png",
    "savefig.transparent": True,
    "axes.grid": True,
    "grid.linewidth": 0.5,
    "grid.linestyle": "--",
    "grid.color": "0.8",
    "image.cmap": "Blues",
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "text.usetex": True, "mathtext.fontset": "cm",
    "pgf.preamble": r"\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage{cmbright}"
})
data = pd.read_csv('../data/preprocessed/engineered_train_set.csv')

data['relevance_grade'] = 5 * data['booking_bool'] + 1 * (data['click_bool'] & ~data['booking_bool'])

features = [col for col in data.columns if col not in ['srch_id', 'prop_id', 'booking_bool', 'click_bool', 'gross_bookings_usd', 'position', 'relevance_grade', 'date_time', 'orig_destination_distance']]

# Split data into training and validation sets based on srch_id % 10
train_data = data[data['srch_id'] % 10 != 1]
valid_data = data[data['srch_id'] % 10 == 1]

# Prepare group information for XGBoost
train_groups = train_data.groupby('srch_id').size().tolist()
valid_groups = valid_data.groupby('srch_id').size().tolist()
full_groups = data.groupby('srch_id').size().tolist()

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(train_data[features], label=train_data['relevance_grade'])
dtrain.set_group(train_groups)
dvalid = xgb.DMatrix(valid_data[features], label=valid_data['relevance_grade'])
dvalid.set_group(valid_groups)
full_dmat = xgb.DMatrix(data[features], label=data['relevance_grade'])
full_dmat.set_group(full_groups)


# using tuned hyperparams
config = {
    'objective': 'rank:ndcg',
    'eval_metric': 'ndcg@5',
    'eta': 0.13963013806537555,
    'max_depth': 9,
    'subsample': 0.7038375178678972,
    'colsample_bytree': 0.7015452447331039,
    'seed': 42
}

# Train the model
num_boost_round = 130
bst = xgb.train(config, dtrain, num_boost_round, evals=[(dvalid, 'validate')], early_stopping_rounds=10)
xgb.plot_importance(bst)
plt.show()

# Predict relevance scores on the validation set
valid_data['predicted_relevance'] = bst.predict(dvalid)

# Sort validation data by srch_id and predicted relevance in descending order
sorted_valid_data = valid_data.sort_values(by=['srch_id', 'predicted_relevance'], ascending=[True, False])

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

ndcg_value = calculate_ndcg(sorted_valid_data, k=5)
print(f"NDCG@5 for the validation set: {ndcg_value}")

# bst = xgb.train(config, full_dmat, num_boost_round, evals=[(full_dmat, 'train')], early_stopping_rounds=10)

bst.save_model('models/final_model.json')
