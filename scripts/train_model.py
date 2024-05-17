import pandas as pd
import xgboost as xg
from sklearn.metrics import ndcg_score
from load_data import DataLoader
import matplotlib.pyplot as plt
import pdb

def create_dmatrix(data, label_column, group_column, drop_columns):
    data_dropped = data.drop(drop_columns, axis=1)
    labels = data[label_column]
    groups = data[group_column].value_counts().sort_index().values
    return xg.DMatrix(data_dropped, label=labels, group=groups)

def train_model(dmatrix, params, num_rounds):
    return xg.train(params, dmatrix, num_boost_round=num_rounds)

def predict_and_evaluate(model, dmatrix, t_data, true_order_columns):
    test_pred = model.predict(dmatrix)
    results_data = t_data.copy()
    results_data['pred'] = test_pred

    ordered_results = results_data.sort_values(['srch_id', 'pred'], ascending=[True, False])
    grouped = ordered_results.groupby('srch_id')['prop_id'].apply(list).reset_index()
    true_order = t_data[true_order_columns]

    grouped['true'] = true_order.groupby('srch_id')['prop_id'].apply(list).reset_index()['prop_id']
    grouped['ndcg'] = grouped.apply(
        lambda x: ndcg_score([x['true']], [x['prop_id']], k=5) if len(x['true']) > 1 else None, axis=1
    )
    return grouped['ndcg'].mean()

if __name__ == "__main__":
    # Load and prepare data
    data, ranking = DataLoader.load_data('../data/preprocessed/engineered_training_set.csv')
    features, target = DataLoader.get_features_and_target(data, ranking)
    train_data, test_data, train_ranking, test_ranking = DataLoader.get_train_test_data(data, ranking)

    # XGBoost configuration
    config = {
        'objective': 'rank:pairwise',
        'lambdarank_pair_method': 'topk',
        'lambdarank_num_pair_per_sample': 3,
        'eval_metric': 'ndcg',
        'learning_rate': 0.013904786971647168,
        'max_depth': 3,
        'subsample': 0.920165421462507,
        'colsample_bytree': 0.7809858505017382,
        'seed': 42
    }
    
    # Create DMatrix objects
    train_dmatrix = create_dmatrix(train_data, 'prop_id', 'srch_id', ['srch_id', 'booking_bool', 'gross_bookings_usd', 'position', 'click_bool', 'prop_id', 'date_time', 'orig_destination_distance'])
    test_dmatrix = create_dmatrix(test_data, 'prop_id', 'srch_id', ['srch_id', 'booking_bool', 'gross_bookings_usd', 'position', 'click_bool', 'prop_id', 'date_time', 'orig_destination_distance'])
    full_dmatrix = xg.DMatrix(
        features.drop(['booking_bool', 'gross_bookings_usd', 'position', 'click_bool', 'prop_id', 'date_time', 'orig_destination_distance'], axis=1),
        label=target,
        group=data['srch_id'].value_counts().sort_index().values
    )

    # Training and plotting importance
    trained_model = train_model(train_dmatrix, config, 100)
    # xg.plot_importance(trained_model)

    # Evaluate model
    mean_ndcg = predict_and_evaluate(trained_model, test_dmatrix, test_data, ['srch_id', 'prop_id'])
    print(f'mean_ndcg: {mean_ndcg}')

    # Full training and save model
    full_model = train_model(full_dmatrix, config, 100)
    full_model.save_model('models/tuned_model.json')

