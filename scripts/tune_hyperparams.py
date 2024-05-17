import optuna
import pandas as pd
import xgboost as xg
from sklearn.metrics import ndcg_score
from load_data import DataLoader
import matplotlib.pyplot as plt
from train_model import create_dmatrix, train_model, predict_and_evaluate

def load_and_prepare_data():
    """
    Load and prepare data for training and testing.
    """
    data, ranking = DataLoader.load_data('../data/preprocessed/engineered_training_set.csv')
    features, target = DataLoader.get_features_and_target(data, ranking)
    train_data, test_data, train_ranking, test_ranking = DataLoader.get_train_test_data(data, ranking)
    return train_data, test_data, features, target

train_data, test_data, features, target = load_and_prepare_data()

def setup_objective(train_data, test_data):
    """
    Setup the objective function for Optuna to optimize.
    """
    def objective(trial):
        """
        Objective function for Optuna optimization.
        """
        param = {
            'objective': 'rank:pairwise',
            'lambdarank_pair_method': 'topk',
            'lambdarank_num_pair_per_sample': trial.suggest_int('lambdarank_num_pair_per_sample', 2, 10),
            'eval_metric': 'ndcg',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'seed': 42
        }

        train_dmatrix = create_dmatrix(train_data, 'prop_id', 'srch_id', ['srch_id', 'booking_bool', 'gross_bookings_usd', 'position', 'click_bool', 'prop_id', 'date_time', 'orig_destination_distance'])
        model = train_model(train_dmatrix, param, 100)
        test_dmatrix = create_dmatrix(test_data, 'prop_id', 'srch_id', ['srch_id', 'booking_bool', 'gross_bookings_usd', 'position', 'click_bool', 'prop_id', 'date_time', 'orig_destination_distance'])
        mean_ndcg = predict_and_evaluate(model, test_dmatrix, test_data, ['srch_id', 'prop_id'])

        trial_results = {
            'trial_number': trial.number,
            'params': trial.params,
            'mean_ndcg': mean_ndcg
        }
        # trial_results_df = pd.DataFrame([trial_results])
        # trial_results_df.to_csv('optuna_trials.csv', mode='a', header=False, index=False)

        return mean_ndcg

    return objective

# Optuna setup
objective = setup_objective(train_data, test_data)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Plotting and saving results
study.trials_dataframe().to_csv('tuning_results.csv')
# optuna.visualization.plot_optimization_history(study)
# optuna.visualization.plot_param_importances(study)

# Save the best model
# full_dmatrix = xg.DMatrix(
#     features.drop(['booking_bool', 'gross_bookings_usd', 'position', 'click_bool'], axis=1),
#     label=target,
#     group=train_data['srch_id'].value_counts().sort_index().values
# )
# best_params = study.best_trial.params
# full_model = train_model(full_dmatrix, best_params, 100)
# full_model.save_model('models/tuned_model.json')
