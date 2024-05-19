import pandas as pd
import xgboost as xgb

test_data = pd.read_csv('../data/preprocessed/engineered_test_set.csv')

features = [col for col in test_data.columns if col not in ['srch_id', 'prop_id', 'booking_bool', 'click_bool', 'gross_bookings_usd', 'position', 'relevance_grade', 'date_time', 'orig_destination_distance']]

dmatrix_test = xgb.DMatrix(test_data[features])

trained_model = xgb.Booster()
trained_model.load_model('models/final_model.json')

# predict relevance scores using the trained model
test_data['predicted_relevance'] = trained_model.predict(dmatrix_test)

# the highest predicted relevance properties are listed first for each search
sorted_test_data = test_data.sort_values(by=['srch_id', 'predicted_relevance'], ascending=[True, False])

# srch_id and prop_id sorted by predicted relevance
submission = sorted_test_data[['srch_id', 'prop_id']]
submission.to_csv('../data/submit/submission.csv', index=False)
