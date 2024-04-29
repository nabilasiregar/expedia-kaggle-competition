import dask.dataframe as dd

train_dd = dd.read_csv('/Users/sophieengels/Desktop/dmt-2024-2nd-assignment/training_set_VU_DM.csv')
test_dd = dd.read_csv('/Users/sophieengels/Desktop/dmt-2024-2nd-assignment/test_set_VU_DM.csv')
all_data_dd = dd.concat([train_dd, test_dd], ignore_index=True)

unique_ids = sorted(all_data_dd['prop_id'].unique().compute())

hotels_missing_vals = []
for i, prop_id in enumerate(unique_ids):
    if i % 10000 == 0: 
        prop_data = all_data_dd[all_data_dd['prop_id'] == prop_id]
        missing_values = prop_data.isnull().sum()
        
        total_data = len(prop_data)
        percent_missing = (missing_values / total_data) * 100
        
        if percent_missing.any().compute() > 50:
            print(f"Hotel ID with more than 50% missing data: {prop_id}")
            hotels_missing_vals.append(prop_id)
       
print(f"Total number of hotels with more than 50% missing values: {len(hotels_missing_vals)}")