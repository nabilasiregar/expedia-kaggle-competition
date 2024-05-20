## Expedia Kaggle Competition
**Topic**: Recommender Systems

**Task**: Predict what hotel a user is most likely to book

**Description**: The dataset contains information about a search query of a user for a hotel, the hotel properties that resulted and for the training set, whether the user clicked on the hotel and booked it. [source] (https://www.kaggle.com/competitions/dmt-2024-2nd-assignment)

## Installation
Clone the repository

Install the required Python packages:
`pip install -r requirements.txt`

## Project Structure

```bash
├── README.md
├── data
│   ├── preprocessed
│   ├── raw
│   └── submit
├── figures
├── notebooks
│   ├── eda.ipynb
│   ├── feature_engineering.ipynb
│   ├── feature_importance.ipynb
│   ├── models
│   ├── restructuring.ipynb
│   └── xgboost.ipynb
└── scripts
    ├── evaluate.py
    ├── models
    ├── train.py
    └── tune_hyperparams.py
```

## Exploratory Data Analysis
Navigate to the notebooks folder and open eda.ipynb to start the exploratory data analysis.

## Feature Engineering
1. Navigate to the notebooks folder.
2. Open `feature_engineering.ipynb`.
3. Update the file paths to the raw datasets for both training and testing data under the `data/raw/` directory.
4. Execute the notebook separately for the training and testing datasets by updating the filepath for each set

## Hyperparameter Tuning
To tune the hyperparameters, run the following command from the project's root:
`python scripts/tune_hyperparams.py`

## Model Training
To train the model, run the following command from the project's root directory:
`python scripts/train.py`

## Submission File Generation
After training the model:

1. Navigate to the scripts directory.
2. Run evaluate.py to load the trained model and generate submission.csv in the data/submissions/ folder:
`python scripts/evaluate.py`

## Authors
Nabila Siregar, Amir Sahrani, Sophie Engels
