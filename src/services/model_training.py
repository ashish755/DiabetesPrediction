import pickle

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from scipy import stats
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier


class DiabetesPrediction:
    def __init__(self) -> None:
        pass

    def encode_label(self, df):
        # Loop over each column in the DataFrame where dtype is 'object'
        for col in df.select_dtypes(include=['object']).columns:

            # Initialize a LabelEncoder object
            label_encoder = preprocessing.LabelEncoder()

            # Fit the encoder to the unique values in the column
            label_encoder.fit(df[col].unique())

            # Transform the column using the encoder
            df[col] = label_encoder.transform(df[col])

        return df

    def resample_label(self, df):
        X = df.drop('diabetes', axis=1)
        y = df['diabetes']

        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X, y)

        # create new DataFrame with undersampled data
        df_resampled = pd.concat([X_resampled, y_resampled], axis=1)

        return df_resampled

    def remove_outliers(self, df_resampled, cols, threshold=3):
        # loop over each selected column
        for col in cols:
            # calculate z-score for each data point in selected column
            z = np.abs(stats.zscore(df_resampled[col]))
            # remove rows with z-score greater than threshold in selected column
            df_resampled = df_resampled[(z < threshold) | (
                df_resampled[col].isnull())]
        return df_resampled

    def data_processing(self, df, selected_cols):
        encoded_df = self.encode_label(df)
        df_resampled = self.resample_label(encoded_df)
        df_clean = self.remove_outliers(df_resampled, selected_cols)

        return df_clean

    def train_model(self, file_path):
        # load file using pandas    
        df = pd.read_csv(file_path)

        selected_cols = ['bmi', 'HbA1c_level', 'blood_glucose_level']

        df_clean = self.data_processing(df, selected_cols)
        X = df_clean.drop('diabetes', axis=1)
        y = df_clean['diabetes']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0)

        dtree = DecisionTreeClassifier()
        param_grid = {
            'max_depth': [3, 4, 5, 6, 7, 8],
            'min_samples_split': [2, 3, 4],
            'min_samples_leaf': [1, 2, 3, 4]
        }

        # Perform a grid search with cross-validation to find the best hyperparameters
        grid_search = GridSearchCV(dtree, param_grid, cv=5)
        grid_search.fit(X_train, y_train)

        max_depth = grid_search.best_params_["max_depth"]
        min_samples_leaf = grid_search.best_params_["min_samples_leaf"]
        min_samples_split = grid_search.best_params_["min_samples_split"]

        dtree = DecisionTreeClassifier(random_state=0, max_depth=max_depth,
                                       min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
        dtree.fit(X_train, y_train)

        # Save the model to a file using pickle
        with open('src/utils/decision_tree_model.pkl', 'wb') as f:
            pickle.dump(dtree, f)


if __name__ == "__main__":
    df = pd.read_csv('diabetes_prediction_dataset.csv')

    diabetes_pred = DiabetesPrediction()
    diabetes_pred.train_model(df)
