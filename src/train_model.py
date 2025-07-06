import pandas as pd
import joblib
import os
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder

def trainer(ROOT):

    processed_data_path = os.path.join(ROOT, "data", "processed", "cleaned_data.csv")
    model_output_path = os.path.join(ROOT, "models")
    report_path = os.path.join(ROOT, "reports")

    # Create directories if they don't exist
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)

    def load_data():
        df = pd.read_csv(processed_data_path)
        return df

    def encode_categorical(df):
        encoders = {}
        for col in ['Category', 'Accident_type']:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        return df, encoders

    def train_model(df, encoders):
        X = df[['Category', 'Accident_type', 'Year', 'Month']]
        y = df['Value']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        base_model = LGBMRegressor(objective='regression', min_data_in_leaf=5, random_state=42, verbose=-1)

        param_grid = {
            'num_leaves': [15, 31],
            'max_depth': [5, 7],
            'learning_rate': [0.005],
            'n_estimators': [2000],
            'lambda_l1': [0.5],
            'lambda_l2': [0.5],
            'feature_fraction': [0.8, 0.9, 1.0],
            'bagging_fraction': [0.8, 0.9],
            'bagging_freq': [5],
            'min_child_samples': [20, 30, 40],
            'boosting_type': ['gbdt', 'dart']
        }        

        random_search = RandomizedSearchCV(
            estimator = base_model,
            param_distributions = param_grid,
            n_iter = 50,
            scoring='neg_mean_absolute_error',
            cv=3,
            verbose=0,
            random_state=42,
            n_jobs=-1
        )

        random_search.fit(X_train, y_train)

        best_model = random_search.best_estimator_
        y_pred = best_model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)

        # Save model & encoder
        joblib.dump(best_model, os.path.join(model_output_path, "lgbm_model.joblib"))
        joblib.dump(encoders, os.path.join(model_output_path, "label_encoders.joblib"))

        # Save evaluation
        with open(os.path.join(report_path, "model_metrics.txt"), "w") as f:
            f.write(f"MAE: {mae:.2f}\n")
            f.write(f"RMSE: {rmse:.2f}\n")
            f.write(f"Best Params: {random_search.best_params_}\n")
        print(f"[✔] Evaluation metrics saved to: {report_path}")

    df = load_data()
    df_encoded, encoders = encode_categorical(df)
    train_model(df_encoded, encoders)
