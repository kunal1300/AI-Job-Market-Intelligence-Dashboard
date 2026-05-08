import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
import joblib

def train_salary_models():
    df = pd.read_csv('processed_ai_jobs.csv')
    
    # Define features and target
    # Features specified: skills + experience + industry + remote_ratio
    # Skills are all columns that were binarized (from mlb)
    # Industry are columns starting with 'industry_'
    # experience_encoded and remote_ratio are base features
    
    target = 'salary_usd'
    X = df.drop(columns=['salary_usd', 'job_title'])
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
        "LightGBM": LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    }
    
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'RMSE': rmse, 'R2': r2}
        print(f"{name} - RMSE: {rmse:.2f}, R2: {r2:.4f}")
        
        # Save XGBoost for Feature Importance (Step 6)
        if name == "XGBoost":
            joblib.dump(model, 'xgb_salary_model.joblib')
            joblib.dump(X_train.columns.tolist(), 'salary_feature_names.joblib')

    # Display results
    results_df = pd.DataFrame(results).T
    print("\nModel Performance Comparison:")
    print(results_df)
    results_df.to_csv('salary_prediction_results.csv')

if __name__ == "__main__":
    train_salary_models()
