import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

def train_job_classifier():
    df = pd.read_csv('processed_ai_jobs.csv')
    
    # 1. Prepare target
    # Roadmap specifies 20 classes. Let's take the top 20 job titles.
    top_20_titles = df['job_title'].value_counts().head(20).index
    df_filtered = df[df['job_title'].isin(top_20_titles)].copy()
    
    le = LabelEncoder()
    y = le.fit_transform(df_filtered['job_title'])
    
    # 2. Prepare features
    # Features specified: skills + education + experience
    # Education columns start with 'education_required_'
    # Skills are from the MLB (binary columns)
    # Experience is experience_encoded
    
    # Actually, we can just use the features we already have, 
    # but filter out industry and remote_ratio if they aren't explicitly requested for this model.
    # Roadmap says: features: skills + education + experience
    
    cols_to_keep = [c for c in df_filtered.columns if c not in ['salary_usd', 'job_title'] and not c.startswith('industry_') and c != 'remote_ratio' and not c.startswith('company_size_')]
    X = df_filtered[cols_to_keep]
    
    print(f"Features used for classification: {X.columns.tolist()[:10]}... (Total {len(X.columns)})")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='macro')
        results[name] = {'F1-Score (macro)': f1}
        print(f"{name} - F1-Score (macro): {f1:.4f}")
    
    # Display results
    results_df = pd.DataFrame(results).T
    print("\nModel Performance Comparison:")
    print(results_df)
    results_df.to_csv('job_classification_results.csv')
    
    # Save best model (let's say XGBoost or RF) for future use
    joblib.dump(le, 'job_title_encoder.joblib')
    joblib.dump(X_train.columns.tolist(), 'classifier_feature_names.joblib')

if __name__ == "__main__":
    train_job_classifier()
