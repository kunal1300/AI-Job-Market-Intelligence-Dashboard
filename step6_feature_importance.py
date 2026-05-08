import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def plot_feature_importance():
    # Load model and feature names
    try:
        model = joblib.load('xgb_salary_model.joblib')
        feature_names = joblib.load('salary_feature_names.joblib')
    except:
        print("Model or feature names not found. Please run Step 3 first.")
        return

    # Extract importance
    importances = model.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    # Filter for skills (those that are likely from the MLB)
    # We can use the MLB classes to identify skills
    mlb = joblib.load('mlb_skills.joblib')
    skill_features = set(mlb.classes_)
    
    skill_importance = importance_df[importance_df['Feature'].isin(skill_features)]
    
    # Plot top 20 influential skills
    plt.figure(figsize=(12, 8))
    sns.barplot(data=skill_importance.head(20), x='Importance', y='Feature', palette='magma')
    plt.title('Top 20 Skills Driving Salary (XGBoost Feature Importance)')
    plt.xlabel('Importance Score')
    plt.ylabel('Skill')
    plt.tight_layout()
    plt.savefig('skill_importance_salary.png')
    print("Saved skill_importance_salary.png")
    
    # Also plot overall top features
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df.head(20), x='Importance', y='Feature', palette='viridis')
    plt.title('Top 20 Features Driving Salary (Overall)')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('overall_feature_importance.png')
    print("Saved overall_feature_importance.png")

if __name__ == "__main__":
    plot_feature_importance()
