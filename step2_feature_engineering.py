import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
import joblib

def feature_engineering():
    df = pd.read_csv('combined_ai_jobs.csv')
    
    # 1. Multi-label binarize required_skills
    print("Binarizing skills...")
    df['skills_list'] = df['required_skills'].apply(lambda x: [s.strip() for s in str(x).split(',')] if pd.notnull(x) else [])
    mlb = MultiLabelBinarizer()
    skills_encoded = mlb.fit_transform(df['skills_list'])
    skills_df = pd.DataFrame(skills_encoded, columns=mlb.classes_)
    
    # 2. Encode experience_level ordinally: EN=0, MI=1, SE=2, EX=3
    print("Encoding experience level...")
    exp_map = {'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3}
    df['experience_encoded'] = df['experience_level'].map(exp_map).fillna(-1) # Handle unknown
    
    # 3. One-hot encode industry, company_size, education_required
    print("One-hot encoding categorical features...")
    cat_cols = ['industry', 'company_size', 'education_required']
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_encoded = ohe.fit_transform(df[cat_cols])
    cat_df = pd.DataFrame(cat_encoded, columns=ohe.get_feature_names_out(cat_cols))
    
    # 4. Merge datasets
    # Base features: experience_encoded, remote_ratio
    base_features = df[['experience_encoded', 'remote_ratio']].reset_index(drop=True)
    
    # Combine all
    final_df = pd.concat([base_features, skills_df, cat_df], axis=1)
    
    # Add targets
    final_df['salary_usd'] = df['salary_usd'].values
    final_df['job_title'] = df['job_title'].values
    
    print(f"Final feature matrix shape: {final_df.shape}")
    
    # Save encoders for later use (Step 5)
    joblib.dump(mlb, 'mlb_skills.joblib')
    joblib.dump(ohe, 'ohe_categorical.joblib')
    
    # Save final dataset
    final_df.to_csv('processed_ai_jobs.csv', index=False)
    print("Saved processed_ai_jobs.csv")

if __name__ == "__main__":
    feature_engineering()
