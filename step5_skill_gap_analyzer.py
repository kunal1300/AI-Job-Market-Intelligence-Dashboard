import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def skill_gap_analysis(user_skills, target_job_title):
    # Load data
    df = pd.read_csv('processed_ai_jobs.csv')
    mlb = joblib.load('mlb_skills.joblib')
    
    # 1. Calculate centroids for each job title
    # We only care about the skill columns
    skill_cols = mlb.classes_
    centroids = df.groupby('job_title')[skill_cols].mean()
    
    if target_job_title not in centroids.index:
        return f"Job title '{target_job_title}' not found in dataset."
    
    # 2. Vectorize user skills
    user_vector = mlb.transform([user_skills])
    user_vector_df = pd.DataFrame(user_vector, columns=skill_cols)
    
    # 3. Get target role centroid
    target_centroid = centroids.loc[[target_job_title]]
    
    # 4. Calculate similarity (for informational purposes)
    similarity = cosine_similarity(user_vector, target_centroid)[0][0]
    
    # 5. Identify missing skills
    # Skills where user has 0 and centroid has high value
    user_has = user_vector[0]
    job_needs = target_centroid.values[0]
    
    # Score missing skills by how prevalent they are in the job role
    missing_scores = []
    for i, skill in enumerate(skill_cols):
        if user_has[i] == 0:
            missing_scores.append((skill, job_needs[i]))
    
    # Sort by score descending
    missing_scores.sort(key=lambda x: x[1], reverse=True)
    
    return similarity, missing_scores[:10]

if __name__ == "__main__":
    # Example usage
    user_skills = ["Python", "SQL"]
    target_role = "Data Scientist"
    
    print(f"Analyzing skill gap for User: {user_skills} -> Target Role: {target_role}")
    sim, gaps = skill_gap_analysis(user_skills, target_role)
    
    if isinstance(sim, str):
        print(sim)
    else:
        print(f"Cosine Similarity: {sim:.4f}")
        print("\nTop Missing Skills:")
        for skill, score in gaps:
            print(f"- {skill} (Prevalence: {score:.2%})")
