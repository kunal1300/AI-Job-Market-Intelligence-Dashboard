import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set aesthetic style
sns.set_theme(style="whitegrid", palette="viridis")

def load_and_combine_data():
    df1 = pd.read_csv('ai_job_dataset.csv')
    df2 = pd.read_csv('ai_job_dataset1.csv')
    
    # Check columns
    print(f"Dataset 1 columns: {df1.columns.tolist()}")
    print(f"Dataset 2 columns: {df2.columns.tolist()}")
    
    # Drop salary_local if present to merge
    if 'salary_local' in df2.columns:
        df2 = df2.drop(columns=['salary_local'])
    
    df = pd.concat([df1, df2], ignore_index=True)
    print(f"Combined dataset shape: {df.shape}")
    return df

def explode_skills(df):
    # Parse required_skills string into list
    df['skills_list'] = df['required_skills'].apply(lambda x: [s.strip() for s in str(x).split(',')] if pd.notnull(x) else [])
    # Explode into individual rows
    df_exploded = df.explode('skills_list')
    return df_exploded

def plot_top_skills(df_exploded, top_n=20):
    plt.figure(figsize=(12, 8))
    skill_counts = df_exploded['skills_list'].value_counts().head(top_n)
    sns.barplot(x=skill_counts.values, y=skill_counts.index)
    plt.title(f'Top {top_n} Most In-Demand Skills Overall')
    plt.xlabel('Frequency')
    plt.ylabel('Skill')
    plt.tight_layout()
    plt.savefig('top_skills_overall.png')
    print("Saved top_skills_overall.png")

def plot_top_skills_per_title(df_exploded, top_n=5):
    # Get top 10 job titles
    top_titles = df_exploded['job_title'].value_counts().head(10).index
    
    plt.figure(figsize=(15, 12))
    for i, title in enumerate(top_titles):
        plt.subplot(5, 2, i+1)
        skill_counts = df_exploded[df_exploded['job_title'] == title]['skills_list'].value_counts().head(top_n)
        sns.barplot(x=skill_counts.values, y=skill_counts.index)
        plt.title(f'Top Skills for {title}')
        plt.xlabel('')
        plt.ylabel('')
    
    plt.tight_layout()
    plt.savefig('top_skills_per_title.png')
    print("Saved top_skills_per_title.png")

def plot_salary_distributions(df):
    plt.figure(figsize=(14, 6))
    
    # Salary by Experience Level
    plt.subplot(1, 2, 1)
    # Order for experience level
    exp_order = ['EN', 'MI', 'SE', 'EX']
    sns.boxplot(data=df, x='experience_level', y='salary_usd', order=[o for o in exp_order if o in df['experience_level'].unique()])
    plt.title('Salary Distribution by Experience Level')
    plt.xticks(rotation=45)
    
    # Salary by Industry (Top 10)
    plt.subplot(1, 2, 2)
    top_industries = df['industry'].value_counts().head(10).index
    sns.boxplot(data=df[df['industry'].isin(top_industries)], x='industry', y='salary_usd')
    plt.title('Salary Distribution by Top 10 Industries')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('salary_distributions.png')
    print("Saved salary_distributions.png")

if __name__ == "__main__":
    df = load_and_combine_data()
    df_exploded = explode_skills(df)
    
    plot_top_skills(df_exploded)
    plot_top_skills_per_title(df_exploded)
    plot_salary_distributions(df)
    
    # Save combined data for next steps
    df.to_csv('combined_ai_jobs.csv', index=False)
    print("Saved combined_ai_jobs.csv")
