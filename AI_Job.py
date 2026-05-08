import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import root_mean_squared_error, r2_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import PyPDF2
import docx
import re

# Page Config
st.set_page_config(
    page_title="AI Job Market Intelligence",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        color: #e0e0e0;
    }
    .css-1d391kg {
        background-color: rgba(25, 25, 25, 0.7);
        backdrop-filter: blur(10px);
    }
    .stButton>button {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    h1, h2, h3 {
        color: #00d4ff !important;
    }
    /* Style the radio buttons in the sidebar */
    div[data-testid="stSidebarNav"] {
        background-color: transparent !important;
    }
    .st-emotion-cache-17l69k {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 10px !important;
        margin-bottom: 5px !important;
        padding: 10px !important;
        transition: all 0.3s ease !important;
    }
    .st-emotion-cache-17l69k:hover {
        background: rgba(0, 212, 255, 0.1) !important;
        transform: translateX(5px);
    }
    .st-emotion-cache-6qob1r {
        color: #00d4ff !important;
        font-weight: bold !important;
    }
    .hero-section {
        background: linear-gradient(135deg, rgba(75, 108, 183, 0.2) 0%, rgba(24, 40, 72, 0.4) 100%);
        padding: 40px;
        border-radius: 20px;
        border: 1px solid rgba(0, 212, 255, 0.2);
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
    }
    .hero-title {
        background: linear-gradient(90deg, #00d4ff, #00ff88);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        margin-bottom: 10px;
        letter-spacing: -1px;
    }
    .hero-subtitle {
        color: #e0e0e0;
        font-size: 1.2rem;
        opacity: 0.8;
    }
    </style>
    """, unsafe_allow_html=True)

# --- DATA PROCESSING ---

@st.cache_data
def load_and_preprocess_data():
    df1 = pd.read_csv('ai_job_dataset.csv')
    df2 = pd.read_csv('ai_job_dataset1.csv')
    
    if 'salary_local' in df2.columns:
        df2 = df2.drop(columns=['salary_local'])
    
    df = pd.concat([df1, df2], ignore_index=True)
    
    # Advanced Preprocessing
    df['posting_date'] = pd.to_datetime(df['posting_date'])
    df['month_year'] = df['posting_date'].dt.to_period('M').astype(str)
    
    # Clean skills
    df['skills_list'] = df['required_skills'].apply(lambda x: tuple(s.strip() for s in str(x).split(',')) if pd.notnull(x) else ())
    
    # Filter out any zero or negative salaries
    df = df[df['salary_usd'] > 10000]
    
    return df

@st.cache_resource
def train_models(df):
    # 1. Feature Engineering
    mlb = MultiLabelBinarizer()
    skills_encoded = mlb.fit_transform(df['skills_list'])
    skills_df = pd.DataFrame(skills_encoded, columns=mlb.classes_)
    
    exp_map = {'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3}
    df['experience_encoded'] = df['experience_level'].map(exp_map).fillna(-1)
    
    cat_cols = ['industry', 'company_size', 'education_required']
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_encoded = ohe.fit_transform(df[cat_cols])
    cat_df = pd.DataFrame(cat_encoded, columns=ohe.get_feature_names_out(cat_cols))
    
    X = pd.concat([df[['experience_encoded', 'remote_ratio']].reset_index(drop=True), skills_df, cat_df], axis=1)
    y_reg = df['salary_usd']
    
    # 2. Regression Model (XGBoost)
    X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    reg_model = XGBRegressor(n_estimators=100, random_state=42)
    reg_model.fit(X_train, y_train)
    
    # 3. Classification Model (XGBoost)
    top_20_titles = df['job_title'].value_counts().head(20).index
    df_cls = df[df['job_title'].isin(top_20_titles)].copy()
    
    # Match indices for X
    X_cls = X.iloc[df_cls.index]
    le = LabelEncoder()
    y_cls = le.fit_transform(df_cls['job_title'])
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
    cls_model = XGBClassifier(n_estimators=100, random_state=42)
    cls_model.fit(X_train_c, y_train_c)
    
    return reg_model, cls_model, mlb, ohe, le, X.columns.tolist()

def extract_text_from_file(uploaded_file):
    text = ""
    if uploaded_file.type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text()
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text

# --- APP UI ---

def main():
    st.markdown("""
        <div class="hero-section">
            <h1 class="hero-title">🗺️ AI Job Market Intelligence</h1>
            <p class="hero-subtitle">Advanced Analytics • Salary Prediction • CV Intelligence</p>
        </div>
    """, unsafe_allow_html=True)

    # Load Data and Models
    df = load_and_preprocess_data()
    reg_model, cls_model, mlb, ohe, le, feature_names = train_models(df)
    df_exploded = df.explode('skills_list')

    # Sidebar Navigation
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio(
        "Explore Sections",
        ["📊 Market Overview", "💰 Salary Predictor", "🎯 Job Classifier", "🧠 Skill Gap Analyzer", "📄 CV Analyzer", "📈 Insights"],
        index=0,
        help="Select a section to dive into the data"
    )

    # Experience Level Mapping
    EXP_LEVEL_NAMES = {
        "EN": "Junior / Entry-level",
        "MI": "Intermediate / Mid-level",
        "SE": "Expert / Senior-level",
        "EX": "Director / Executive"
    }
    # Inverse mapping for logic
    INV_EXP_MAP = {v: k for k, v in EXP_LEVEL_NAMES.items()}

    # Company Size Mapping
    COMPANY_SIZE_NAMES = {
        "S": "Small (Startup / <50)",
        "M": "Medium (Mid-sized / 50-250)",
        "L": "Large (Enterprise / 250+)"
    }
    INV_SIZE_MAP = {v: k for k, v in COMPANY_SIZE_NAMES.items()}

    if choice == "📊 Market Overview":
        st.subheader("🌐 Global Industry & Market Intelligence")
        
        # Row 1: Skills and Exp
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔥 Top 10 Most In-Demand Skills")
            skill_counts = df_exploded['skills_list'].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=skill_counts.values, y=skill_counts.index, palette="viridis", ax=ax)
            ax.set_facecolor('#0e1117')
            fig.patch.set_facecolor('#0e1117')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('#00d4ff')
            ax.yaxis.label.set_color('#00d4ff')
            # Add labels
            for i, v in enumerate(skill_counts.values):
                ax.text(v + 5, i, str(v), color='white', va='center', fontweight='bold')
            sns.despine(left=True, bottom=True)
            st.pyplot(fig)
            
        with col2:
            st.markdown("#### 💰 Salary by Experience Level")
            df_plot = df.copy()
            df_plot['Experience'] = df_plot['experience_level'].map(EXP_LEVEL_NAMES)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df_plot, x='Experience', y='salary_usd', palette="magma", 
                        order=[EXP_LEVEL_NAMES[o] for o in ['EN', 'MI', 'SE', 'EX'] if o in df['experience_level'].unique()], ax=ax)
            ax.set_facecolor('#0e1117')
            fig.patch.set_facecolor('#0e1117')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('#00d4ff')
            ax.yaxis.label.set_color('#00d4ff')
            plt.xticks(rotation=15)
            sns.despine()
            st.pyplot(fig)

        # Row 2: Industries and Market Velocity
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("#### 🏆 Top 10 Paying Industries")
            top_paying_ind = df.groupby('industry')['salary_usd'].median().sort_values(ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=top_paying_ind.values, y=top_paying_ind.index, palette="flare", ax=ax)
            ax.set_facecolor('#0e1117')
            fig.patch.set_facecolor('#0e1117')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('#00d4ff')
            ax.yaxis.label.set_color('#00d4ff')
            # Add currency labels
            for i, v in enumerate(top_paying_ind.values):
                ax.text(v + 1000, i, f"${v/1000:.0f}k", color='white', va='center', fontweight='bold')
            sns.despine(left=True, bottom=True)
            st.pyplot(fig)
            
        with col4:
            st.markdown("#### ⚡ Market Velocity (Job Trends)")
            velocity = df.groupby('month_year').size()
            fig, ax = plt.subplots(figsize=(10, 6))
            velocity.plot(kind='line', marker='o', color='#00d4ff', linewidth=3, ax=ax)
            ax.set_facecolor('#0e1117')
            fig.patch.set_facecolor('#0e1117')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('#00d4ff')
            ax.yaxis.label.set_color('#00d4ff')
            ax.grid(alpha=0.1, color='white')
            sns.despine()
            st.pyplot(fig)

        st.markdown("#### 🌍 Global Talent Distribution (Top Hiring Locations)")
        location_counts = df['company_location'].value_counts().head(15)
        st.bar_chart(location_counts, color="#4b6cb7")

    elif choice == "💰 Salary Predictor":
        st.subheader("Predict Your AI Market Value")
        
        with st.form("salary_form"):
            col1, col2 = st.columns(2)
            with col1:
                exp_choice = st.selectbox("Your Experience Level", list(EXP_LEVEL_NAMES.values()))
                exp = INV_EXP_MAP[exp_choice]
                industry = st.selectbox("Industry", df['industry'].unique())
                remote = st.slider("Remote Ratio (%)", 0, 100, 50)
            with col2:
                size_choice = st.selectbox("Company Size", list(COMPANY_SIZE_NAMES.values()))
                comp_size = INV_SIZE_MAP[size_choice]
                edu = st.selectbox("Education Required", df['education_required'].unique())
                user_skills = st.multiselect("Select Your Skills", mlb.classes_)
            
            submit = st.form_submit_button("Calculate Estimated Salary")
            
            if submit:
                # Prepare input vector
                exp_map = {'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3}
                input_data = {
                    'experience_encoded': exp_map[exp],
                    'remote_ratio': remote,
                    'industry': industry,
                    'company_size': comp_size,
                    'education_required': edu
                }
                
                # Categorical
                cat_input = pd.DataFrame([input_data])[['industry', 'company_size', 'education_required']]
                cat_encoded = ohe.transform(cat_input)
                cat_df = pd.DataFrame(cat_encoded, columns=ohe.get_feature_names_out(['industry', 'company_size', 'education_required']))
                
                # Skills
                skills_vec = mlb.transform([user_skills])
                skills_df = pd.DataFrame(skills_vec, columns=mlb.classes_)
                
                # Combine
                final_input = pd.concat([pd.DataFrame([{'experience_encoded': exp_map[exp], 'remote_ratio': remote}]), skills_df, cat_df], axis=1)
                final_input = final_input[feature_names] # Ensure order
                
                prediction = reg_model.predict(final_input)[0]
                
                st.balloons()
                st.markdown(f"""
                <div style="background: rgba(0, 212, 255, 0.1); padding: 30px; border-radius: 20px; border: 2px solid #00d4ff; text-align: center;">
                    <h1 style="margin: 0; color: #00d4ff;">${prediction:,.2f}</h1>
                    <p style="font-size: 1.2rem; color: #e0e0e0;">Estimated Annual Salary (USD)</p>
                </div>
                """, unsafe_allow_html=True)

                # --- ADVANCED: Salary Impact Analysis (What-If) ---
                st.markdown("---")
                st.subheader("🚀 Salary Boost: 'What-If' Analysis")
                st.write("How much more could you earn if you added these skills?")
                
                # Find skills the user DOESN'T have
                all_skills = set(mlb.classes_)
                user_skills_set = set(user_skills)
                missing_skills = list(all_skills - user_skills_set)
                
                # Sample a subset of missing skills to test (to save time)
                # We'll pick skills that are generally high-paying or frequent
                test_skills = [s for s in missing_skills if s in df_exploded['skills_list'].value_counts().head(50).index]
                
                impacts = []
                for skill in test_skills[:10]: # Test top 10 most frequent missing skills
                    temp_input = final_input.copy()
                    if skill in temp_input.columns:
                        temp_input[skill] = 1
                        new_pred = reg_model.predict(temp_input)[0]
                        gain = new_pred - prediction
                        if gain > 0:
                            impacts.append((skill, gain))
                
                impacts.sort(key=lambda x: x[1], reverse=True)
                
                if impacts:
                    cols = st.columns(3)
                    for i, (skill, gain) in enumerate(impacts[:3]):
                        with cols[i]:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3 style="color: #00ff88; margin-bottom: 5px;">+${gain:,.0f}</h3>
                                <p style="font-size: 0.9rem;">Potential increase if you learn <b>{skill}</b></p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("You already have a very high-value skill set for this role!")

                # --- NEW: Industry Skill Benchmark ---
                st.markdown("---")
                st.markdown(f"#### 🏦 Industry Benchmark: {industry}")
                st.write("How your selected skills compare to the top requirements in this sector.")
                
                ind_data = df[df['industry'] == industry]
                ind_skills_freq = ind_data.explode('skills_list')['skills_list'].value_counts(normalize=True).head(15)
                
                plot_df = pd.DataFrame({
                    'Skill': ind_skills_freq.index,
                    'Industry Demand': ind_skills_freq.values,
                    'Status': ['Achieved' if s in user_skills_set else 'Required' for s in ind_skills_freq.index]
                })
                
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = {'Achieved': '#00ff88', 'Required': '#444444'}
                sns.barplot(data=plot_df, x='Industry Demand', y='Skill', hue='Status', palette=colors, ax=ax, dodge=False)
                
                ax.set_facecolor('#0e1117')
                fig.patch.set_facecolor('#0e1117')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('#00d4ff')
                ax.yaxis.label.set_color('#00d4ff')
                ax.legend(facecolor='#0e1117', edgecolor='#00d4ff', labelcolor='white')
                sns.despine(left=True, bottom=True)
                st.pyplot(fig)

                # --- NEW: Tabular Breakdown ---
                st.markdown("#### 📋 Detailed Industry Benchmark")
                table_df = plot_df.copy()
                table_df.columns = ['Skill', 'Industry Demand (%)', 'Your Status']
                table_df['Industry Demand (%)'] = table_df['Industry Demand (%)'].apply(lambda x: f"{x:.1%}")
                table_df['Your Status'] = table_df['Your Status'].apply(lambda x: "✅ Achieved" if "Achieved" in x else "❌ Missing")
                
                st.dataframe(table_df, use_container_width=True, hide_index=True)

    elif choice == "🎯 Job Classifier":
        st.subheader("Which Job Matches Your Skills?")
        user_skills = st.multiselect("Select Your Skills", mlb.classes_, key="cls_skills")
        
        if st.button("Predict Best Fit"):
            if not user_skills:
                st.warning("Please select at least one skill.")
            else:
                # We need full feature vector, using defaults for others
                skills_vec = mlb.transform([user_skills])
                skills_df = pd.DataFrame(skills_vec, columns=mlb.classes_)
                
                # Create dummy for others to match feature names
                dummy_input = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
                for col in skills_df.columns:
                    dummy_input[col] = skills_df[col].values
                
                pred_idx = cls_model.predict(dummy_input)[0]
                pred_title = le.inverse_transform([pred_idx])[0]
                
                # Get probabilities
                probs = cls_model.predict_proba(dummy_input)[0]
                top_indices = probs.argsort()[-3:][::-1]
                
                st.markdown(f"### You are most likely a: **{pred_title}**")
                st.markdown("#### Top 3 Recommendations:")
                for idx in top_indices:
                    st.write(f"- {le.inverse_transform([idx])[0]} ({probs[idx]:.1%})")

    elif choice == "🧠 Skill Gap Analyzer":
        st.subheader("Bridge the Gap to Your Dream Role")
        
        col1, col2 = st.columns(2)
        with col1:
            target_role = st.selectbox("Target Job Title", df['job_title'].unique())
        with col2:
            current_skills = st.multiselect("Your Current Skills", mlb.classes_, key="gap_skills")
            
        if st.button("Analyze Gaps"):
            # Role analysis
            skill_cols = mlb.classes_
            target_data = df[df['job_title'] == target_role]
            skill_freq = target_data.explode('skills_list')['skills_list'].value_counts(normalize=True).head(15)
            
            user_has = set(current_skills)
            
            # Metrics
            match_count = sum(1 for skill in skill_freq.index if skill in user_has)
            match_pct = match_count / len(skill_freq) if len(skill_freq) > 0 else 0
            
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Skill Coverage", f"{match_pct:.1%}", help="Percentage of top 15 required skills you possess")
            with m2:
                st.metric("Skills to Learn", len(skill_freq) - match_count)

            st.markdown("---")
            st.markdown(f"#### 📊 Skill Comparison: Achieved vs. Required for {target_role}")
            
            # Visualization Data
            plot_df = pd.DataFrame({
                'Skill': skill_freq.index,
                'Market Demand': skill_freq.values,
                'Status': ['Achieved' if s in user_has else 'Required' for s in skill_freq.index]
            })
            
            fig, ax = plt.subplots(figsize=(10, 8))
            colors = {'Achieved': '#00ff88', 'Required': '#444444'}
            sns.barplot(data=plot_df, x='Market Demand', y='Skill', hue='Status', palette=colors, ax=ax, dodge=False)
            
            ax.set_facecolor('#0e1117')
            fig.patch.set_facecolor('#0e1117')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('#00d4ff')
            ax.yaxis.label.set_color('#00d4ff')
            ax.set_title(f"Top 15 Skills for {target_role}", color='white', pad=20)
            
            # Legend styling
            leg = ax.legend()
            plt.setp(leg.get_texts(), color='white')
            
            st.pyplot(fig)
            
            # --- NEW: Tabular Breakdown ---
            st.markdown("#### 📋 Detailed Gap Analysis")
            table_df = plot_df.copy()
            table_df['Market Demand'] = table_df['Market Demand'].apply(lambda x: f"{x:.1%}")
            table_df['Status'] = table_df['Status'].apply(lambda x: "✅ Achieved" if "Achieved" in x else "❌ Missing")
            
            st.dataframe(table_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.markdown("#### 🚀 Action Plan")
            gaps = [s for s in skill_freq.index if s not in user_has]
            if not gaps:
                st.success("🎉 You have a perfect match for the top skills in this role!")
            else:
                st.write(f"Focus on learning **{gaps[0]}** first—it's the most significant gap in your profile.")

    elif choice == "📄 CV Analyzer":
        st.subheader("🕵️ Advanced CV Skill Intelligence")
        st.markdown("""
        Upload your CV to see your market compatibility and get a personalized 
        roadmap to bridge the gap for your target role.
        """)
        
        col_cv1, col_cv2 = st.columns([1, 2], gap="large")
        
        with col_cv1:
            uploaded_file = st.file_uploader("Upload CV (PDF or DOCX)", type=["pdf", "docx"])
            target_role_cv = st.selectbox("Your Target Role", df['job_title'].unique(), key="cv_target")
            
        if uploaded_file is not None:
            with st.spinner("🧠 Deep Scanning CV..."):
                cv_text = extract_text_from_file(uploaded_file)
                
                # Match skills from dataset
                detected_skills = []
                for skill in mlb.classes_:
                    pattern = rf'\b{re.escape(skill)}\b'
                    if re.search(pattern, cv_text, re.IGNORECASE):
                        detected_skills.append(skill)
                
                # Analysis Logic
                target_data = df[df['job_title'] == target_role_cv]
                role_skills_freq = target_data.explode('skills_list')['skills_list'].value_counts(normalize=True)
                
                user_has = set(detected_skills)
                critical_gaps = []
                competitive_gaps = []
                
                for skill, freq in role_skills_freq.items():
                    if skill not in user_has:
                        if freq > 0.5:
                            critical_gaps.append((skill, freq))
                        elif freq > 0.2:
                            competitive_gaps.append((skill, freq))

                # Profile Compatibility
                sim = 0.0
                if detected_skills:
                    user_vec = mlb.transform([detected_skills])
                    role_skills_matrix = mlb.transform(target_data['skills_list'])
                    role_centroid = role_skills_matrix.mean(axis=0).reshape(1, -1)
                    sim = cosine_similarity(user_vec, role_centroid)[0][0]

                with col_cv2:
                    st.markdown(f"### Analysis for **{target_role_cv}**")
                    
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric("Profile Match Score", f"{sim:.1%}")
                    with m2:
                        st.metric("Skills Detected", len(detected_skills))
                    
                    if sim > 0.6:
                        st.success("✅ **Strong Candidate:** Your profile is highly aligned with market expectations for this role.")
                    else:
                        st.warning("⚠️ **Gap Detected:** To become a top-tier candidate, focus on the priority skills below.")

                st.markdown("---")
                
                # --- NEW: Skill Comparison Chart ---
                st.markdown("#### 📊 CV Skill Match Comparison")
                plot_df = pd.DataFrame({
                    'Skill': role_skills_freq.index[:15],
                    'Market Demand': role_skills_freq.values[:15],
                    'Status': ['Achieved' if s in user_has else 'Required' for s in role_skills_freq.index[:15]]
                })
                
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = {'Achieved': '#00ff88', 'Required': '#444444'}
                sns.barplot(data=plot_df, x='Market Demand', y='Skill', hue='Status', palette=colors, ax=ax, dodge=False)
                
                ax.set_facecolor('#0e1117')
                fig.patch.set_facecolor('#0e1117')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('#00d4ff')
                ax.yaxis.label.set_color('#00d4ff')
                ax.legend(facecolor='#0e1117', edgecolor='#00d4ff', labelcolor='white')
                sns.despine(left=True, bottom=True)
                st.pyplot(fig)
                
                # --- NEW: Tabular Breakdown ---
                st.markdown("#### 📋 Detailed Skill Match Breakdown")
                table_df = plot_df.copy()
                table_df['Market Demand'] = table_df['Market Demand'].apply(lambda x: f"{x:.1%}")
                table_df['Status'] = table_df['Status'].apply(lambda x: "✅ Achieved" if "Achieved" in x else "❌ Missing")
                
                st.dataframe(table_df, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.markdown("#### 🎯 Priority Learning Roadmap")
                    if not critical_gaps and not competitive_gaps:
                        st.balloons()
                        st.success("You have mastered all the core technical requirements!")
                    else:
                        if critical_gaps:
                            st.error(f"**Critical Gaps (Missing in >50% of Job Postings)**")
                            for skill, freq in critical_gaps[:5]:
                                st.write(f"🔴 **{skill}** (Required in {freq:.0%})")
                        
                        if competitive_gaps:
                            st.info(f"**Competitive Edge (Missing in >20% of Postings)**")
                            for skill, freq in competitive_gaps[:5]:
                                st.write(f"🟡 **{skill}** (Found in {freq:.0%})")

                with res_col2:
                    st.markdown("#### 🛠️ Recommended Portfolio Project")
                    if critical_gaps:
                        top_skill = critical_gaps[0][0]
                        sec_skill = critical_gaps[1][0] if len(critical_gaps) > 1 else "Cloud Deployment"
                        
                        st.markdown(f"""
                        <div style="background: rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 15px; border-left: 5px solid #00d4ff;">
                            <p style="color: #00d4ff; font-weight: bold;">Project Idea: "The {top_skill} Architect"</p>
                            <p style="font-size: 0.9rem;">Build a project integrating <b>{top_skill}</b> with <b>{sec_skill}</b>. 
                            This combination is highly sought after for {target_role_cv} roles.</p>
                            <hr style="opacity: 0.2;">
                            <p style="font-size: 0.8rem; color: #aaa;"><b>Benefit:</b> Proves technical competency where your CV is currently weakest.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("#### 🔍 ATS Keyword Optimization")
                    # Suggest keywords based on competitive gaps
                    keywords = [g[0] for g in critical_gaps[:3]] + [g[0] for g in competitive_gaps[:3]]
                    if keywords:
                        st.write("Add these keywords to your CV summary/skills section to improve search visibility:")
                        st.write(", ".join([f"`{k}`" for k in keywords]))

    elif choice == "📈 Insights":
        st.subheader("What Drives AI Salaries?")
        
        importances = reg_model.feature_importances_
        imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False)
        
        # Clean up feature names for display
        def clean_feature_name(name):
            if name.startswith('industry_'):
                return f"Industry: {name.replace('industry_', '')}"
            if name.startswith('company_size_'):
                # Map back if possible
                code = name.replace('company_size_', '')
                return f"Size: {COMPANY_SIZE_NAMES.get(code, code)}"
            if name.startswith('education_required_'):
                return f"Edu: {name.replace('education_required_', '')}"
            if name == 'experience_encoded':
                return "Years of Experience / Level"
            if name == 'remote_ratio':
                return "Remote Work Flexibility"
            return name

        imp_df['Display Feature'] = imp_df['Feature'].apply(clean_feature_name)
        
        st.markdown("#### Top 15 Salary Drivers (Market Impact)")
        
        # Select top 15 for a cleaner plot
        top_imp = imp_df.head(15)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        # Use a more modern palette
        sns.barplot(data=top_imp, x='Importance', y='Display Feature', palette="coolwarm", ax=ax)
        
        # Style adjustments
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')
        ax.tick_params(colors='white', labelsize=10)
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.set_xlabel("Relative Influence on Salary", color='#00d4ff', fontsize=12)
        ax.set_ylabel("")
        plt.title("What Most Influences Your Pay?", color='#00d4ff', fontsize=14, pad=20)
        
        st.pyplot(fig)
        
        # Dynamic insight description
        top_feature = top_imp.iloc[0]['Display Feature']
        st.markdown(f"""
        <div style="background: rgba(0, 212, 255, 0.05); padding: 20px; border-radius: 10px; border-left: 5px solid #00d4ff;">
            <h4>🔍 Key Takeaway</h4>
            <p>The chart above reveals that <b>{top_feature}</b> is the single most influential factor in determining AI salaries in this market.</p>
            <ul>
                <li><b>Skills vs Industry:</b> While your domain (Industry) matters, technical mastery of specific AI tools often carries more weight.</li>
                <li><b>The Experience Premium:</b> Seniority remains a massive driver, showing a clear career progression in pay.</li>
                <li><b>Flexibility:</b> Remote ratio can sometimes impact base pay, but it's often secondary to technical expertise.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
