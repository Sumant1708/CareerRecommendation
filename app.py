import streamlit as st
import joblib
import numpy as np
import pandas as pd

import streamlit as st
import os

MODEL_PATH = "career_recommender_model.joblib"

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_model():
    return joblib.load("career_recommender_model.joblib")

pipeline = load_model()

# ---------------- PREDICTION FUNCTION ---------------- #
def predict_top_3_careers(age, education, skills_list, interests_list):
    combined_text = ";".join(skills_list + interests_list)

    user_df = pd.DataFrame([{
        "Age": age,
        "Education": education,
        "Combined_Text": combined_text
    }])

    proba = pipeline.predict_proba(user_df)[0]
    classes = pipeline.classes_

    top3_idx = np.argsort(proba)[-3:][::-1]

    return [(classes[i], float(proba[i])) for i in top3_idx]


# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="AI Career Recommender",)
import time

# ---------------- WELCOME LOADING SCREEN ---------------- #
loading_placeholder = st.empty()

with loading_placeholder.container():

    st.markdown(
        """
        <div style='text-align: center; padding-top: 120px;'>
            <h1 style='font-size: 50px;'>🎯</h1>
            <h1>Welcome to the Career Recommendation System</h1>
            <p style='font-size:18px; color:gray;'>
                Loading AI Model...
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.spinner("Initializing System..."):
        time.sleep(2)

loading_placeholder.empty()

# ---------------- SESSION STATE ---------------- #
if "results" not in st.session_state:
    st.session_state.results = None

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("⚙️ About")
st.sidebar.write("AI Career Recommendation System")

st.sidebar.markdown("### Instructions")
st.sidebar.write("""
1. Enter your details  
2. Click recommend  
3. View top careers  
""")

# ---------------- HEADER ---------------- #
st.title("🎯 AI Career Recommender")
st.markdown("Get personalized career suggestions based on your profile")

st.divider()

# ---------------- INPUT ---------------- #
st.subheader("📋 Enter Your Details")

age = st.number_input("Age", min_value=15, max_value=60, value=20)

education = st.selectbox(
    "Education Level",
    [
        "B.Tech CSE",
        "B.Tech IT",
        "BCA",
        "MCA",
        "B.Sc Computer Science"
    ]
)

skills = st.text_input(
    "Skills (comma separated)",
    placeholder="e.g. Python, Machine Learning, SQL"
)
interests = st.text_input(
    "Interests (comma separated)",
    placeholder="e.g. Artificial Intelligence, Technology"
)

st.divider()

# ---------------- BUTTON ---------------- #
if st.button("🚀 Get Career Recommendations"):

    if not skills or not interests:
        st.warning("⚠️ Please enter both skills and interests")
    else:
        skills_list = [s.strip() for s in skills.split(",")]
        interests_list = [i.strip() for i in interests.split(",")]

        with st.spinner("Analyzing your profile..."):
            st.session_state.results = predict_top_3_careers(
                age, education, skills_list, interests_list
            )

# ---------------- RESULTS ---------------- #
if st.session_state.results:

    st.success("✅ Here are your top career matches!")
    st.subheader("🏆 Top 3 Career Recommendations")

    # Career descriptions (customize more if you want)
    career_info = {
    "Data Scientist": "Works with data, analytics, statistics and machine learning.",
    
    "Web Developer": "Builds websites and web applications using frontend/backend technologies.",
    
    "Network Engineer": "Manages computer networks, servers and connectivity infrastructure.",
    
    "UI/UX Designer": "Designs user interfaces and improves user experience for applications.",
    
    "Cybersecurity Analyst": "Protects systems and networks from cyber threats and attacks.",
    
    "AI Engineer": "Builds intelligent systems using AI and deep learning technologies.",
    
    "Software Developer": "Develops software applications, systems and business solutions.",
    
    "Cloud Engineer": "Works with cloud platforms, deployment and infrastructure systems.",
    
    "DevOps Engineer": "Automates deployment pipelines and manages software operations."
}

    # Display results
    for career, prob in st.session_state.results:
        st.markdown(f"### {career}")
        st.progress(float(prob))
        st.write(f"Match Score: **{round(prob * 100, 2)}%**")

        if prob > 0.80:
            st.success("High Match")
        elif prob > 0.60:
            st.info("Moderate Match")
        else:
            st.warning("Low Match")
        if career in career_info:
            st.write(career_info[career])

        st.write("---")

    # ---------------- CHART ---------------- #
    df = pd.DataFrame(st.session_state.results, columns=["Career", "Score"])
    st.subheader("📊 Career Match Visualization")
    st.bar_chart(df.set_index("Career"))

# ---------------- FOOTER ---------------- #
st.markdown("""
---
👨‍💻 Developed as a Machine Learning Project  
""")