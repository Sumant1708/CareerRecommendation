import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_model():
    return joblib.load("career_recommender_model.joblib")

pipeline = load_model()

# ---------------- PREDICTION FUNCTION ---------------- #
def predict_top_3_careers(age, education, skills_list, interests_list):
    combined_text = ";".join(skills_list) + ";" + ";".join(interests_list)

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
st.set_page_config(page_title="AI Career Recommender", page_icon="🎯")

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
    ["High School", "Diploma", "Undergraduate", "Postgraduate"]
)

skills = st.text_input("Skills (comma separated)", placeholder="e.g. Python, Machine Learning, Communication")
interests = st.text_input("Interests (comma separated)", placeholder="e.g. AI, Business, Design")

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
        "Data Scientist": "Works with data, machine learning and analytics.",
        "Software Engineer": "Develops applications, systems and software.",
        "Web Developer": "Builds and maintains websites.",
        "Designer": "Focuses on UI/UX and creative design."
    }

    # Display results
    for career, prob in st.session_state.results:
        st.markdown(f"### {career}")
        st.progress(int(prob * 100))
        st.write(f"Match Score: **{round(prob * 100, 2)}%**")

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