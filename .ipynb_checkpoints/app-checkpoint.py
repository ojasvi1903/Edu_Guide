import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load trained model
model = joblib.load("edu_model.pkl")

# Sample study materials
study_materials = [
    "Basic concepts of Data Structures and Algorithms",
    "Intermediate Python practice problems and quizzes",
    "Advanced Machine Learning techniques and research papers",
    "Beginner guide to Time Management and Focus",
    "High-level readings on Data Science Trends",
    "Intro to DBMS with examples",
    "Practice quiz on Operating Systems",
    "Research paper on Deep Learning in Education"
]

# Performance category to query mapping
queries = {
    0: "beginner topics and concepts",
    1: "practice quiz and tests",
    2: "advanced readings and research"
}

# Recommendation function
def recommend_materials(category, top_n=3):
    query = queries.get(category, "beginner topics")
    docs = [query] + study_materials
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(docs)
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    top_indices = cosine_sim.argsort()[-top_n:][::-1]
    return [study_materials[i] for i in top_indices]

# Streamlit UI setup
st.set_page_config(page_title="Smart Edu-Recommender", layout="wide")
st.title("📚 Smart Edu-Recommender System")
st.write("Upload student data, predict performance, and get personalized study material.")

# File uploader
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # 🐞 Show uploaded file's structure for debugging
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())
    st.write("🧾 Columns in uploaded file:", df.columns.tolist())


    # Encode categorical columns
    cat_cols = df.select_dtypes(include='object').columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # Load expected feature list from training
    expected_columns = joblib.load("model_features.pkl")

    # Add missing columns as 0
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match training data
    df = df[expected_columns]

    # Predict
    predictions = model.predict(df)
    df['Predicted Category'] = predictions
        st.subheader("🔍 Predictions")
    st.write("Predicted performance category for each student:")
    st.dataframe(df[['Predicted Category']])

    st.subheader("📌 Recommendations")
    st.write("Study material recommendations based on predicted categories:")

    for i, row in df.iterrows():
        st.markdown(f"**Student {i + 1}** - Category: {row['Predicted Category']}")
        recs = recommend_materials(row['Predicted Category'])
        for rec in recs:
            st.write(f"- {rec}")



        
    

