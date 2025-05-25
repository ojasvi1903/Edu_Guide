import streamlit as st

# ✅ Must be FIRST Streamlit command!
st.set_page_config(page_title="Edu-Guide", layout="centered")

import pandas as pd
import joblib
import os
import numpy as np
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

# ✅ Load the trained model
if os.path.exists("edu_model.pkl"):
    model = joblib.load("edu_model.pkl")
    # Extract feature names from the model to handle the mismatch
    try:
        expected_features = model.get_booster().feature_names
    except:
        # Fallback if get_booster() is not available
        expected_features = []
else:
    model = None
    expected_features = []

# ✅ Define performance categories
performance_categories = {
    0: {
        "label": "Needs Improvement",
        "description": "This category indicates you may benefit from strengthening your fundamentals and developing better study habits.",
        "color": "#FF4B4B",  # Red
        "recommendations": "Focus on basic concepts, structured study schedules, and seeking guidance from instructors."
    },
    1: {
        "label": "Average Performance",
        "description": "You're doing well in some areas but have room for growth in others. Practice and consistent effort will help you improve.",
        "color": "#FFC300",  # Yellow
        "recommendations": "Regular practice with quizzes, peer study groups, and applying concepts to real-world problems."
    },
    2: {
        "label": "High Performance",
        "description": "You demonstrate excellent academic performance! You're ready to explore advanced concepts and deeper learning.",
        "color": "#00C851",  # Green
        "recommendations": "Engage with advanced material, research opportunities, and mentoring other students."
    }
}

# ✅ Enhanced Study Material Links - Organized by category
study_materials = {
    0: {  # Needs Improvement
        "Basic concepts of Data Structures and Algorithms": "https://www.geeksforgeeks.org/data-structures/",
        "Intro to DBMS with examples": "https://www.tutorialspoint.com/dbms/index.htm",
        "Beginner guide to Time Management and Focus": "https://www.mindtools.com/pages/main/newMN_HTE.htm",
        "Fundamentals of Programming Languages": "https://www.freecodecamp.org/news/the-programming-language-pipeline-91d3f449c919/",
        "Academic Writing Basics": "https://owl.purdue.edu/owl/general_writing/academic_writing/index.html"
    },
    1: {  # Average Performance
        "Intermediate Python practice problems and quizzes": "https://www.w3schools.com/python/python_exercises.asp",
        "Practice quiz on Operating Systems": "https://www.geeksforgeeks.org/operating-systems-quizzes/",
        "Data Analysis with Python tutorials": "https://realpython.com/tutorials/data-science/",
        "Study Skills for College Success": "https://www.edx.org/learn/study-skills",
        "Project-based learning resources": "https://www.coursera.org/collections/project-based-learning"
    },
    2: {  # High Performance
        "Advanced Machine Learning techniques and research papers": "https://www.analyticsvidhya.com/blog/2021/06/advanced-machine-learning-techniques-every-data-scientist-should-know/",
        "High-level readings on Data Science Trends": "https://www.kdnuggets.com/2023/01/10-data-science-trends-2023.html",
        "Research paper on Deep Learning in Education": "https://arxiv.org/abs/1801.00001",
        "Competitive Programming Challenges": "https://leetcode.com/problemset/algorithms/",
        "Research Methodology and Academic Publishing": "https://www.coursera.org/learn/research-methods"
    }
}

def recommend_materials(category):
    """Return all 5 materials for the given performance category"""
    return list(study_materials[category].items())

def encode_inputs(df):
    """Encode categorical variables as numerical values"""
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col])
    return df_copy

def analyze_performance(input_data):
    """Analyze input data to determine performance category - BALANCED VERSION"""
    # Calculate average semester marks - Direct access to values
    sem_1 = input_data["SEM_1"].values[0]
    sem_2 = input_data["SEM_2"].values[0]
    sem_3 = input_data["SEM_3"].values[0]
    sem_4 = input_data["SEM_4"].values[0]
    sem_5 = input_data["SEM_5"].values[0]
    
    avg_marks = (sem_1 + sem_2 + sem_3 + sem_4 + sem_5) / 5
    
    # Get other metrics - Direct access to values
    weekly_study = input_data["Weekly_study_time__________________"].values[0]
    stress = input_data["Stress_Level"].values[0]
    sleep = input_data["average_sleep_duration"].values[0]
    
    # SIMPLIFIED AND BALANCED SCORING SYSTEM
    # Academic performance score (0-50 points)
    if avg_marks >= 85:
        academic_score = 50
    elif avg_marks >= 75:
        academic_score = 40
    elif avg_marks >= 65:
        academic_score = 30
    elif avg_marks >= 55:
        academic_score = 20
    elif avg_marks >= 45:
        academic_score = 10
    else:
        academic_score = 5
    
    # Study habits score (0-30 points)
    if weekly_study >= 25:
        study_score = 30
    elif weekly_study >= 20:
        study_score = 25
    elif weekly_study >= 15:
        study_score = 20
    elif weekly_study >= 10:
        study_score = 15
    elif weekly_study >= 5:
        study_score = 10
    else:
        study_score = 5
        
    # Sleep quality score (0-10 points)
    if 7 <= sleep <= 9:
        sleep_score = 10  # Optimal sleep
    elif 6 <= sleep < 7 or 9 < sleep <= 10:
        sleep_score = 8   # Good sleep
    elif 5 <= sleep < 6 or 10 < sleep <= 11:
        sleep_score = 6   # Fair sleep
    else:
        sleep_score = 3   # Poor sleep
        
    # Stress management score (0-10 points) - lower stress is better
    if stress <= 2:
        stress_score = 10
    elif stress == 3:
        stress_score = 7
    elif stress == 4:
        stress_score = 4
    else:
        stress_score = 1
        
    # Calculate total score (0-100)
    total_score = academic_score + study_score + sleep_score + stress_score
    
    # BALANCED THRESHOLDS - More realistic distribution
    if total_score >= 75:
        return 2  # High performance (top 25%)
    elif total_score >= 50:
        return 1  # Average performance (middle 50%)
    else:
        return 0  # Needs improvement (bottom 25%)

# ✅ Streamlit App UI
st.title("🎓 Edu-Guide")
st.markdown("Fill in your details below to get a prediction of your performance category and personalized study materials.")

with st.form("user_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧾 Student Information")
        Age = st.number_input("Age", min_value=15, max_value=35, value=20)
        Sex = st.selectbox("Sex", options=["Male", "Female"])
        Program = st.selectbox("Program", options=["BTech", "MTech", "BSc", "BA", "Other"])
        
    with col2:
        st.subheader("🧠 Wellness & Study Habits")
        Stress_Level = st.selectbox("Stress Level (1-5)", options=[1, 2, 3, 4, 5], 
                                  help="1 = Very Low, 5 = Very High")
        average_sleep_duration = st.slider("Average Sleep Duration (hours)", 4.0, 12.0, 7.0,
                                        help="Research suggests 7-9 hours is optimal for most students")
        Weekly_study_time__________________ = st.slider("Weekly Study Time (hours)", 0, 50, 10,
                                                    help="Time spent studying outside of class")

    st.subheader("🧪 Academic Records")
    col1, col2 = st.columns(2)
    
    with col1:
        SEM_1 = st.number_input("SEM 1 Marks", 0, 100, 70)
        SEM_2 = st.number_input("SEM 2 Marks", 0, 100, 75)
        SEM_3 = st.number_input("SEM 3 Marks", 0, 100, 78)
    
    with col2:
        SEM_4 = st.number_input("SEM 4 Marks", 0, 100, 72)
        SEM_5 = st.number_input("SEM 5 Marks", 0, 100, 80)

    submitted = st.form_submit_button("🔍 Predict Performance")

# ✅ Prediction + Recommendation
if submitted:
    # Create input dataframe with all collected features
    input_data = pd.DataFrame([{
        "Age": Age,
        "Sex": Sex,
        "Program": Program,
        "Stress_Level": Stress_Level,
        "average_sleep_duration": average_sleep_duration,
        "Weekly_study_time__________________": Weekly_study_time__________________,
        "SEM_1": SEM_1,
        "SEM_2": SEM_2,
        "SEM_3": SEM_3,
        "SEM_4": SEM_4,
        "SEM_5": SEM_5
    }])

    # Store original input before encoding for use in analysis
    original_input = input_data.copy()
    
    # Encode categorical inputs
    encoded_input = encode_inputs(input_data)
    
    # Try to predict with the model first
    model_prediction = None
    if model:
        try:
            # Create a full dataframe with all expected features and default values
            if expected_features:
                full_input_data = pd.DataFrame(columns=expected_features)
                for feature in expected_features:
                    if feature in encoded_input.columns:
                        full_input_data[feature] = encoded_input[feature]
                    else:
                        # Fill missing features with default values
                        full_input_data[feature] = 0
                
                # Make prediction with properly formatted data
                model_prediction = model.predict(full_input_data)[0]
            else:
                # If we couldn't get expected features, try using just the input data
                model_prediction = model.predict(encoded_input)[0]
                
        except Exception as e:
            st.warning(f"Model prediction failed: {str(e)}")
            st.info("Using backup analysis method instead.")
    
    # Fallback: use our analytical approach if model fails
    # prediction = model_prediction
# st.info("✅ Using trained model for prediction")
# ⬇️ Replaced by this:
        prediction = analyze_performance(original_input)
        st.info("🔧 Using analytical method for prediction")

        
        # Show debug information
        avg_marks = (SEM_1 + SEM_2 + SEM_3 + SEM_4 + SEM_5) / 5
        
        # Academic score calculation (0-50)
        if avg_marks >= 85:
            academic_score = 50
        elif avg_marks >= 75:
            academic_score = 40
        elif avg_marks >= 65:
            academic_score = 30
        elif avg_marks >= 55:
            academic_score = 20
        elif avg_marks >= 45:
            academic_score = 10
        else:
            academic_score = 5
        
        # Study score calculation (0-30)
        if Weekly_study_time__________________ >= 25:
            study_score = 30
        elif Weekly_study_time__________________ >= 20:
            study_score = 25
        elif Weekly_study_time__________________ >= 15:
            study_score = 20
        elif Weekly_study_time__________________ >= 10:
            study_score = 15
        elif Weekly_study_time__________________ >= 5:
            study_score = 10
        else:
            study_score = 5
            
        # Sleep score (0-10)
        if 7 <= average_sleep_duration <= 9:
            sleep_score = 10
        elif 6 <= average_sleep_duration < 7 or 9 < average_sleep_duration <= 10:
            sleep_score = 8
        elif 5 <= average_sleep_duration < 6 or 10 < average_sleep_duration <= 11:
            sleep_score = 6
        else:
            sleep_score = 3
            
        # Stress score (0-10)
        if Stress_Level <= 2:
            stress_score = 10
        elif Stress_Level == 3:
            stress_score = 7
        elif Stress_Level == 4:
            stress_score = 4
        else:
            stress_score = 1
            
        total_score = academic_score + study_score + sleep_score + stress_score
        
        with st.expander("🔍 Scoring Breakdown (Debug Info)"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Average Marks:** {avg_marks:.1f}/100")
                st.write(f"**Academic Score:** {academic_score}/50")
                st.write(f"**Study Score:** {study_score}/30")
            with col2:
                st.write(f"**Sleep Score:** {sleep_score}/10") 
                st.write(f"**Stress Score:** {stress_score}/10")
                st.write(f"**Total Score:** {total_score}/100")
            
            st.write("**NEW Thresholds:** High ≥75, Average 50-74, Needs Improvement <50")
    
    # Display the prediction with appropriate styling
    category = performance_categories[prediction]
    
    # Display prediction in a nice box with the category color
    st.markdown(f"""
    <div style="padding: 20px; border-radius: 10px; background-color: {category['color']}20; 
                border: 2px solid {category['color']};">
        <h3 style="color: {category['color']};">📊 Performance Category: {category['label']}</h3>
        <p>{category['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show analytics with smaller font
    with st.expander("📈 Performance Analysis"):
        # Calculate average semester marks
        avg_marks = (SEM_1 + SEM_2 + SEM_3 + SEM_4 + SEM_5) / 5
        
        # Display academic metrics with smaller font
        st.markdown("### Academic Metrics")
        st.markdown("""
        <style>
        .metric-container {
            font-size: 0.8em;
        }
        </style>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Marks", f"{avg_marks:.1f}")
        
        with col2:
            trend = SEM_5 - SEM_1
            st.metric("Academic Trend", f"{trend:+.1f}", delta=trend)
            
        with col3:
            recent_avg = (SEM_4 + SEM_5) / 2
            st.metric("Recent Performance", f"{recent_avg:.1f}")
        
        # Display wellness metrics with smaller font
        st.markdown("### Wellness & Study Habits")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 7 <= average_sleep_duration <= 9:
                sleep_status = "Optimal ✅"
            elif 6 <= average_sleep_duration < 7 or 9 < average_sleep_duration <= 10:
                sleep_status = "Good ✓"
            else:
                sleep_status = "Suboptimal ⚠️"
            st.metric("Sleep Quality", sleep_status, f"{average_sleep_duration} hrs")
        
        with col2:
            if Weekly_study_time__________________ >= 20:
                study_status = "Excellent ✅"
            elif Weekly_study_time__________________ >= 10:
                study_status = "Good ✓"
            else:
                study_status = "Needs Attention ⚠️"
            st.metric("Study Habit", study_status, f"{Weekly_study_time__________________} hrs/week")
            
        with col3:
            if Stress_Level <= 2:
                stress_status = "Low ✅"
            elif Stress_Level == 3:
                stress_status = "Moderate ⚠️"
            else:
                stress_status = "High ⚠️⚠️"
            st.metric("Stress Management", stress_status, f"Level {Stress_Level}/5")
    
    # Show recommendations
    st.subheader("📚 Recommended Study Materials")
    st.write(category['recommendations'])
    
    # Get all 5 personalized recommendations for the category
    recommendations = recommend_materials(prediction)
    
    # Display all 5 recommendations
    st.markdown("### Top 5 Resources for Your Category:")
    
    for i, (title, link) in enumerate(recommendations, 1):
        st.markdown(f"""
        <div style="padding: 15px; border-radius: 8px; background-color: #f8f9fa; 
                    margin-bottom: 10px; border-left: 4px solid {category['color']};">
            <h4 style="margin: 0 0 10px 0; color: #333; font-size: 1.1em;">{i}. {title}</h4>
            <a href="{link}" target="_blank" style="text-decoration: none; 
               background-color: {category['color']}; color: white; padding: 8px 15px; 
               border-radius: 5px; font-size: 0.9em;">📖 Access Resource</a>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional tips based on category
    st.subheader("📝 Personalized Action Plan")
    if prediction == 0:
        st.markdown("""
        ### Recommended Actions:
        1. **Establish a consistent study schedule** - Set aside specific times each day for studying
        2. **Seek help for challenging subjects** - Don't hesitate to visit office hours or tutoring centers
        3. **Improve note-taking skills** - Try different methods like Cornell notes or mind mapping
        4. **Join a study group** - Collaborative learning can help reinforce concepts
        5. **Focus on physical wellness** - Ensure adequate sleep and exercise to support learning
        """)
    elif prediction == 1:
        st.markdown("""
        ### Recommended Actions:
        1. **Apply active learning techniques** - Practice problems, teach concepts to others
        2. **Explore your interests deeper** - Find connections between coursework and your passions
        3. **Develop efficient study strategies** - Try techniques like spaced repetition and retrieval practice
        4. **Build a network** - Connect with peers and professors in your field
        5. **Balance academics with personal development** - Participate in relevant extracurriculars
        """)
    else:  # prediction == 2
        st.markdown("""
        ### Recommended Actions:
        1. **Pursue advanced learning opportunities** - Research projects, competitions, internships
        2. **Mentor other students** - Teaching is one of the best ways to deepen your understanding
        3. **Develop specialized skills** - Focus on areas that align with your career goals
        4. **Network with professionals** - Attend conferences, seminars, and industry events
        5. **Maintain work-life balance** - Continue prioritizing wellness to sustain performance
        """)
            
else:
    if not os.path.exists("edu_model.pkl"):
        st.warning("⚠️ No trained model found. The app will use analytical methods for prediction instead.")