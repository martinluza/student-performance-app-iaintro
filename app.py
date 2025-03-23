import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from transformers import RobertaTokenizer, RobertaModel
import torch

# Load the dataset
@st.cache_data  # Cache the dataset for faster loading
def load_data():
    # Replace this with your actual dataset loading logic
    df = pd.read_csv('personalized_learning_dataset.csv')
    return df

df = load_data()

# Preprocess the data
def preprocess_data(df):
    # Select relevant features for clustering
    features = df[['Time_Spent_on_Videos', 'Quiz_Scores', 'Forum_Participation', 
                   'Assignment_Completion_Rate', 'Final_Exam_Score']]
    
    # Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(features_scaled)
    df['performance_level'] = df['cluster'].map({0: 'Low', 1: 'Medium', 2: 'High'})
    
    # Map Engagement_Level to numerical values
    engagement_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    df['Engagement_Level'] = df['Engagement_Level'].map(engagement_mapping)
    
    return df, features_scaled

df, features_scaled = preprocess_data(df)

# Function to recommend courses
def recommend_courses(student_id):
    # Create a feature matrix for recommendation
    recommendation_features = df[['Quiz_Scores', 'Final_Exam_Score', 'Engagement_Level']]
    
    # Fit a Nearest Neighbors model
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(recommendation_features)
    
    # Get the student's data
    student_data = df[df['Student_ID'] == student_id][['Quiz_Scores', 'Final_Exam_Score', 'Engagement_Level']]
    
    # Find nearest neighbors
    distances, indices = nn.kneighbors(student_data)
    nearest_neighbors = df.iloc[indices[0]]
    
    # Exclude the student themselves from the recommendations
    nearest_neighbors = nearest_neighbors[nearest_neighbors['Student_ID'] != student_id]
    
    # Get the most common courses among the nearest neighbors
    recommended_courses = nearest_neighbors['Course_Name'].mode().tolist()
    
    return recommended_courses

# Function for adaptive learning suggestions
def adaptive_learning(student_id):
    # Get the student's data
    student_data = df[df['Student_ID'] == student_id]
    
    # Initialize a list to store weaknesses
    weaknesses = []
    
    # Analyze Quiz Scores
    quiz_score = student_data['Quiz_Scores'].values[0]
    if quiz_score < 60:  # Example threshold for low quiz scores
        weaknesses.append("quiz performance")
    
    # Analyze Final Exam Scores
    final_exam_score = student_data['Final_Exam_Score'].values[0]
    if final_exam_score < 60:  # Example threshold for low final exam scores
        weaknesses.append("final exam performance")
    
    # Analyze Engagement Level
    engagement_level = student_data['Engagement_Level'].values[0]
    if engagement_level == 0:  # 0 corresponds to 'Low'
        weaknesses.append("engagement and participation")
    
    # Analyze Assignment Completion Rate
    assignment_completion = student_data['Assignment_Completion_Rate'].values[0]
    if assignment_completion < 70:  # Example threshold for low assignment completion
        weaknesses.append("assignment completion")
    
    # Generate recommended content based on weaknesses
    if weaknesses:
        recommended_content = f"Focus on improving: {', '.join(weaknesses)}."
    else:
        recommended_content = "You're doing great! Keep up the good work."
    
    return recommended_content

# Function to analyze course materials using RoBERTa
def analyze_course_materials(course_materials):
    # Load pre-trained RoBERTa model and tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')
    
    # Tokenize and get embeddings
    inputs = tokenizer(course_materials, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    
    # Get the embeddings (CLS token)
    embeddings = outputs.last_hidden_state[:, 0, :]
    
    return embeddings

# Streamlit app
def main():
    st.title("Student Performance Clustering and Recommendation System")
    st.write("This app helps analyze student performance and provides personalized recommendations.")
    
    # Input for Student ID
    student_id = st.text_input("Enter Student ID (e.g., S00001):")
    
    if student_id:
        if student_id not in df['Student_ID'].values:
            st.error(f"Student ID {student_id} not found in the dataset.")
        else:
            # Display performance level
            performance_level = df[df['Student_ID'] == student_id]['performance_level'].values[0]
            st.subheader(f"Performance Level: {performance_level}")
            
            # Display recommended courses
            recommended_courses = recommend_courses(student_id)
            st.subheader("Recommended Courses:")
            if recommended_courses:
                for course in recommended_courses:
                    st.write(f"- {course}")
            else:
                st.write("No recommendations available.")
            
            # Display adaptive learning suggestions
            st.subheader("Adaptive Learning Suggestions:")
            suggestions = adaptive_learning(student_id)
            st.write(suggestions)
    
    # Section for RoBERTa-based course material analysis
    st.header("Analyze Course Materials")
    course_materials = st.text_area("Enter course materials (one per line):", 
                                   "Introduction to Python\nAdvanced Machine Learning\nData Structures and Algorithms")
    if st.button("Analyze"):
        materials_list = course_materials.split("\n")
        embeddings = analyze_course_materials(materials_list)
        st.subheader("Course Material Embeddings:")
        st.write(embeddings)

# Run the app
if __name__ == "__main__":
    main()