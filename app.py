import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from transformers import RobertaTokenizer, RobertaModel
import torch
import plotly.express as px
import plotly.graph_objects as go

# --- Page Config (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configuration ---
EMBEDDING_CACHE_DIR = "cached_embeddings"
os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)

# --- Cached Resources ---
@st.cache_resource
def load_roberta():
    return RobertaTokenizer.from_pretrained('roberta-base'), RobertaModel.from_pretrained('roberta-base')

tokenizer, model = load_roberta()

# --- Embedding Management ---
def generate_embeddings(text_list):
    inputs = tokenizer(text_list, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

def get_cached_embeddings(df):
    """Load or generate and cache embeddings"""
    cache_files = {
        'course': os.path.join(EMBEDDING_CACHE_DIR, "course_embeddings.npy"),
        'learning_style': os.path.join(EMBEDDING_CACHE_DIR, "learning_style_embeddings.npy")
    }
    
    # Generate if not cached
    if not all(os.path.exists(f) for f in cache_files.values()):
        course_embs = generate_embeddings(df['Course_Name'].tolist())
        style_embs = generate_embeddings(df['Learning_Style'].apply(lambda x: f"{x} learner").tolist())
        
        np.save(cache_files['course'], course_embs)
        np.save(cache_files['learning_style'], style_embs)
    else:
        course_embs = np.load(cache_files['course'])
        style_embs = np.load(cache_files['learning_style'])
    
    return course_embs, style_embs

# --- Elbow Method ---
def plot_elbow_method(features, max_k=8):
    """Calculate and plot the elbow method for optimal k"""
    distortions = []
    K = range(1, max_k+1)
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features)
        distortions.append(kmeans.inertia_)
    
    # Create elbow plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(K),
        y=distortions,
        mode='lines+markers',
        name='Distortion'
    ))
    
    fig.update_layout(
        title='<b>Elbow Method for Optimal k</b>',
        xaxis_title='Number of Clusters (k)',
        yaxis_title='Distortion (Within-cluster SSE)',
        height=500,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

# --- Data Processing ---
@st.cache_data
def load_and_process_data():
    df = pd.read_csv('personalized_learning_dataset.csv')
    
    # Get embeddings (cached or generated)
    course_embs, style_embs = get_cached_embeddings(df)
    
    # Numerical features
    num_features = df[['Quiz_Scores', 'Final_Exam_Score', 'Time_Spent_on_Videos', 'Assignment_Completion_Rate']]
    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(num_features)
    
    # Combine all features
    combined_features = np.hstack([num_scaled, course_embs, style_embs])
    
    return df, combined_features


# --- Recommendation System ---
def get_recommended_courses(student_id, df, features, n_recommendations=3):
    """Get recommended courses based on similar students using KNN"""
    student_idx = df[df['Student_ID'] == student_id].index[0]
    current_course = df.loc[student_idx, 'Course_Name']
    
    # Use cosine distance metric which works well with embeddings
    nn = NearestNeighbors(n_neighbors=20, metric='cosine')  # Find more neighbors to get diverse courses
    nn.fit(features)
    
    distances, indices = nn.kneighbors([features[student_idx]])
    similar_students = df.iloc[indices[0]]
    
    # Filter out current course and get top recommendations
    recommendations = (
        similar_students[similar_students['Course_Name'] != current_course]
        .groupby('Course_Name')
        .agg({
            'Quiz_Scores': 'mean',  # Prioritize courses with higher performance
            'Student_ID': 'count'   # Count how many similar students took each course
        })
        .sort_values(['Student_ID', 'Quiz_Scores'], ascending=[False, False])
        .index
        .tolist()
    )
    
    return recommendations[:n_recommendations]

# --- Interactive Visualization ---
def create_interactive_plot(df):
    fig = px.scatter(
        df,
        x='pca_x',
        y='pca_y',
        color='performance_level',
        hover_data={
            'Student_ID': True,
            'Course_Name': True,
            'Quiz_Scores': True,
            'Final_Exam_Score': True,
            'performance_level': True
        },
        title='<b>Student Performance Clusters</b>',
        labels={'performance_level': 'Performance Level'}
    )
    
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        ),
        plot_bgcolor='rgba(240,240,240,0.8)',
        height=600,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


# --- Main App ---
def main():
    st.title("🎓 Interactive Student Performance Dashboard")
    
    # Load data (cached)
    df, combined_features = load_and_process_data()
    
    # Show elbow method before clustering
    st.subheader("🔍 Determine Optimal Number of Clusters")
    elbow_fig = plot_elbow_method(combined_features)
    st.plotly_chart(elbow_fig, use_container_width=True)
    
    # Let user select k based on elbow plot (outside cached function)
    selected_k = st.slider(
        "Select number of clusters (k) based on elbow plot",
        min_value=2,
        max_value=8,
        value=3,
        step=1
    )
    
    # Perform clustering with selected k
    kmeans = KMeans(n_clusters=selected_k, random_state=42)
    df['cluster'] = kmeans.fit_predict(combined_features)
    
    # Automatically label clusters if k=3
    if selected_k == 3:
        df['performance_level'] = df['cluster'].map({0: 'Low', 1: 'Medium', 2: 'High'})
    else:
        # For other k values, just number them
        df['performance_level'] = df['cluster'].apply(lambda x: f"Cluster {x+1}")
    
    # Calculate silhouette score
    silhouette = silhouette_score(combined_features, df['cluster'])
    st.metric("Silhouette Score", f"{silhouette:.3f}",
             help="Measures cluster separation (-1 to 1, higher is better)")
    
    # Cluster stats
    cluster_stats = df.groupby('cluster').agg({
        'Quiz_Scores': 'mean',
        'Final_Exam_Score': 'mean',
        'Time_Spent_on_Videos': 'mean',
        'Assignment_Completion_Rate': 'mean',
        'Student_ID': 'count'
    }).rename(columns={
        'Quiz_Scores': 'Avg Quiz Score',
        'Final_Exam_Score': 'Avg Final Exam',
        'Time_Spent_on_Videos': 'Avg Time Spent',
        'Assignment_Completion_Rate': 'Avg Assignments',
        'Student_ID': 'Students Count'
    })
    
    # Add PCA coordinates for visualization
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(combined_features)
    df['pca_x'] = emb_2d[:, 0]
    df['pca_y'] = emb_2d[:, 1]
    
    # Rest of your main function remains the same...
    # Cluster statistics
    with st.expander("📊 Cluster Performance Overview", expanded=True):
        st.dataframe(
            cluster_stats.style.format({
                'Avg Quiz Score': '{:.1f}%',
                'Avg Final Exam': '{:.1f}%',
                'Avg Assignments': '{:.1f}%',
                'Avg Time Spent': '{:.1f}',
                'Students Count': '{:.0f}'
            }),
            use_container_width=True
        )
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("👨‍🎓 Student Selector")
        selected_id = st.selectbox(
            "Choose Student ID",
            df['Student_ID'].unique(),
            index=0
        )
        
        if selected_id:
            student = df[df['Student_ID'] == selected_id].iloc[0]
            
            st.metric("Performance Level", student['performance_level'])
            st.metric("Current Course", student['Course_Name'])
            st.metric("Quiz Score", f"{student['Quiz_Scores']}%")
            st.metric("Final Exam", f"{student['Final_Exam_Score']}%")
            
            # Progress bars
            st.progress(student['Assignment_Completion_Rate']/100, 
                       text=f"Assignments Completed: {student['Assignment_Completion_Rate']}%")
            st.progress(min(student['Time_Spent_on_Videos']/500, 1.0),
                       text=f"Time Spent: {student['Time_Spent_on_Videos']} mins")
            
            # Recommended courses
            st.subheader("📚 Recommended Courses")
            recommendations = get_recommended_courses(selected_id, df, combined_features)
            if recommendations:
                for i, course in enumerate(recommendations, 1):
                    st.markdown(f"{i}. **{course}**")
            else:
                st.write("No recommendations available")
    
    with col2:
        st.subheader("🔍 Cluster Visualization")
        fig = create_interactive_plot(df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation guide
        with st.expander("ℹ️ How to use the Elbow Method"):
            st.markdown("""
            1. Look for the 'elbow' point where the distortion starts decreasing linearly
            2. The optimal k is typically at this elbow point
            3. Silhouette score helps validate your choice (higher is better)
            4. Adjust the slider to test different k values
            """)

if __name__ == "__main__":
    main()