import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from transformers import RobertaTokenizer, RobertaModel
import torch
import plotly.express as px

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
    
    # Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(combined_features)
    df['performance_level'] = df['cluster'].map({0: 'Low', 1: 'Medium', 2: 'High'})
    
    # Calculate cluster metrics
    silhouette = silhouette_score(combined_features, df['cluster'])
    davies_bouldin = davies_bouldin_score(combined_features, df['cluster'])
    calinski_harabasz = calinski_harabasz_score(combined_features, df['cluster'])
    
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
    
    return df, combined_features, cluster_stats, silhouette, davies_bouldin, calinski_harabasz

# --- Recommendation System ---
def get_recommended_courses(student_id, df, features, n_recommendations=3):
    """Get recommended courses based on similar students"""
    student_idx = df[df['Student_ID'] == student_id].index[0]
    current_course = df.loc[student_idx, 'Course_Name']
    
    nn = NearestNeighbors(n_neighbors=n_recommendations+5)  # Extra to account for same courses
    nn.fit(features)
    
    distances, indices = nn.kneighbors([features[student_idx]])
    similar_students = df.iloc[indices[0]]
    
    # Filter out current course and get top recommendations
    recommendations = (
        similar_students[similar_students['Course_Name'] != current_course]
        ['Course_Name']
        .value_counts()
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
        color_discrete_map={
            'Low': '#FF6B6B',
            'Medium': '#4ECDC4',
            'High': '#45B7D1'
        },
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
    st.title("🎓 Student Performance Analytics Dashboard")
    
    # Load data with cluster metrics
    df, combined_features, cluster_stats, silhouette, davies_bouldin, calinski_harabasz = load_and_process_data()
    
    # Cluster evaluation metrics
    with st.expander("📊 Cluster Quality Metrics", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Silhouette Score", f"{silhouette:.3f}",
                     help="Higher values (closer to 1) indicate better separation")
        with col2:
            st.metric("Davies-Bouldin Index", f"{davies_bouldin:.3f}",
                     help="Lower values (closer to 0) indicate better clustering")
        with col3:
            st.metric("Calinski-Harabasz Index", f"{calinski_harabasz:.1f}",
                     help="Higher values indicate better clustering")
        
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
        st.subheader("👨‍🎓 Student Profile")
        selected_id = st.selectbox(
            "Select Student ID",
            df['Student_ID'].unique(),
            index=0
        )
        
        if selected_id:
            student = df[df['Student_ID'] == selected_id].iloc[0]
            
            st.metric("Performance Level", student['performance_level'])
            st.metric("Current Course", student['Course_Name'])
            
            # Performance metrics
            with st.container(border=True):
                st.write("**Performance Metrics**")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Quiz Score", f"{student['Quiz_Scores']}%")
                    st.metric("Time Spent", f"{student['Time_Spent_on_Videos']} mins")
                with col_b:
                    st.metric("Final Exam", f"{student['Final_Exam_Score']}%")
                    st.progress(student['Assignment_Completion_Rate']/100,
                               text=f"Assignments: {student['Assignment_Completion_Rate']}%")
            
            # Recommended courses
            with st.container(border=True):
                st.write("**Recommended Courses**")
                recommendations = get_recommended_courses(selected_id, df, combined_features)
                if recommendations:
                    for i, course in enumerate(recommendations, 1):
                        st.markdown(f"{i}. {course}")
                else:
                    st.write("No alternative courses found")
    
    with col2:
        st.subheader("🔍 Cluster Visualization")
        fig = create_interactive_plot(df)
        st.plotly_chart(fig, use_container_width=True,
                       config={'displayModeBar': False})
        
        # Interpretation guide
        with st.expander("ℹ️ How to interpret these metrics"):
            st.markdown("""
            - **Silhouette Score**: Measures how similar an object is to its own cluster vs others  
              → Ideal: Close to 1 (well-separated clusters)
            - **Davies-Bouldin**: Ratio of within-cluster to between-cluster distances  
              → Ideal: Close to 0 (tight, well-separated clusters)
            - **Calinski-Harabasz**: Ratio of between-cluster to within-cluster dispersion  
              → Higher values indicate better clustering
            """)

if __name__ == "__main__":
    main()