import streamlit as st
import pickle

@st.cache_data
def load_data():
    movies = pickle.load(open('movies.pkl', 'rb'))
    similarity = pickle.load(open('similarity.pkl', 'rb'))
    return movies, similarity

def recommend_movies(movie, movies_df, similarity_matrix):
    movie_index = movies_df[movies_df['title'] == movie].index[0]
    distances = similarity_matrix[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommendations = []
    for i in movies_list:
        recommendations.append(movies_df.iloc[i[0]]['title'])
    return recommendations

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")

st.title("🎬 Movie Recommendation System")

movies_df, similarity_matrix = load_data()

selected_movie = st.selectbox("Choose a movie:", movies_df['title'].values)

if st.button("Get Recommendations"):
    recommendations = recommend_movies(selected_movie, movies_df, similarity_matrix)
    
    st.subheader("Recommended Movies:")
    for i, movie in enumerate(recommendations, 1):
        st.write(f"**{i}.** {movie}")