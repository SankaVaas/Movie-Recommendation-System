from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from fuzzywuzzy import fuzz
import random

app = FastAPI()

# (Optional) For development: allow React dev server to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://localhost:3000"] for stricter CORS
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load movie dataset
movies_df = pd.read_csv('data/movies.csv')
ratings_df = pd.read_csv('data/ratings.csv')

# Combine movie titles and genres into a single string for content-based filtering
movies_df['combined'] = movies_df['title'].str.lower() + ' ' + movies_df['genres'].str.lower()
count = CountVectorizer()
count_matrix = count.fit_transform(movies_df['combined'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# SVD for collaborative filtering
collaborative_model = TruncatedSVD(n_components=50)
user_movie_ratings = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
user_movie_ratings_matrix = user_movie_ratings.values
collaborative_model.fit(user_movie_ratings_matrix)

def get_content_based_recommendations(title, cosine_sim=cosine_sim, movies_df=movies_df, top_n=10):
    matching_titles = movies_df[movies_df['title'].apply(lambda x: fuzz.partial_ratio(x.lower(), title.lower())) >= 80]
    if matching_titles.empty:
        return []
    idx = matching_titles.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    random.shuffle(sim_scores)
    sim_scores = sim_scores[:top_n]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices].tolist()

def get_collaborative_filtering_recommendations(user_id, model=collaborative_model, top_n=10):
    if user_id not in user_movie_ratings.index:
        return []
    user_idx = user_movie_ratings.index.get_loc(user_id)
    user_vec = model.transform(user_movie_ratings_matrix[user_idx:user_idx+1])[0]
    movie_vecs = model.components_.T
    scores = np.dot(movie_vecs, user_vec)
    top_indices = np.argsort(scores)[::-1][:top_n]
    movie_ids = user_movie_ratings.columns[top_indices]
    return movies_df[movies_df['movieId'].isin(movie_ids)]['title'].tolist()

def get_recommendations(movie_title, user_id):
    content_based = get_content_based_recommendations(movie_title)
    collaborative = get_collaborative_filtering_recommendations(user_id)
    all_recommendations = list(content_based) + list(collaborative)
    random.shuffle(all_recommendations)
    return all_recommendations

# Pydantic model for request body
class RecommendRequest(BaseModel):
    movie_title: str
    user_id: int

@app.post("/recommend")
async def api_recommend(request: RecommendRequest):
    recommendations = get_recommendations(request.movie_title, request.user_id)
    return JSONResponse(content={"recommendations": recommendations})