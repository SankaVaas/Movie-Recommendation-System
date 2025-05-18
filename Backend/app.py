# from flask import Flask, render_template, request, jsonify
# import pandas as pd
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.decomposition import TruncatedSVD
# from fuzzywuzzy import fuzz
# import random
#
# app = Flask(__name__)
#
# # Load movie dataset
# movies_df = pd.read_csv('./data/movies.csv')
#
# # Load ratings dataset (Replace 'ratings.csv' with the filename of your ratings dataset)
# ratings_df = pd.read_csv('./data/ratings.csv')
#
# # Combine movie titles and genres into a single string for content-based filtering
# movies_df['combined'] = movies_df['title'].str.lower() + ' ' + movies_df['genres'].str.lower()
#
# # Create a count vectorizer to convert the combined text to a matrix of word counts
# count = CountVectorizer()
# count_matrix = count.fit_transform(movies_df['combined'])
#
# # Compute cosine similarity between movies based on the count matrix
# cosine_sim = cosine_similarity(count_matrix, count_matrix)
#
#
# # Perform Singular Value Decomposition (SVD) for collaborative filtering
# collaborative_model = TruncatedSVD(n_components=50)  # Adjust n_components as needed
# user_movie_ratings = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
# user_movie_ratings_matrix = user_movie_ratings.values
# collaborative_model.fit(user_movie_ratings_matrix)
#
# # Define a function to get the top N similar movies with some randomness
# def get_content_based_recommendations(title, cosine_sim=cosine_sim, movies_df=movies_df, top_n=10):
#     # Get the index of the movie that best matches the title using fuzzy matching
#     matching_titles = movies_df[movies_df['title'].apply(lambda x: fuzz.partial_ratio(x.lower(), title.lower())) >= 80]
#     if matching_titles.empty:
#         print(f"Movie '{title}' not found in the dataset.")
#         return []
#
#     idx = matching_titles.index[0]
#
#     # Compute the cosine similarity scores between the movie and all other movies
#     sim_scores = list(enumerate(cosine_sim[idx]))
#
#     # Sort the movies based on the cosine similarity scores
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#
#     # Introduce randomness to the recommendations by shuffling the top N movies
#     random.shuffle(sim_scores)
#
#     # Get the top N similar movies
#     sim_scores = sim_scores[:top_n]
#     movie_indices = [i[0] for i in sim_scores]
#     return movies_df['title'].iloc[movie_indices]
#
# # Define a function to get recommendations using collaborative filtering
# def get_collaborative_filtering_recommendations(user_id, model=collaborative_model, top_n=10):
#     # Create a list of user-movie pairs for prediction
#     user_movie_pairs = []
#     user_movie_ids = movies_df['movieId'].unique()
#     for movie_id in user_movie_ids:
#         user_movie_pairs.append([user_id, movie_id])
#
#     # Predict user's preferences for each movie using the collaborative filtering model
#     user_predictions = model.predict_proba(user_movie_pairs)
#
#     # Sort the predictions to get top N recommended movie indices
#     movie_indices = user_movie_ids[user_predictions[:, 1].argsort()[::-1]][:top_n]
#
#     return movies_df[movies_df['movieId'].isin(movie_indices)]['title']
#
# # Get recommendations for a movie title and user ID
# def get_recommendations(movie_title, user_id):
#     content_based_recommendations = get_content_based_recommendations(movie_title)
#     collaborative_recommendations = get_collaborative_filtering_recommendations(user_id=user_id)
#     all_recommendations = list(content_based_recommendations) + list(collaborative_recommendations)
#     random.shuffle(all_recommendations)
#     return all_recommendations
#
# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         movie_title = request.form['movie_title']
#         user_id = int(request.form['user_id'])
#         recommendations = get_recommendations(movie_title, user_id)
#         return render_template('index.html', recommendations=recommendations)
#     return render_template('index.html', recommendations=[])
#
# if __name__ == '__main__':
#     app.run(debug=True)


from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from fuzzywuzzy import fuzz
import random

app = FastAPI()

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Load movie dataset
movies_df = pd.read_csv('data/movies.csv')

# Load ratings dataset
ratings_df = pd.read_csv('data/ratings.csv')

# Combine movie titles and genres into a single string for content-based filtering
movies_df['combined'] = movies_df['title'].str.lower() + ' ' + movies_df['genres'].str.lower()

# Create a count vectorizer to convert the combined text to a matrix of word counts
count = CountVectorizer()
count_matrix = count.fit_transform(movies_df['combined'])

# Compute cosine similarity between movies based on the count matrix
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Perform Singular Value Decomposition (SVD) for collaborative filtering
collaborative_model = TruncatedSVD(n_components=50)
user_movie_ratings = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
user_movie_ratings_matrix = user_movie_ratings.values
collaborative_model.fit(user_movie_ratings_matrix)

# Content-based recommendations
def get_content_based_recommendations(title, cosine_sim=cosine_sim, movies_df=movies_df, top_n=10):
    matching_titles = movies_df[movies_df['title'].apply(lambda x: fuzz.partial_ratio(x.lower(), title.lower())) >= 80]
    if matching_titles.empty:
        print(f"Movie '{title}' not found in the dataset.")
        return []

    idx = matching_titles.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    random.shuffle(sim_scores)
    sim_scores = sim_scores[:top_n]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices]

# Collaborative filtering recommendations
def get_collaborative_filtering_recommendations(user_id, model=collaborative_model, top_n=10):
    # NOTE: The original code's use of predict_proba is incorrect for TruncatedSVD.
    # Instead, we use the dot product of user and item latent vectors for ranking.
    if user_id not in user_movie_ratings.index:
        return []
    user_idx = user_movie_ratings.index.get_loc(user_id)
    user_vec = model.transform(user_movie_ratings_matrix[user_idx:user_idx+1])[0]
    movie_vecs = model.components_.T
    scores = np.dot(movie_vecs, user_vec)
    top_indices = np.argsort(scores)[::-1][:top_n]
    movie_ids = user_movie_ratings.columns[top_indices]
    return movies_df[movies_df['movieId'].isin(movie_ids)]['title']

# Unified recommendations
def get_recommendations(movie_title, user_id):
    content_based_recommendations = get_content_based_recommendations(movie_title)
    collaborative_recommendations = get_collaborative_filtering_recommendations(user_id=user_id)
    all_recommendations = list(content_based_recommendations) + list(collaborative_recommendations)
    random.shuffle(all_recommendations)
    return all_recommendations

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "recommendations": []})

@app.post("/", response_class=HTMLResponse)
async def recommend(request: Request, movie_title: str = Form(...), user_id: int = Form(...)):
    recommendations = get_recommendations(movie_title, user_id)
    return templates.TemplateResponse("index.html", {"request": request, "recommendations": recommendations})

@app.post("/api/recommend", response_class=JSONResponse)
async def api_recommend(movie_title: str = Form(...), user_id: int = Form(...)):
    recommendations = get_recommendations(movie_title, user_id)
    return JSONResponse(content={"recommendations": recommendations})

# To run: uvicorn main:app --reload