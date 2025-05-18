import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
 

movies_df = pd.read_csv('../../Hybrid_Recommendation_System/data/movies.csv')
print(movies_df.head())  # Print the first few rows of the DataFrame


movies_df['title'] = movies_df['title'].fillna('')  # Replace missing titles with empty strings
# Content-Based Filtering: Create a TF-IDF matrix for the movie genres
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['genres'].fillna(''))

# Compute the cosine similarity between movies based on their genres
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Collaborative Filtering: Load the ratings dataset
ratings_df = pd.read_csv('../../Hybrid_Recommendation_System/data/ratings.csv')
 

print(len(movies_df))  # Print the number of rows in the DataFrame


def get_content_based_recommendations(title, cosine_sim=cosine_sim, movies_df=movies_df, top_n=10):
    # Get the index of the movie that matches the title
    idx = movies_df[movies_df['title'] == title].index[0]

    # Compute the cosine similarity scores between the movie and all other movies
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top N similar movies (excluding the movie itself)
    top_similar_movies = sim_scores[1:top_n+1]

    # Get the movie indices and titles of the top N similar movies
    movie_indices = [movie[0] for movie in top_similar_movies]
    similar_movies = movies_df.iloc[movie_indices]['title']

    return similar_movies

# Test the content-based filtering
movie_title = 'Toy Story (1995) '
content_based_recommendations = get_content_based_recommendations(movie_title)
print(f"Content-Based Recommendations for '{movie_title}':")
print(content_based_recommendations)

def get_accuracy_content_based(test_size=0.2):
    # Split the movies dataset into a training set and a testing set
    train_movies, test_movies = train_test_split(movies_df, test_size=test_size, random_state=42)

    # Test the content-based filtering on the testing set
    num_correct_predictions = 0
    for _, movie in test_movies.iterrows():
        movie_title = movie['title']
        recommendations = get_content_based_recommendations(movie_title, cosine_sim=cosine_sim, movies_df=train_movies)
        if movie['title'] in recommendations.tolist():
            num_correct_predictions += 1

    # Calculate accuracy
    accuracy = num_correct_predictions / len(test_movies) * 100
    return accuracy

def get_accuracy_collaborative_filtering(test_size=0.2):
    # Collaborative Filtering: Split the ratings data into a training set and a testing set
    train_ratings, test_ratings = train_test_split(ratings_df, test_size=test_size, random_state=42)

    # Collaborative Filtering: Train the collaborative filtering model using the ratings dataset
    collaborative_model = LogisticRegression()
    collaborative_model.fit(train_ratings[['userId', 'movieId']], train_ratings['rating'])

    # Test the collaborative filtering model on the testing set
    test_predictions = collaborative_model.predict(test_ratings[['userId', 'movieId']])
    accuracy = np.mean(test_predictions == test_ratings['rating']) * 100
    return accuracy

# Test the accuracy of the Ninja1 system
test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
content_based_accuracies = []
collaborative_filtering_accuracies = []

for test_size in test_sizes:
    content_based_accuracy = get_accuracy_content_based(test_size)
    content_based_accuracies.append(content_based_accuracy)

    collaborative_filtering_accuracy = get_accuracy_collaborative_filtering(test_size)
    collaborative_filtering_accuracies.append(collaborative_filtering_accuracy)

# Plot the accuracy results
plt.figure(figsize=(8, 5))
plt.plot(test_sizes, content_based_accuracies, marker='o', label='Content-Based Filtering')
plt.plot(test_sizes, collaborative_filtering_accuracies, marker='o', label='Collaborative Filtering')
plt.xlabel('Test Size')
plt.ylabel('Accuracy (%)')
plt.title('Ninja1 System Accuracy')
plt.legend()
plt.grid(True)
plt.show()
