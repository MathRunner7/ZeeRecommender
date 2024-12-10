"""
This is Zee Recommendation System for movie suggestion to user based on similarity score

To create and activate virtual environment run following codes in terminal
1. python -m venv environment
2. environment/Scripts/activate
"""
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle

# Load the model from the file
filename = 'item_pearson_similarity.pkl'

with open(filename, 'rb') as file:
    item_pearson_similarity = pickle.load(file)

# Make self similarity lowest to avoid recommending same movie
item_pearson_similarity = item_pearson_similarity - (np.eye(item_pearson_similarity.shape[0]) * 2)

# Load the model from the file
filename1 = 'item_cosine_similarity.pkl'

with open(filename1, 'rb') as file:
    item_cosine_similarity = pickle.load(file)

# Make self similarity lowest to avoid recommending same movie
item_cosine_similarity = item_cosine_similarity - (np.eye(item_cosine_similarity.shape[0]) * 2)


# Read all files into Pandas dataframe format
users = pd.read_csv('zee-users.dat', delimiter='::')
movies = pd.read_csv('zee-movies.dat', delimiter='::', encoding='ISO-8859-1')
ratings = pd.read_csv('zee-ratings.dat', delimiter='::')
interaction_matrix = pd.merge(ratings, movies, left_on='MovieID', right_on='Movie ID').pivot_table(index='UserID', columns='Title', values='Rating')

#TODO 1 Flask code
app = Flask(__name__)

# create end points
# Welcome point
@app.route('/')
def home():
    return 'Zee Recommender System by Ankit Thummar'

# Pearson recommendation
@app.route('/pearson', methods=['GET', 'POST'])
def pearson():
    if request.method=='GET':
        return ('This is pearson correlation based recommendation system.'
                'Send POST request to get prediction'
                'JSON Format: {"user":<integer between 1 and 6040>, "n_recommend":<integer between 1 and 50>}')
    if request.method == 'POST':
        # Function to predict a rating for given user-movie combination
        def rating_prediction(user_id, movie_title):
            user_rating = interaction_matrix.loc[user_id]
            movie_similarity = item_pearson_similarity[movie_title]
            rating_similarity = pd.merge(user_rating, movie_similarity, left_index=True, right_index=True).reset_index()
            rating_similarity = rating_similarity[rating_similarity[user_id] > 0]
            rating_similarity['prediction'] = rating_similarity[user_id] * rating_similarity[movie_title]
            prediction = rating_similarity['prediction'].sum() / (rating_similarity[movie_title]).sum()
            return prediction

        # Pearson Recommender Function
        def pearson_recommender(user_id, num_recommendations=10):
            # Watched moveis by user
            watched_movies = ratings[(ratings['UserID'] == user_id)]['MovieID'].values
            unrated_movies_title = item_pearson_similarity.columns[~(item_pearson_similarity.columns.isin(watched_movies))]

            # Define empty dataframe
            suggested_movies = {'title': [], 'rating': []}

            # Iterate through MovieIDs to find similar movies based on pearson correlation
            for movie_title in unrated_movies_title:
                # Predict rating for userID, movie_title combination
                pred = np.clip(rating_prediction(user_id, movie_title), 1, 5)
                suggested_movies['title'].append(movie_title)
                suggested_movies['rating'].append(pred)

            # Convert dictionary to dataframe
            suggested_movies = pd.DataFrame(suggested_movies)

            # Filter movies with high prediction score
            if (suggested_movies.rating > 4.5).sum() < 50:
                suggested_movies = suggested_movies.sort_values(by='rating', ascending=False).head(50).sample(
                    num_recommendations)
            else:
                suggested_movies = suggested_movies[suggested_movies.rating > 4.5].sample(num_recommendations)
            return suggested_movies.title.values

        # Query point : UserID
        userID = request.get_json()['user']
        n_recommendations = request.get_json()['n_recommend']
        return jsonify({'movies':pearson_recommender(userID, n_recommendations).tolist()})

# Cosine recommendation
@app.route('/cosine', methods=['GET', 'POST'])
def cosine():
    if request.method=='GET':
        return ('This is Cosine Similarity based recommendation system.'
                'Send POST request to get prediction'
                'JSON Format: {"user":<integer between 1 and 6040>, "n_recommend":<integer between 1 and 50>}')
    if request.method == 'POST':
        # Function to predict a rating for given user-movie combination
        def cosine_rating_prediction(user_id, movie_title):
            user_rating = interaction_matrix.loc[user_id]
            movie_similarity = item_cosine_similarity[movie_title]
            rating_similarity = pd.merge(user_rating, movie_similarity, left_index=True, right_index=True).reset_index()
            rating_similarity = rating_similarity[rating_similarity[user_id] > 0]
            rating_similarity['prediction'] = rating_similarity[user_id] * rating_similarity[movie_title]
            prediction = rating_similarity['prediction'].sum() / (rating_similarity[movie_title]).sum()
            return prediction

        # @title Pearson Recommender Function
        def cosine_recommender(user_id, num_recommendations=10):
            # Watched moveis by user
            watched_movies = ratings[(ratings['UserID'] == user_id)]['MovieID'].values
            unrated_movies_title = item_cosine_similarity.columns[~(item_cosine_similarity.columns.isin(watched_movies))]

            # Define empty dataframe
            suggested_movies = {'title': [], 'rating': []}

            # Iterate through MovieIDs to find similar movies based on pearson correlation
            for movie_title in unrated_movies_title:
                # Predict rating for userID, movie_title combination
                pred = np.clip(cosine_rating_prediction(user_id, movie_title), 1, 5)
                suggested_movies['title'].append(movie_title)
                suggested_movies['rating'].append(pred)

            # Convert dictionary to dataframe
            suggested_movies = pd.DataFrame(suggested_movies)

            # Filter movies with high prediction score
            if (suggested_movies.rating > 4.5).sum() < 50:
                suggested_movies = suggested_movies.sort_values(by='rating', ascending=False).head(50).sample(
                    num_recommendations)
            else:
                suggested_movies = suggested_movies[suggested_movies.rating > 4.5].sample(num_recommendations)
            return suggested_movies.title.values

        # Query point : UserID
        userID = request.get_json()['user']
        n_recommendations = request.get_json()['n_recommend']
        return jsonify({'movies':cosine_recommender(userID, n_recommendations).tolist()})

# CMF recommendation
@app.route('/cmf')
def cmf():
    return 'CMF Prediction'