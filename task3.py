import utils
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from scipy.stats import pearsonr

def create_similarity_matrix(data):
    user_ratings = data.pivot_table(index='user_id', columns='film_id', values='rating')
    similarity_matrix = pd.DataFrame(index=user_ratings.index, columns=user_ratings.index)
    for user1 in user_ratings.index:
        for user2 in user_ratings.index:
            if user1 != user2:
                if pd.isna(similarity_matrix.at[user1, user2]):
                    similarity = calculate_pearson_similarity(user_ratings.loc[user1], user_ratings.loc[user2])
                    similarity_matrix.at[user1, user2] = similarity
                    similarity_matrix.at[user2, user1] = similarity

    return similarity_matrix

def calculate_pearson_similarity(user1_ratings, user2_ratings):
    common_movies = user1_ratings.dropna().index.intersection(user2_ratings.dropna().index)

    if len(common_movies) < 2:
        return 0

    ratings1 = user1_ratings[common_movies]
    ratings2 = user2_ratings[common_movies]

    if np.std(ratings1) == 0 or np.std(ratings2) == 0:
        return 0

    correlation, _ = pearsonr(ratings1, ratings2)
    return correlation

def predict_rating(user_id, movie_id, similarity_matrix, user_ratings):
    similar_users = similarity_matrix.loc[user_id].dropna().sort_values(ascending=False)
    total_weight = 0
    weighted_sum = 0

    for similar_user, similarity_score in similar_users.items():
        if movie_id in user_ratings.loc[similar_user].dropna().index:
            rating = user_ratings.loc[similar_user, movie_id]
            weighted_sum += rating * similarity_score
            total_weight += similarity_score

    if total_weight > 0:
        return weighted_sum / total_weight
    else:
        return 3

def test():
    train = utils.load_train()

    trainDataFrame = pd.DataFrame(train, columns=['id', 'user_id', 'film_id', 'rating'])
    testDataFrame = trainDataFrame.sample(n=500, random_state=42)
    trainDataFrame = trainDataFrame.drop(testDataFrame.index)
    actual_ratings = testDataFrame['rating']
    similarity_matrix = create_similarity_matrix(trainDataFrame)
    user_ratings_pivot = trainDataFrame.pivot_table(index='user_id', columns='film_id', values='rating', aggfunc='first')
    predicted_ratings = []
    for _, row in testDataFrame.iterrows():
        id = row['id']
        user_id = row['user_id']
        film_id = row['film_id']
        predicted_rating = predict_rating(user_id, film_id, similarity_matrix, user_ratings_pivot)
        predicted_rating = np.clip(np.rint(predicted_rating), 0, 5).astype(int)
        predicted_ratings.append(predicted_rating)

    predicted_ratings = np.array(predicted_ratings)
    accuracy = calculate_accuracy(predicted_ratings, actual_ratings)
    conf_matrix = confusion_matrix(actual_ratings, predicted_ratings)
    utils.save_confusion_matrix(conf_matrix, 'confusion_matrix3.png')
    print(f"Accuracy: {accuracy}")
    return accuracy

def calculate_accuracy(predicted_ratings, actual_ratings, threshold=0.5):
    correct_predictions = 0
    for predicted, actual in zip(predicted_ratings, actual_ratings):
        if abs(predicted - actual) <= threshold:
            correct_predictions += 1

    accuracy = correct_predictions / len(actual_ratings)
    return accuracy  
    
def predict_all_ratings(user_id, similarity_matrix, user_ratings):
    all_movies = user_ratings.columns
    predicted_ratings = {movie_id: predict_rating(user_id, movie_id, similarity_matrix, user_ratings) for movie_id in all_movies}

    return predicted_ratings

def prod():
    train = utils.load_train()
    test = utils.load_test()

    trainDataFrame = pd.DataFrame(train, columns=['id', 'user_id', 'film_id', 'rating'])
    testDataFrame = pd.DataFrame(test, columns=['id', 'user_id', 'film_id'])

    similarity_matrix = create_similarity_matrix(trainDataFrame)
    user_ratings_pivot = trainDataFrame.pivot_table(index='user_id', columns='film_id', values='rating', aggfunc='first')
    result = []

    for _, row in testDataFrame.iterrows():
        id = row['id']
        user_id = row['user_id']
        film_id = row['film_id']
        predicted_rating = predict_rating(user_id, film_id, similarity_matrix, user_ratings_pivot)
        predicted_rating = np.clip(np.rint(predicted_rating), 0, 5).astype(int)
        result.append([id, user_id, film_id, predicted_rating])

    utils.save_results('submission_random_picking.csv', result)
    return result
            

test()