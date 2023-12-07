import utils
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

def compute_gradients(features_matrix, parameters_matrix, ratings, movie_id_to_index, person_id_to_index):
    gradients_features = np.zeros_like(features_matrix)
    gradients_parameters = np.zeros_like(parameters_matrix)

    for index, row in ratings.iterrows():
        movie_index = movie_id_to_index[row['movie_id']]
        person_index = person_id_to_index[row['person_id']]
        actual_rating = row['actual_rating']

        predicted_rating = np.dot(features_matrix[movie_index, :], parameters_matrix[person_index, :].T)

        error = predicted_rating - actual_rating

        gradients_features[movie_index, :] += error * parameters_matrix[person_index, :]
        gradients_parameters[person_index, :] += error * features_matrix[movie_index, :]

    return gradients_features, gradients_parameters

def predict_ratings(features_matrix, parameters_matrix, test_data, movie_id_to_index, person_id_to_index):
    predicted_ratings_test = []

    for _, row in test_data.iterrows():
        movie_index = movie_id_to_index.get(row['movie_id'])
        person_index = person_id_to_index.get(row['person_id'])

        predicted_rating = np.dot(features_matrix[movie_index, :], parameters_matrix[person_index, :].T)
        predicted_rating = max(0, min(5, predicted_rating))
        predicted_rating = round(predicted_rating)
        predicted_ratings_test.append(predicted_rating)
    

    return predicted_ratings_test

def train_model(train_data, num_features, learning_rate, max_iterations, epsilon):
    df = pd.DataFrame(train_data, columns=["id", "person_id", "movie_id", "rating"])

    pivot_table = df.pivot_table(index="movie_id", columns="person_id", values="rating", aggfunc=np.sum)

    movie_id_to_index = {movie_id: index for index, movie_id in enumerate(pivot_table.index)}
    person_id_to_index = {person_id: index for index, person_id in enumerate(pivot_table.columns)}

    num_movies = len(movie_id_to_index)
    num_persons = len(person_id_to_index)

    features_matrix = np.random.uniform(-1, 1, (num_movies, num_features))
    parameters_matrix = np.random.uniform(-1, 1, (num_persons, num_features))

    ratings = pivot_table.stack().reset_index()
    ratings.columns = ['movie_id', 'person_id', 'actual_rating']

    for iteration in range(max_iterations):
        gradients_features, gradients_parameters = compute_gradients(
            features_matrix, parameters_matrix, ratings, movie_id_to_index, person_id_to_index
        )

        features_matrix -= learning_rate * gradients_features
        parameters_matrix -= learning_rate * gradients_parameters

        predicted_ratings = np.array([
            np.dot(features_matrix[movie_id_to_index[movie_id]], parameters_matrix[person_id_to_index[person_id]].T)
            for movie_id, person_id in zip(ratings['movie_id'], ratings['person_id'])
        ])
        actual_ratings = ratings['actual_rating'].values
        cost = np.sum((actual_ratings - predicted_ratings) ** 2) / 2

        if cost < epsilon:
            print(f"Convergence reached after {iteration} iterations.")
            break

        print(f"Iteration {iteration}, Cost: {cost}")
            

    return features_matrix, parameters_matrix, movie_id_to_index, person_id_to_index

def prod():
    train = utils.load_train()
    test = utils.load_test()

    num_features = 100
    learning_rate = 0.001
    max_iterations = 10000
    epsilon = 0.001
    
    features_matrix, parameters_matrix, movie_id_to_index, person_id_to_index = train_model(
        train, num_features, learning_rate, max_iterations, epsilon
    )

    test_df = pd.DataFrame(test, columns=["id", "person_id", "movie_id"])
    
    test_df['predicted_rating'] = predict_ratings(
        features_matrix, parameters_matrix, test_df, movie_id_to_index, person_id_to_index
    )
    
    predictions_array = test_df.to_numpy().tolist()

    return predictions_array

predicted_test_df = prod()
utils.save_results('submission_lol.csv', predicted_test_df)