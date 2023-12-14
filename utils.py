import json
import numpy as np 
import pandas as pd 
import csv
import matplotlib.pyplot as plt
import seaborn as sns  # Optional, for a nicer heatmap style
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

def load_json():
    with open('./download/data.json', 'r') as file:
        data_array = json.load(file)
        for element in data_array:
            element['release_date'] = element['release_date'].split('-')[0]

    return data_array

def load_train():
    result = []
    with open('./movie/train.csv', mode='r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        for row in reader:
            result.append([int(value) for value in row])
    return result

def save_results(path, data):
    with open(path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')
        # Write the data
        for row in data:
            writer.writerow(row)
        
def get_films_for_user(array, condition):
    return [element for element in array if condition(element[1])]

def get_unique_users(array):
    seen = set()
    unique_values = []
    for element in array:
        if element[1] not in seen:
            seen.add(element[1])
            unique_values.append(element[1])
    return unique_values

def load_test():
    result = []
    with open('./movie/task.csv', mode='r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        for row in reader:
            row_result = []
            for value in row:
                try:
                    int_value = int(value)
                    row_result.append(int_value)
                except ValueError:
                    continue  # Skip values that can't be converted to int
            result.append(row_result)
    return result

def filter_ids(data, array_names):
    if not isinstance(data, list) or not isinstance(array_names, list):
        raise ValueError("Invalid input: data should be a list and array_names should be a list")

    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Invalid item in data: Each item should be a dictionary")

        # Filter ids for each specified name in array_names
        for name in array_names:
            if name in item and isinstance(item[name], list):
                ids = [element['id'] for element in item[name] if 'id' in element]
                item[name] = ids
            else:
                item[name] = []

    return data

def encode_lists(data, feature_names):
    df = pd.DataFrame(data)
    result_df = pd.DataFrame(index=df.index)
    for name in feature_names:
        exploded_df = df[name].explode()
        one_hot_encoded = pd.get_dummies(exploded_df)
        one_hot_encoded.columns = [str(name) + '_' + str(col) for col in one_hot_encoded.columns]
        grouped_df = one_hot_encoded.groupby(level=0).sum()
        result_df = result_df.merge(grouped_df, left_index=True, right_index=True, how='left')

    return result_df

def encode_categorical_vars(data, feature_names):
    df = pd.DataFrame(data)
    result_df = pd.DataFrame(index=df.index)  # Initialize with the same index as df

    for name in feature_names:
        one_hot_encoded = pd.get_dummies(df[name])
        one_hot_encoded.columns = [str(name) + '_' + str(col) for col in one_hot_encoded.columns]
        result_df = result_df.merge(one_hot_encoded, left_index=True, right_index=True, how='left')

    result_df = result_df.fillna(0)
    return result_df

def encode_numerical_vars(data, feature_names):
    # Initialize the scaler
    scaler = MinMaxScaler()
    df = pd.DataFrame(data)
    result_df = pd.DataFrame(index=df.index)

    for column in feature_names:
        values = df[column].values.reshape(-1, 1)
        normalized_values = scaler.fit_transform(values)
        result_df[column] = normalized_values.flatten()
        
    return result_df

def extract_features():
    data = load_json()
    data = filter_ids(data, ['crew', 'keywords', 'cast', 'belongs_to_collection', 'genres'])
    features = encode_lists(data, ['cast', 'genres'])
    features = features.merge(encode_categorical_vars(data, ['original_language', 'adult']), left_index=True, right_index=True, how='left')
    features = features.merge(encode_numerical_vars(data, ['budget', 'popularity', 'release_date', 'revenue', 'runtime', 'vote_average']), left_index=True, right_index=True, how='left')
    # print(features)
    return features

def get_dateframe_row(dataframe, number):
    return dataframe.loc[[number - 1]]

def save_confusion_matrix(conf_matrix, filename):
    plt.figure(figsize=(10,7))
    sns.heatmap(conf_matrix, annot=True, fmt='g')  # 'g' format avoids scientific notation
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(filename)
    plt.close()  # Close the plot to free up memory

extract_features()