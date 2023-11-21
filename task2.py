import utils
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

TREE_MODE = False

def train_and_predict_tree(X_train, X_test, Y_train, user_id, Y_val=None, use_random_forest=False):
    scaler = StandardScaler()
    feature_names = X_train.columns
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Choose the model based on the use_random_forest flag
    if use_random_forest:
        model = RandomForestClassifier()
    else:
        model = DecisionTreeClassifier()

    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    # print(X_train)
    if not use_random_forest:
        plt.figure(figsize=(100,40))
        plot_tree(model, filled=True, feature_names=feature_names, class_names=True, rounded=True)
        plt.savefig('./tree-'+str(user_id)+'.png')
        plt.close()
    
    if Y_val is None:
        return Y_pred
    else:
        accuracy = accuracy_score(Y_val, Y_pred)
        print(str(user_id) + ': ' + str(accuracy))
        return accuracy
    
def test():
    train = utils.load_train();
    users = utils.get_unique_users(train)
    allFilmsFeatures = utils.extract_features()
    total_accuracy = 0
    num_users = len(users)

    for user in users:
        films = utils.get_films_for_user(train, lambda x: x == user)
        # print(len(films))
        userFilmsFeatures = pd.DataFrame()
        for filmInfo in films:
            filmFeatures = utils.get_dateframe_row(allFilmsFeatures, filmInfo[2])
            filmFeatures['evaluation'] = filmInfo[3]
            if userFilmsFeatures.empty: 
                userFilmsFeatures = filmFeatures
            else:
                userFilmsFeatures = pd.concat([userFilmsFeatures, filmFeatures], ignore_index=True)
                
        X = userFilmsFeatures.drop('evaluation', axis=1)
        y = userFilmsFeatures['evaluation']
        X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        total_accuracy += train_and_predict_tree(X_train, X_val, Y_train, filmInfo[1], Y_val, TREE_MODE)

    avg_accuracy = total_accuracy / num_users
    print(f'Average Validation Accuracy: {avg_accuracy}')  
    
def prod():
    train = utils.load_train();
    test = utils.load_test();
    users = utils.get_unique_users(train)
    allFilmsFeatures = utils.extract_features()
    result = []
    
    for user in users:
        filmsTrain = utils.get_films_for_user(train, lambda x: x == user)
        filmsTest = utils.get_films_for_user(test, lambda x: x == user)
        
        userFilmsTrainFeatures = pd.DataFrame()
        userFilmsTestFeatures = pd.DataFrame()
        
        for filmInfo in filmsTrain:
            filmFeatures = utils.get_dateframe_row(allFilmsFeatures, filmInfo[2])
            filmFeatures['evaluation'] = filmInfo[3]
            if userFilmsTrainFeatures.empty: 
                userFilmsTrainFeatures = filmFeatures
            else:
                userFilmsTrainFeatures = pd.concat([userFilmsTrainFeatures, filmFeatures], ignore_index=True)
        
        for filmInfo in filmsTest:
            filmFeatures = utils.get_dateframe_row(allFilmsFeatures, filmInfo[2])
            filmFeatures['evaluation'] = None
            if userFilmsTestFeatures.empty: 
                userFilmsTestFeatures = filmFeatures
            else:
                userFilmsTestFeatures = pd.concat([userFilmsTestFeatures, filmFeatures], ignore_index=True)
                
        X_train = userFilmsTrainFeatures.drop('evaluation', axis=1)
        Y_train = userFilmsTrainFeatures['evaluation']
        X_test = userFilmsTestFeatures.drop('evaluation', axis=1)
        
        predictions = train_and_predict_tree(X_train, X_test, Y_train, filmInfo[1], use_random_forest=TREE_MODE)
        for i, filmInfo in enumerate(filmsTest):
            filmInfo.append(predictions[i])
            result.append(filmInfo)
           
    utils.save_results('test_forest.csv', result)
    return result
            

prod()