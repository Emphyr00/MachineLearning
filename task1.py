import utils
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

totalPred = []
totalVal = []
def train_and_predict_knn(X_train, X_test, Y_train, Y_val=None):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=30)
    knn.fit(X_train, Y_train)

    Y_pred = knn.predict(X_test)
    
    if Y_val is None:
        return Y_pred 
    else:
        # print('real: ' + str(list(y_val)))
        # print('pred: ' + str(y_pred))
        totalPred.append(Y_pred)
        totalVal.append(Y_val)
        accuracy = accuracy_score(Y_val, Y_pred)
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
        total_accuracy += train_and_predict_knn(X_train, X_val, Y_train, Y_val)
        
    totalPredFlattened = [item for array in totalPred for item in array]
    totalValFlattened = [item for array in totalVal for item in array]
    conf_matrix = confusion_matrix(totalValFlattened, totalPredFlattened)
    utils.save_confusion_matrix(conf_matrix, 'confusion_matrix1.png')
    
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
        
        # print(filmsTest)
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
        
        predictions = train_and_predict_knn(X_train, X_test, Y_train)
        for i, filmInfo in enumerate(filmsTest):
            filmInfo.append(predictions[i])
            result.append(filmInfo)
           
    utils.save_results('test.csv', result)
    return result
            

test()