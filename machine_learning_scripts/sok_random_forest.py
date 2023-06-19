import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import pickle

supervised_path = 'C:/Users/12103/PycharmProjects/cdl_remade/venv/train_test_supervised_with_timestamp/'
file1 = open('apps-sok.txt', 'r')
Lines = file1.readlines()
list_of_train = [1,2,4]
list_of_test = [3]
df_train_all = pd.DataFrame()
df_test_all  = pd.DataFrame()

def display(results):
    print(f'Best parameters are: {results.best_params_}')
    print("\n")
    mean_score = results.cv_results_['mean_test_score']
    std_score = results.cv_results_['std_test_score']
    params = results.cv_results_['params']
    for mean,std,params in zip(mean_score,std_score,params):
        print(f'{round(mean,3)} + or -{round(std,3)} for the {params}')



for line in Lines:

    content = line.strip()
    #print(content)
    for k in list_of_train:
        path = supervised_path  + content + '-' + str(k) +'.pkl'
        #path = 'C:/Users/sadun/PycharmProjects/cdl_remade/venv/data-CDL/' + content + '/' + content + '-' + str(k) + '_freqvector_full.csv'
        #print(path)
        picklefile_train = open(path, 'rb')
        df_individual_train = pickle.load(picklefile_train)
        picklefile_train.close()
        df_train_all = pd.concat([df_train_all, df_individual_train], axis=0)

    for ki in list_of_test:
        path = supervised_path + content + '-' + str(ki) +'.pkl'
        #path = 'C:/Users/sadun/PycharmProjects/cdl_remade/venv/data-CDL/' + content + '/' + content + '-' + str(k) + '_freqvector_full.csv'
        print(path)
        picklefile_test = open(path, 'rb')
        df_individual_test = pickle.load(picklefile_test)
        picklefile_test.close()
        df_test_all = pd.concat([df_test_all, df_individual_test], axis=0)



#exit()
print(df_train_all.shape)
print(df_test_all.shape)

train_data_x = df_train_all.iloc[:, :-1]
train_data_y = df_train_all.iloc[:, -1:]

test_data_x = df_test_all.iloc[:, :-1]
test_data_y = df_test_all.iloc[:, -1:]

sc = StandardScaler()
#sc = Normalizer()
scaled_train_data_x = sc.fit_transform(train_data_x)
scaled_test_data_x = sc.fit_transform(test_data_x)

print(train_data_y.values.ravel())
rfc = RandomForestClassifier(n_estimators=200, max_depth=16, max_features=100)
parameters = {
    "n_estimators":[200]
}


rfc.fit(scaled_train_data_x, train_data_y.values.ravel())
y_pred = rfc.predict(scaled_test_data_x)
print(y_pred)
print(y_pred.shape)
accuracy = metrics.accuracy_score(y_pred, test_data_y)


#---------------Metrics------------

print(accuracy)
print(confusion_matrix(test_data_y,y_pred))
print(classification_report(test_data_y,y_pred))
print(accuracy_score(test_data_y, y_pred))


precision = precision_score(test_data_y, y_pred)
print('Precision: %.3f', precision)
recall = recall_score(test_data_y, y_pred)
print('Recall: %.3f', recall)
score = f1_score(test_data_y, y_pred)
print('F-Measure: %.3f', score)
