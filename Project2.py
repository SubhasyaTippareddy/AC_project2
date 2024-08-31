import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
# 5 data_types
# dia – Diastolic BP
# sys – Systolic BP
# eda – EDA
# res – Respiration
# all – Fusion of all data types 

def preprocess_data(path,type):
    data_type = []
    c = []
    faces = []
    mean = []
    min = []
    var = []
    max = []
    with open(path,'r') as f:
        for line in f:
            l = line.split(',')
            faces.append(l[0])
            data_type.append(l[1])
            c.append(l[2])
            l[-1] = l[-1].strip('\n')
            arr = np.array(l[3:],dtype=float)

            min.append(np.min(arr))
            max.append(np.max(arr))
            var.append(np.var(arr))
            mean.append(np.mean(arr))
    data_list = []
    for i in range(0,len(faces),4):
        record = {}
        record['Subject ID'] = faces[i]
        record['class'] = c[i]

        record['mean_diastolic_rate'] = mean[i]
        record['min_diastolic_rate'] = min[i]
        record['max_diastolic_rate'] = max[i]
        record['variance_diastolic_rate'] = var[i]

        record['mean_eda'] = mean[i+1]
        record['min_eda'] = min[i+1]
        record['max_eda'] = max[i+1]
        record['variance_eda'] = var[i+1]

        record['mean_systolic_rate'] = mean[i+2]
        record['min_systolic_rate'] = min[i+2]
        record['max_systolic_rate'] = max[i+2]
        record['variance_systolic_rate'] = var[i+2]

        record['mean_respiration'] = mean[i+3]
        record['min_respiration'] = min[i+3]
        record['max_respiration'] = max[i+3]
        record['variance_respiration'] = var[i+3]
        data_list.append(record)
        
    df = pd.DataFrame(data_list)
    selected_columns = ['Subject ID','class']
    if(type=='all'):
        return df
    elif(type =='dia'):
        selected_columns.extend(['mean_diastolic_rate','min_diastolic_rate','max_diastolic_rate','variance_diastolic_rate'])
    elif(type == 'sys'):
        selected_columns.extend(['mean_systolic_rate','min_systolic_rate','max_systolic_rate','variance_systolic_rate'])
    elif(type=='eda'):
        selected_columns.extend(['mean_eda','min_eda','max_eda','variance_eda'])
    elif(type=='res'):
        selected_columns.extend(['mean_respiration','min_respiration','max_respiration','variance_respiration'])
    else:
        raise ValueError("Incorrect Type Provided. There are only 5 data types - dia, sys, eda, res, all.")
    df = df[selected_columns]
    return df

def classification_evaluation(df):

    X = df.drop(['class', 'Subject ID'], axis=1)
    y = df['class']

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Initializing lists to store metrics for each fold
    conf_matrices = []
    precisions = []
    recalls = []
    accuracies = []
    f1s = []

    classifier = RandomForestClassifier()

    #10-fold cross-validation
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        classifier.fit(X_train, y_train)
        predicted = classifier.predict(X_test)
        
        #confusion matrix
        conf_matrix = confusion_matrix(y_test, predicted)
        conf_matrices.append(conf_matrix)
        
        precision = precision_score(y_test, predicted, average='weighted',zero_division=1)
        precisions.append(precision)
        
        recall = recall_score(y_test, predicted, average='weighted')
        recalls.append(recall)
        
        f1 = f1_score(y_test, predicted, average='weighted')
        f1s.append(f1)

        acc = accuracy_score(y_test, predicted)
        accuracies.append(acc)

    print("Following are the evaluation metrics for data type -", type)
    print("Confusion Matrix:")
    print(np.sum(conf_matrices,axis=0))
    print("Accuracy:", round(np.mean(accuracies),3))
    print("Precision:", round(np.mean(precisions),3))
    print("Recall:", round(np.mean(recalls),3))
    print("F1 Score:", round(np.mean(f1s),3))
    

args = sys.argv
type = args[1]
#path = './Project2Data.csv'
path = args[2]
df = preprocess_data(path,type)
classification_evaluation(df)

