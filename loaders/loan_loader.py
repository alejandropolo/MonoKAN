## this has been extracted from the original code (Nolte et al, Expressive Monotonic Neural Networks, 2023)

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge


mono_list = [0, 1, 2, 3, 4]

def generate_one_hot_mat(mat):
    upper_bound = np.max(mat)
    mat_one_hot = np.zeros((mat.shape[0], int(upper_bound+1)))
    
    for j in range(mat.shape[0]):
        mat_one_hot[j, int(mat[j])] = 1.
        
    return mat_one_hot

def generate_normalize_numerical_mat(mat):
    mat = (mat - np.min(mat))/(np.max(mat) - np.min(mat))
    #mat = 2 * (mat - 0.5)
    
    return mat
    
def normalize_data_ours(data_train, data_test):
    ### in this function, we normalize all the data to [0, 1], and bring education_num, capital gain, hours per week to the first three columns, norm to [0, 1]
    n_train = data_train.shape[0]
    n_test = data_test.shape[0]
    data_feature = np.concatenate((data_train, data_test), axis=0)
    
    data_feature_normalized = np.zeros((n_train+n_test, 1))
    class_list = []
#     class_list = [1, 3, 5, 6, 7, 8, 9, 13]
#     mono_list = [4, 10, 12]
    ### store the class variables
    start_index = []
    cat_length = []
    ### Normalize Mono Features
    for i in range(data_feature.shape[1]):
        if i in mono_list:
            if i == mono_list[0]:
                mat = data_feature[:, i]
                mat = mat[:, np.newaxis]
                data_feature_normalized = generate_normalize_numerical_mat(mat)
            else:
                mat = data_feature[:, i]
                mat = generate_normalize_numerical_mat(mat)
                mat = mat[:, np.newaxis]
                #print(adult_feature_normalized.shape, mat.shape)
                data_feature_normalized = np.concatenate((data_feature_normalized, mat), axis=1)
            
        else:
            continue
    ### Normalize non-mono features and turn class labels to one-hot vectors
    for i in range(data_feature.shape[1]):
        if i in mono_list:
            continue
        elif i in class_list:
            continue
        else:
            mat = data_feature[:, i]
            mat = generate_normalize_numerical_mat(mat)
            mat = mat[:, np.newaxis]
            data_feature_normalized = np.concatenate((data_feature_normalized, mat), axis=1)
    
    for i in range(data_feature.shape[1]):
        if i in mono_list:
            continue
        elif i in class_list:
            mat = data_feature[:, i]
            mat = generate_one_hot_mat(mat)
            start_index.append(data_feature_normalized.shape[1])
            cat_length.append(mat.shape[1])
            data_feature_normalized = np.concatenate((data_feature_normalized, mat), axis=1)
        else:
            continue
    
    data_train = data_feature_normalized[:n_train, :]
    data_test = data_feature_normalized[n_train:, :]
    
    assert data_test.shape[0] == n_test
    assert data_train.shape[0] == n_train
    
    return data_train, data_test, start_index, cat_length 

def load_data(path="./data/preprocessed.csv", get_categorical_info=True):

    data = pd.read_csv(path)
    data['grade'] = data['grade'].replace({'A':7, 'B':6, 'C':5, 'D':4, 'E':3, 'F':2, 'G':1})
    grade = data['grade']
    data = data.drop('grade', axis=1)
    data.insert(1, 'grade', grade)
    data['emp_length'] = data['emp_length'].replace({'< 1 year':1, '1 year':2, '2 years':3, '3 years':4, '4 years':5, '5 years':6, '6 years':7, '7 years':8, '8 years':9, '9 years':10, '10+ years':11})

    # Data cleaning as performed by propublica
    #print(data.shape)
    #print(data['pub_rec_bankruptcies'].value_counts())
    #data.dropna(axis=0)
    #print(data.shape)
    # for col in data.columns:
    #     print(col) 

    data = np.array(data.values)
    data = data[:, 1:]
    n = data.shape[0]
    n_train = int(n * .8)
    n_test = n - n_train
    # Shuffle for train/test splitting
    np.random.seed(seed=78712)
    np.random.shuffle(data)
    #print(data.shape)
    
    cols = []
    for i in range(data.shape[0]):
        if (np.isnan(data[i, 1])) or (np.isnan(data[i, 2])):
            cols.append(i)
 
    data=np.delete(data, cols, axis=0)
    data_train = data[:n_train, :]
    data_test = data[n_train:, :]
    X_train = data_train[:, :28].astype(np.float64)
    y_train = data_train[:, 28].astype(np.uint8)

    X_test = data_test[:, :28].astype(np.float64)
    y_test = data_test[:, 28].astype(np.uint8)
    
    X_train, X_test, start_index, cat_length = normalize_data_ours(X_train, X_test)
   
    # mono: grade(-), pub-bankrupt(+), emp_length(-), annual inc(-), dti(+), 
    X_train[:, 0] = 1.- X_train[:, 0]
    X_train[:, 2] = 1.- X_train[:, 2]
    X_train[:, 3] = 1.- X_train[:, 3]
    X_test[:, 0] = 1.- X_test[:, 0]
    X_test[:, 2] = 1.- X_test[:, 2]
    X_test[:, 3] = 1.- X_test[:, 3]

    if get_categorical_info:
        return X_train, y_train, X_test, y_test, start_index, cat_length 
    else:
        return X_train, y_train, X_test, y_test

def load_data_loan(file_path,ridged = False, get_categorical_info=False):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    X_train, y_train, X_test, y_test = load_data(path=file_path,get_categorical_info=get_categorical_info)

    original_X_train = X_train.copy()
    original_X_test = X_test.copy()

    if ridged:
        print('Ridge Regression')
        ridge = Ridge()
        ridge.fit(X_train, y_train)
        top_features = np.argsort(np.abs(ridge.coef_))[-15:]


        X_train = X_train[:, top_features]
        X_test = X_test[:, top_features]

    X_train_tensor = torch.tensor(X_train).float().to(device)
    X_test_tensor = torch.tensor(X_test).float().to(device)
    y_train_tensor = torch.tensor(y_train).float().unsqueeze(1).to(device)
    y_test_tensor = torch.tensor(y_test).float().unsqueeze(1).to(device)

    # mean = X_train_tensor.mean(0)
    # std = X_train_tensor.std(0)
    # X_train_tensor = (X_train_tensor - mean) / std
    # X_test_tensor = (X_test_tensor - mean) / std

    ## Print number of instances in train_data and test_data
    print('Number of instances in train_data:', X_train_tensor.shape)
    print('Number of instances in test_data:', X_test_tensor.shape)


    n_var = X_train_tensor.shape[1]

    dataset = dict()
    dataset['train_input'] = X_train_tensor
    dataset['train_label'] = y_train_tensor
    dataset['test_input'] = X_test_tensor
    dataset['test_label'] = y_test_tensor

    monotone_constraints = np.array([int(i in mono_list) for i in range(original_X_train.shape[1])])
    if ridged:
        monotone_constraints = monotone_constraints[top_features]
    mono_vars = {i: value for i, value in enumerate(monotone_constraints)}
    classification = True
    
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, dataset, mono_vars, classification



