## this has been extracted from the original code (Nolte et al, Expressive Monotonic Neural Networks, 2023)

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

mono_list = [50, 51, 52, 53, 55, 56, 57, 58]

def generate_one_hot_mat(mat):
    upper_bound = np.max(mat)
    mat_one_hot = np.zeros((mat.shape[0], int(upper_bound+1)))
    
    for j in range(mat.shape[0]):
        mat_one_hot[j, int(mat[j])] = 1.
        
    return mat_one_hot

def generate_normalize_numerical_mat(mat):
    if np.max(mat) == np.min(mat):
        return mat
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
            if np.max(mat) == np.min(mat):
                continue
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

def load_data(path,get_categorical_info=True):

    data_train = pd.read_csv(path + 'train.csv')
    data_test = pd.read_csv(path + 'test.csv')

    data_train = np.array(data_train.values)
    data_test = np.array(data_test.values)
    
    
    X_train = data_train[:, :280].astype(np.float64)
    y_train = data_train[:, 280].astype(np.uint8)

    X_test = data_test[:, :280].astype(np.float64)
    y_test = data_test[:, 280].astype(np.uint8)
    
    y = np.concatenate([y_train, y_test], axis=0)
    q = np.percentile(y, 90)
    
    cols = []
    for i in range(y_train.shape[0]):
        if y_train[i] > q:
            cols.append(i)
    X_train=np.delete(X_train, cols, axis=0)
    y_train=np.delete(y_train, cols, axis=0)
    
    cols = []
    for i in range(y_test.shape[0]):
        if y_test[i] > q:
            cols.append(i)
    X_test=np.delete(X_test, cols, axis=0)
    y_test=np.delete(y_test, cols, axis=0)
    X_train, X_test, start_index, cat_length = normalize_data_ours(X_train, X_test)
    
    normalized_y = generate_normalize_numerical_mat(np.concatenate([y_train, y_test], axis=0))
    y_train = normalized_y[:y_train.shape[0]]
    y_test = normalized_y[y_train.shape[0]:]

   
    if get_categorical_info:
        return X_train, y_train, X_test, y_test, start_index, cat_length 
    else:
        return X_train, y_train, X_test, y_test
    
def load_data_blog(file_path,ridged,get_categorical_info=False):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    X_train, y_train, X_test, y_test = load_data(path ='../data/Preprocessed_Data/Blog/',get_categorical_info=False)
    original_X_train = X_train.copy()
    original_X_test = X_test.copy()


    if ridged:
        print('Ridge Regression')
        model = Ridge()
        model.fit(
            X_train, y_train,
        )
        rmse = np.sqrt(np.mean((model.predict(X_test) - y_test) ** 2))
        important_feature_idxs = np.argsort(model.coef_)[::-1][:20]

        X_train = X_train[:, important_feature_idxs]
        X_test = X_test[:, important_feature_idxs]

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

    mean = X_train_tensor.mean(0)
    std = X_train_tensor.std(0)
    X_train_tensor = (X_train_tensor - mean) / std
    X_test_tensor = (X_test_tensor - mean) / std

    ## Print number of instances in train_data and test_data
    print('Number of instances in train_data:', X_train_tensor.shape)
    print('Number of instances in test_data:', X_test_tensor.shape)


    n_var = X_train_tensor.shape[1]

    dataset = dict()
    dataset['train_input'] = X_train_tensor
    dataset['train_label'] = y_train_tensor
    dataset['test_input'] = X_test_tensor
    dataset['test_label'] = y_test_tensor

    monotone_constraints = np.array(
    [1 if i in mono_list else 0 for i in range(original_X_train.shape[1])])
    if ridged:
        monotone_constraints = monotone_constraints[important_feature_idxs]
    mono_vars = {i: value for i, value in enumerate(monotone_constraints)}
    classification = False
    
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, dataset, mono_vars, classification