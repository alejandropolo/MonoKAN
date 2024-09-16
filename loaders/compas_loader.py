## this has been extracted from the original code (Nolte et al, Expressive Monotonic Neural Networks, 2023)

import numpy as np
import pandas as pd
import torch

mono_list = [0, 1, 2, 3]

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
    class_list = [5, 6]
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

def load_data(path, get_categorical_info=True):

    data = pd.read_csv(path)
    #print(data) 
    # Data cleaning as performed by propublica
    data = data[data['days_b_screening_arrest'] <= 30]
    data = data[data['days_b_screening_arrest'] >= -30]
    data = data[data['is_recid'] != -1]
    data = data[data['c_charge_degree'] <= "O"]
    data = data[data['score_text'] != 'N/A']
    
    n = data.shape[0]
    n_train = int(n * .8)
    n_test = n - n_train

    replace_data = [
        ['African-American', 'Hispanic', 'Asian', 'Caucasian', 'Native American', 'Other'],
        ['Male', 'Female']
        ]
    
    for row in replace_data:
        data = data.replace(row, range(len(row)))

    data = np.array(pd.concat([
        data[[
            'priors_count',
            'juv_fel_count',
            'juv_misd_count',
            'juv_other_count',
            'age',
            'race',
            'sex',
            #'vr_charge_desc'
        ]],
        data[['two_year_recid']],
    ], axis = 1).values)+0
    
    # Shuffle for train/test splitting
    np.random.seed(seed=78712)
    np.random.shuffle(data)

    data_train = data[:n_train, :]
    data_test = data[n_train:, :]
    
    X_train = data_train[:, :7].astype(np.float64)
    y_train = data_train[:, 7].astype(np.uint8)

    X_test = data_test[:, :7].astype(np.float64)
    y_test = data_test[:, 7].astype(np.uint8)
    
    X_train, X_test, start_index, cat_length = normalize_data_ours(X_train, X_test)
    
    if get_categorical_info:
        return X_train, y_train, X_test, y_test, start_index, cat_length 
    else:
        return X_train, y_train, X_test, y_test


## ADDED FUNCTION TO LOAD DATA IN KAN FORMAT

def load_data_compas(file_path,get_categorical_info=False):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    X_train, y_train, X_test, y_test = load_data(path =file_path,get_categorical_info=False)
    
    X_train_tensor = torch.tensor(X_train).float().to(device)
    X_test_tensor = torch.tensor(X_test).float().to(device)
    y_train_tensor = torch.tensor(y_train).float().unsqueeze(1).to(device)
    y_test_tensor = torch.tensor(y_test).float().unsqueeze(1).to(device)
    
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

    monotone_constraints = [1 if i in mono_list else 0 for i in range(X_train.shape[1])]
    mono_vars = {i: value for i, value in enumerate(monotone_constraints)}
    classification = True

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, dataset, mono_vars, classification

