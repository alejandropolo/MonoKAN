## this has been extracted from the original code (Nolte et al, Expressive Monotonic Neural Networks, 2023)

import pandas as pd
import torch

def load_data_heart(file_path):
    df_train = pd.read_csv(file_path + "heart_train.csv", index_col=0)
    df_test = pd.read_csv(file_path+"heart_test.csv", index_col=0)
    X_train_tensor, y_train_tensor = preprocess(df_train)
    X_test_tensor, y_test_tensor = preprocess(df_test)

    ## Print number of instances in train_data and test_data
    print('Number of instances in train_data:', X_train_tensor.shape)
    print('Number of instances in test_data:', X_test_tensor.shape)

    dataset = dict()
    dataset['train_input'] = X_train_tensor
    dataset['train_label'] = y_train_tensor
    dataset['test_input'] = X_test_tensor
    dataset['test_label'] = y_test_tensor

    mono_vars = {i:1 if i==df_train.columns.get_loc('trestbps') or i== df_train.columns.get_loc('chol') else 0 for i in range(df_train.shape[1]-1)}
    classification = True

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, dataset, mono_vars, classification


def preprocess(df):
    X = df.drop(columns=['target']).values
    Y = df['target'].values
    X = torch.tensor(X.astype(float), dtype=torch.float32)
    Y = torch.tensor(Y.astype(float), dtype=torch.float32).view(-1, 1)
    X = (X - X.mean(0)) / X.std(0)
    return X, Y