## this has been extracted from the original code (Nolte et al, Expressive Monotonic Neural Networks, 2023)

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

def load_data_heart(file_path):
    df_train = pd.read_csv(file_path + "heart_train.csv", index_col=0)
    df_test = pd.read_csv(file_path + "heart_test.csv", index_col=0)

    # Split df_train into training and validation sets
    df_train_split, df_val_split = train_test_split(df_train, test_size=0.2, random_state=0)

    # Preprocess training, validation, and test sets
    X_train_tensor, y_train_tensor = preprocess(df_train_split)
    X_val_tensor, y_val_tensor = preprocess(df_val_split)
    X_test_tensor, y_test_tensor = preprocess(df_test)

    # Print number of instances in train_data, val_data, and test_data
    print('Number of instances in train_data:', X_train_tensor.shape)
    print('Number of instances in val_data:', X_val_tensor.shape)
    print('Number of instances in test_data:', X_test_tensor.shape)

    # Create dataset dictionary
    dataset = dict()
    dataset['train_input'] = X_train_tensor
    dataset['train_label'] = y_train_tensor
    dataset['val_input'] = X_val_tensor
    dataset['val_label'] = y_val_tensor
    dataset['test_input'] = X_test_tensor
    dataset['test_label'] = y_test_tensor

    mono_vars = {i: 1 if i == df_train.columns.get_loc('trestbps') or i == df_train.columns.get_loc('chol') else 0 for i in range(df_train.shape[1] - 1)}
    classification = True

    return X_train_tensor, X_test_tensor, X_val_tensor, y_train_tensor, y_test_tensor, y_val_tensor, dataset, mono_vars, classification

def preprocess(df):
    X = df.drop(columns=['target']).values
    Y = df['target'].values
    X = torch.tensor(X.astype(float), dtype=torch.float32)
    Y = torch.tensor(Y.astype(float), dtype=torch.float32).view(-1, 1)
    X = (X - X.mean(0)) / X.std(0)
    return X, Y