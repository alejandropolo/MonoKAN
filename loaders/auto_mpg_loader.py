## this has been extracted from the original code (Nolte et al, Expressive Monotonic Neural Networks, 2023)

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

def load_data_auto(file_path):
    df = pd.read_csv(file_path)
    df = df[df.horsepower != "?"]
    
    X = df.drop(columns=['mpg', 'car name']).values
    Y = df['mpg'].values
    
    X = torch.tensor(X.astype(float), dtype=torch.float32)
    Y = torch.tensor(Y.astype(float), dtype=torch.float32).view(-1, 1)
    
    # X = (X - X.mean(0)) / X.std(0)
    
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = train_test_split(X, Y, test_size=0.2, random_state=0)

    X_tr_mean = X_train_tensor.mean(0)
    X_tr_std = X_train_tensor.std(0)
    X_train_tensor = (X_train_tensor-X_tr_mean)/X_tr_std
    X_test_tensor = (X_test_tensor-X_tr_mean)/X_tr_std
    
    dataset = {
        'train_input': X_train_tensor,
        'train_label': y_train_tensor,
        'test_input': X_test_tensor,
        'test_label': y_test_tensor
    }

    mono_vars = {0:0,1:-1,2:-1,3:-1,4:0,5:0,6:0} ## AUTOMPG
    classification = False
    
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, dataset, mono_vars, classification
    