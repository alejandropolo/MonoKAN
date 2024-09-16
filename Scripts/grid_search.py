
import torch
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, accuracy_score#, root_mean_squared_error
## Import every file in ../Scripts/
from itertools import product
import os
import sys
sys.path.append('../Scripts')
sys.path.append('../loaders')
import KANLayer
import KAN
import spline
import utils
import importlib
import CustomELU
# import grid_search
importlib.reload(KANLayer)
importlib.reload(KAN)
importlib.reload(spline)
importlib.reload(utils)
importlib.reload(CustomELU)
# importlib.reload(grid_search)

from KANLayer import KANLayer
from KAN import KAN
from CustomELU import CustomELU
# from grid_search import grid_search
from auto_mpg_loader import load_data_auto
from heart_disease_loader import load_data_heart
from compas_loader import load_data_compas
from loan_loader import load_data_loan
from blog_loader import load_data_blog

import torch.nn as nn
import torch

def normalize(tensor, min_val=None, max_val=None, mode='normalize'):
    if mode == 'normalize':
        if min_val is None:
            min_val, _ = torch.min(tensor, dim=0)
        if max_val is None:
            max_val, _ = torch.max(tensor, dim=0)
        normalized_tensor = (tensor - min_val) / (max_val - min_val)
        return normalized_tensor, min_val, max_val
    elif mode == 'unnormalize':
        if min_val is None or max_val is None:
            raise ValueError("For unnormalization, min_val and max_val must be provided")
        unnormalized_tensor = tensor * (max_val - min_val) + min_val
        return unnormalized_tensor
    else:
        raise ValueError("Not a valid mode. Choose either 'normalize' or 'unnormalize'")

def grid_search(dataset_name,neurons_list, k_list, lambda_l1_list, lambda_entropy_list, seeds,
                grids, noise_scale, noise_scale_base, grid_eps, symbolic_enabled, opt, patience,
                hermite, normalize, steps, lamb, small_reg_factor, update_grid, lr_list, monotonic, base_function,batch_size):
    
    ## Convert dataset_name to lowercase
    dataset_name = dataset_name.lower()
    ## LOAD DATA
    if dataset_name == 'auto':
        X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, dataset,mono_vars,classification = load_data_auto('../data/Preprocessed_Data/Autompg/auto-mpg.csv')
    elif dataset_name == 'heart':
        X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, dataset,mono_vars,classification = load_data_heart('../data/Preprocessed_Data/heart/')
    elif dataset_name == 'compas':
        X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, dataset,mono_vars,classification = load_data_compas(file_path='../data/Preprocessed_Data/Compas/compas_scores_two_years.csv',get_categorical_info=False)
    elif dataset_name == 'loan':
        X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, dataset,mono_vars,classification = load_data_loan(file_path='../data/Preprocessed_Data/Loan/preprocessed.csv',ridged=True,get_categorical_info=False)
    elif dataset_name == 'blog':
        X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, dataset,mono_vars,classification = load_data_blog(file_path='../data/Preprocessed_Data/Blog/',ridged=True,get_categorical_info=False)
    else:
        raise ValueError('Dataset not found, options are: auto, heart, compas, loan, blog')

    ## For each in the list of neurons_list add at the beginning the number of variables
    n_var = X_train_tensor.shape[1]
    neurons_list = [[n_var] + neurons for neurons in neurons_list]

    ini = X_train_tensor.min().item()
    fin = X_train_tensor.max().item()
    grid_range = [ini, fin]  # Define ini and fin

    ## Print number of instances in train_data and test_data
    print('Number of instances in train_data:', len(X_train_tensor))
    print('Number of instances in test_data:', len(X_test_tensor))

    # Initialize a list to store the results
    results = []

    # Perform grid search
    for seed, neurons, k, grid, lambda_l1, lambda_entropy, lr in product(seeds,neurons_list, k_list, grids, lambda_l1_list, lambda_entropy_list, lr_list):
        print("Training model with parameters: seed={}, neurons={}, k={}, grid={}, lambda_l1={}, lambda_entropy={}, lr={}".format(seed, neurons, k, grid, lambda_l1, lambda_entropy,lr))
        # Initialize the model with current parameters
        model = KAN(width=neurons, grid=grid, grid_range=grid_range, k=k, 
                    noise_scale=noise_scale, noise_scale_base=noise_scale_base, seed=seed, hermite=hermite,
                    grid_eps=grid_eps, base_fun=base_function, symbolic_enabled=symbolic_enabled,classification=classification)

        if hermite:
            model.apply_constraints_hermite(mono_vars)
        
        num_parameters = utils.count_parameters(model)
        print(f"The model has {num_parameters} parameters.")

        if classification:
            print('Classification')
            loss_fn = nn.BCELoss()
            early_stopping_metric = 'accuracy'
            # early_stopping_metric = 'loss'
        else:
            print('Regression')
            early_stopping_metric = 'loss'
            loss_fn = nn.MSELoss()

        # Train the model
        model.train(dataset, opt=opt, steps=steps, lamb=lamb, lamb_l1=lambda_l1, lamb_entropy=lambda_entropy, 
                    small_reg_factor=small_reg_factor, update_grid=update_grid, lr=lr,early_stopping_metric=early_stopping_metric, 
                    monotonic=monotonic, monotonic_vars=mono_vars, patience=patience,loss_fn=loss_fn,batch=batch_size)
        if not classification: 
            y_pred = model.forward(X_train_tensor)
            ## Unnormalize y_train_tensor and y_pred
            train_error = mean_squared_error(y_pred.detach().numpy(),y_train_tensor.detach().numpy())

            y_pred = model.forward(X_test_tensor) 
            ## Unnormalize y_test_tensor and y_pred
            test_error = mean_squared_error(y_pred.detach().numpy(),y_test_tensor.detach().numpy())  
        else:
            y_pred = model.forward(X_train_tensor)
            train_error = accuracy_score(y_train_tensor.detach().numpy(),y_pred.detach().numpy().round())

            y_pred = model.forward(X_test_tensor) 
            test_error = accuracy_score(y_test_tensor.detach().numpy(),y_pred.detach().numpy().round())

        model.plot_history()
        # Store the result
        results.append({
            "neurons": neurons,
            "k": k,
            "lamb": lamb,
            "noise_scale":noise_scale,
            "noise_scale_base":noise_scale_base,
            "lambda_l1": lambda_l1,
            "lambda_entropy": lambda_entropy,
            "grid": grid,
            "seed": seed,
            "opt": opt,
            "patience": patience,
            "hermite": hermite,
            "normalize": normalize,
            "lr": lr,
            "bath_size":batch_size,
            "train": train_error,
            "test": test_error
        })

        # Convert results to a DataFrame
        results_df = pd.DataFrame(results)

        # Save the DataFrame to a CSV file
        results_df.to_csv("grid_search_results.csv", index=False)

    print("Grid search completed and results saved to grid_search_results.csv")


if __name__ == "__main__":
    grid_search()