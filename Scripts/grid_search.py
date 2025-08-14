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
import time

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

def grid_search(dataset_name, neurons_list, k_list, lambda_l1_list, lambda_entropy_list, seeds,
                grids, noise_scale, noise_scale_base, grid_eps, symbolic_enabled, opt, patience,
                hermite, normalize, steps, lamb, small_reg_factor, update_grid, lr_list, monotonic, base_function, batch_size, device):

    ## Convert dataset_name to lowercase
    dataset_name = dataset_name.lower()
    ## LOAD DATA
    if dataset_name == 'auto':
        X_train_tensor, X_test_tensor, X_val_tensor, y_train_tensor, y_test_tensor, y_val_tensor, dataset, mono_vars, classification = load_data_auto('../data/Preprocessed_Data/Autompg/auto-mpg.csv')
    elif dataset_name == 'heart':
        X_train_tensor, X_test_tensor, X_val_tensor, y_train_tensor, y_test_tensor, y_val_tensor, dataset, mono_vars, classification = load_data_heart('../data/Preprocessed_Data/heart/')
    elif dataset_name == 'compas':
        X_train_tensor, X_test_tensor, X_val_tensor, y_train_tensor, y_test_tensor, y_val_tensor, dataset, mono_vars, classification = load_data_compas(file_path='../data/Preprocessed_Data/Compas/compas_scores_two_years.csv', get_categorical_info=False)
    elif dataset_name == 'loan':
        X_train_tensor, X_test_tensor, X_val_tensor, y_train_tensor, y_test_tensor, y_val_tensor, dataset, mono_vars, classification = load_data_loan(file_path='../data/Preprocessed_Data/Loan/preprocessed.csv', ridged=True, get_categorical_info=False)
    elif dataset_name == 'blog':
        X_train_tensor, X_test_tensor, X_val_tensor, y_train_tensor, y_test_tensor, y_val_tensor, dataset, mono_vars, classification = load_data_blog(file_path='../data/Preprocessed_Data/Blog/', ridged=True, get_categorical_info=False)
    else:
        raise ValueError('Dataset not found, options are: auto, heart, compas, loan, blog')

    # Move dataset tensors to the specified device
    X_train_tensor = X_train_tensor.to(device)
    X_test_tensor = X_test_tensor.to(device)
    X_val_tensor = X_val_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)
    y_test_tensor = y_test_tensor.to(device)
    y_val_tensor = y_val_tensor.to(device)

    # Update dataset on-device
    dataset = {
        'train_input': X_train_tensor,
        'test_input': X_test_tensor,
        'val_input': X_val_tensor,
        'train_label': y_train_tensor,
        'test_label': y_test_tensor,
        'val_label': y_val_tensor,
    }

    ## For each in the list of neurons_list add at the beginning the number of variables
    n_var = X_train_tensor.shape[1]
    neurons_list = [[n_var] + neurons for neurons in neurons_list]

    ini = X_train_tensor.min().item()
    fin = X_train_tensor.max().item()
    grid_range = [ini, fin]  # Define ini and fin

    ## Print number of instances in train_data, test_data, and val_data
    print('Number of instances in train_data:', len(X_train_tensor))
    print('Number of instances in test_data:', len(X_test_tensor))
    print('Number of instances in val_data:', len(X_val_tensor))

    # Create directory to save models and results
    model_dir = f"../models/{dataset_name}/"
    results_dir = "./exp_results/"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Initialize a list to store the results
    results = []

    # Perform grid search
    for seed, neurons, k, grid, lambda_l1, lambda_entropy, lr in product(seeds, neurons_list, k_list, grids, lambda_l1_list, lambda_entropy_list, lr_list):
        print(f"Training model with: seed={seed}, neurons={neurons}, k={k}, grid={grid}, lambda_l1={lambda_l1}, lambda_entropy={lambda_entropy}, lr={lr}")
        # Initialize the model with current parameters
        model = KAN(width=neurons, grid=grid, grid_range=grid_range, k=k, 
                    noise_scale=noise_scale, noise_scale_base=noise_scale_base, seed=seed, hermite=hermite,
                    grid_eps=grid_eps, base_fun=base_function, symbolic_enabled=symbolic_enabled, classification=classification, device=device)

        if hermite:
            model.apply_constraints_hermite(mono_vars)
        
        loss_fn = nn.BCELoss() if classification else nn.MSELoss()
        early_stopping_metric = 'accuracy' if classification else 'loss'

        start_time = time.time()

        # Train the model
        model.train(dataset, opt=opt, steps=steps, lamb=lamb, lamb_l1=lambda_l1, lamb_entropy=lambda_entropy, 
                    small_reg_factor=small_reg_factor, update_grid=update_grid, lr=lr,early_stopping_metric=early_stopping_metric, 
                    monotonic=monotonic, monotonic_vars=mono_vars, patience=patience,loss_fn=loss_fn,batch=batch_size, device=device)

        num_parameters = utils.count_parameters(model)
        print(f"The model has {num_parameters} parameters.")

        training_time = time.time() - start_time

        print(f"Training time: {training_time:.2f} seconds")
        print(f"Model device: {next(model.parameters()).device}")

        with torch.no_grad():
            if not classification:
                y_pred_train = model.forward(X_train_tensor)
                train_error = mean_squared_error(y_train_tensor.cpu(), y_pred_train.cpu())

                y_pred_val = model.forward(X_val_tensor)
                val_error = mean_squared_error(y_val_tensor.cpu(), y_pred_val.cpu())

                y_pred_test = model.forward(X_test_tensor)
                test_error = mean_squared_error(y_test_tensor.cpu(), y_pred_test.cpu())
            else:
                y_pred_train = model.forward(X_train_tensor)
                train_error = accuracy_score(y_train_tensor.cpu().numpy(), y_pred_train.cpu().numpy().round())
                print("Train Accuracy",train_error)

                y_pred_val = model.forward(X_val_tensor)
                val_error = accuracy_score(y_val_tensor.cpu().numpy(), y_pred_val.cpu().numpy().round())
                print("Val Accuracy",val_error)

                y_pred_test = model.forward(X_test_tensor)
                test_error = accuracy_score(y_test_tensor.cpu().numpy(), y_pred_test.cpu().numpy().round())
                print("Test Accuracy",test_error)

        # Save the model for the current seed
        model_path = os.path.join(model_dir, f"model_seed_{seed}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

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
            "batch_size": batch_size,
            "train": train_error,
            "val": val_error,
            "test": test_error,
            "training_time": training_time,
            "number_of_parameters": num_parameters
        })

        # Convert results to a DataFrame
        results_df = pd.DataFrame(results)

        # Save the DataFrame to a CSV file
        results_csv_path = os.path.join(results_dir, f"{dataset_name}_grid_search_results.csv")
        results_df.to_csv(results_csv_path, index=False)

    print(f"Grid search completed and results saved to {results_csv_path}")


if __name__ == "__main__":
    grid_search()