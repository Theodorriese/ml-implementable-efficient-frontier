import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import xgboost as xgb


def data_split(data, val_end, val_years, train_start, train_lookback, retrain_lookback, test_inc, test_end):
    """
    Split data into train, validation, and test sets.

    Parameters:
        data (pd.DataFrame): Input data with 'eom' and 'eom_pred_last' columns.
        val_end (pd.Timestamp): End date for the validation period.
        val_years (int): Number of years for the validation period.
        train_start (pd.Timestamp): Start date for training data.
        train_lookback (int): Number of years for the initial training period.
        retrain_lookback (int): Number of years for retraining the full model.
        test_inc (int): Number of years for the test increment period.
        test_end (pd.Timestamp): End date for the test period.

    Returns:
        dict: Dictionary containing train, validation, test, and train_full datasets.
    """
    train_end = val_end - pd.DateOffset(years=val_years)
    train_start = max(train_start, train_end - pd.DateOffset(years=train_lookback))

    print(f"Train Start: {train_start}, Train End: {train_end}")

    # Ensure all date variables are in the correct format
    train_start = pd.Timestamp(train_start)
    train_end = pd.Timestamp(train_end)
    val_end = pd.Timestamp(val_end)
    test_end = pd.Timestamp(test_end)

    split_data = {
        "val": data[(data["eom"] >= train_end) & (data["eom_pred_last"] <= val_end)],

        "train": data[(data["eom"] >= train_start) & (data["eom_pred_last"] <= train_end)],

        "train_full": data[(data["eom"] >= (val_end - pd.DateOffset(years=retrain_lookback))) &
                           (data["eom_pred_last"] <= val_end)],

        "test": data[(data["eom"] >= val_end) &
                     (data["eom"] < min(val_end + pd.DateOffset(years=test_inc), test_end))]
    }

    return split_data


def rff(X, p=None, g=None, W=None):
    """
    Generate random Fourier features for input data X.

    Parameters:
        X (np.ndarray): Input data matrix.
        p (int): Number of random Fourier features to generate (must be divisible by 2).
        g (float): Gaussian kernel width.
        W (np.ndarray, optional): Precomputed random weights.

    Returns:
        dict: Dictionary containing random weights (W), cosine features, and sine features.
    """
    if W is None:
        np.random.seed(0)  # Ensure reproducibility
        k = X.shape[1]
        cov_matrix = g * np.eye(k)
        W = np.random.multivariate_normal(mean=np.zeros(k), cov=cov_matrix, size=p // 2).T
    X_new = X @ W
    return {
        "W": W,
        "X_cos": np.cos(X_new),
        "X_sin": np.sin(X_new)
    }


def rff_hp_search(data, feat, p_vec, g_vec, l_vec, seed):
    """
    Perform hyperparameter search for Random Fourier Features.

    Parameters:
        data (dict): Dictionary containing train, val, train_full, and test datasets.
        feat (list): List of feature names.
        p_vec (list): List of numbers of random Fourier features to evaluate.
        g_vec (list): List of Gaussian kernel widths.
        l_vec (list): List of ridge regularization parameters (lambdas).
        seed (int): Random seed for reproducibility.

    Returns:
        dict: Dictionary containing the fitted model, predictions, hyperparameter search results, weights, and optimal hyperparameters.
    """
    np.random.seed(seed)
    val_errors = []
    optimal_rff_train = None  # Placeholder for the optimal RFF train set

    for g_idx, g in enumerate(g_vec):

        # Create random features in training and validation sets
        rff_train = rff(data['train'][feat].values, p=max(p_vec), g=g)
        rff_val = rff(data['val'][feat].values, W=rff_train['W'])

        for p_idx, p in enumerate(p_vec):

            # Prepare features
            X_train = (p ** -0.5) * np.hstack([rff_train['X_cos'][:, :p // 2], rff_train['X_sin'][:, :p // 2]])
            X_val = (p ** -0.5) * np.hstack([rff_val['X_cos'][:, :p // 2], rff_val['X_sin'][:, :p // 2]])
            y_train = data['train']['ret_pred'].values
            y_val = data['val']['ret_pred'].values

            for l_idx, l in enumerate(l_vec):

                # Fit Ridge regression
                model = Ridge(alpha=l, fit_intercept=True)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                mse = mean_squared_error(y_val, y_pred)

                val_errors.append({
                    "lambda": l,
                    "mse": mse,
                    "p": p,
                    "g": g
                })

                # Save the rff_train corresponding to the best g and p
                if mse == min(val_errors, key=lambda x: x["mse"])["mse"]:
                    optimal_rff_train = rff_train

    val_errors_df = pd.DataFrame(val_errors)
    opt_hps = val_errors_df.loc[val_errors_df['mse'].idxmin()]
    opt_g = opt_hps['g']
    opt_p = int(opt_hps['p'])
    opt_l = opt_hps['lambda']

    print(f"Optimal parameters: p={opt_p}, g={opt_g}, lambda={opt_l}")

    # Refit on train_full with optimal parameters
    rff_train_full = rff(data['train_full'][feat].values, W=optimal_rff_train['W'][:, :opt_p // 2])
    X_train_full = (opt_p ** -0.5) * np.hstack([rff_train_full['X_cos'], rff_train_full['X_sin']])
    y_train_full = data['train_full']['ret_pred'].values

    final_model = Ridge(alpha=opt_l, fit_intercept=True)
    final_model.fit(X_train_full, y_train_full)

    # Predict on test data
    rff_test = rff(data['test'][feat].values, W=optimal_rff_train['W'][:, :opt_p // 2])
    X_test = (opt_p ** -0.5) * np.hstack([rff_test['X_cos'], rff_test['X_sin']])

    if X_test.shape[0] > 0:
        data['test']['pred'] = final_model.predict(X_test)
    else:
        data['test']['pred'] = np.nan

    return {
        "fit": final_model,
        "pred": data['test'][['id', 'eom', 'eom_pred_last', 'pred']],
        "hp_search": val_errors_df,
        "W": optimal_rff_train['W'][:, :opt_p // 2],
        "opt_hps": opt_hps
    }
