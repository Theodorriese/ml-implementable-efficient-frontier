import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
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


def ols_fit(data, feat):
    """
    Perform OLS regression and make predictions.

    Parameters:
        data (dict): Dictionary containing train_full and test datasets.
        feat (list): List of feature names to include in the regression.

    Returns:
        dict: Dictionary containing the fitted model and predictions.
    """
    # Fit OLS model
    model = LinearRegression()
    X_train = data["train_full"][feat].values
    y_train = data["train_full"]["ret_pred"].values
    model.fit(X_train, y_train)

    # Make predictions on the test set
    X_test = data["test"][feat].values
    data["test"]["pred"] = model.predict(X_test)

    # Output results
    return {
        "fit": model,
        "pred": data["test"][["id", "eom", "eom_pred_last", "pred"]]
    }


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
    # Initialize progress tracker
    total_models = len(g_vec) * len(p_vec) * len(l_vec)
    current_model = 0

    np.random.seed(seed)
    val_errors = []
    optimal_rff_train = None  # Placeholder for the optimal RFF train set

    for g_idx, g in enumerate(g_vec):
        print(f"Processing g: {g:.2f} ({g_idx + 1} out of {len(g_vec)})")

        # Create random features in training and validation sets
        rff_train = rff(data['train'][feat].values, p=max(p_vec), g=g)
        rff_val = rff(data['val'][feat].values, W=rff_train['W'])

        for p_idx, p in enumerate(p_vec):
            print(f"--> Processing p: {p} ({p_idx + 1} out of {len(p_vec)})")

            # Prepare features
            X_train = (p ** -0.5) * np.hstack([rff_train['X_cos'][:, :p // 2], rff_train['X_sin'][:, :p // 2]])
            X_val = (p ** -0.5) * np.hstack([rff_val['X_cos'][:, :p // 2], rff_val['X_sin'][:, :p // 2]])
            y_train = data['train']['ret_pred'].values
            y_val = data['val']['ret_pred'].values

            for l_idx, l in enumerate(l_vec):
                # Update progress tracker
                current_model += 1
                if current_model % 10 == 0 or current_model == total_models:
                    print(f"Progress: {current_model}/{total_models} models completed...")

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

    print(f"X_test shape: {X_test.shape}")
    print(f"data['test'] length: {len(data['test'])}")
    print(data['test'].head())

    data['test']['pred'] = final_model.predict(X_test)

    # Plot validation errors
    # val_errors_df['log_lambda'] = np.log10(val_errors_df['lambda'])
    # for g in val_errors_df['g'].unique():
    #     subset = val_errors_df[val_errors_df['g'] == g]
    #     plt.figure(figsize=(8, 6))
    #     for p in subset['p'].unique():
    #         p_subset = subset[subset['p'] == p]
    #         plt.plot(p_subset['log_lambda'], p_subset['mse'], label=f"p={p}")
    #     plt.title(f"Validation MSE for g={g:.2f}")
    #     plt.xlabel("log10(lambda)")
    #     plt.ylabel("Mean Squared Error")
    #     plt.legend()
    #     plt.show()

    return {
        "fit": final_model,
        "pred": data['test'][['id', 'eom', 'eom_pred_last', 'pred']],
        "hp_search": val_errors_df,
        "W": optimal_rff_train['W'][:, :opt_p // 2],
        "opt_hps": opt_hps
    }


# Not used?
def ridge_hp_search(data, feat, vol_scale, lambdas):
    """
    Perform Ridge regression hyperparameter search and prediction.

    Parameters:
        data (dict): A dictionary containing train, val, and test datasets.
        feat (list): List of feature names to use in the model.
        vol_scale (bool): Whether to scale predictions by volatility.
        lambdas (list): List of lambda values (regularization strengths) to search over.

    Returns:
        dict: A dictionary containing the fitted model, hyperparameter search results,
              optimal lambda, predictions, and feature importance.
    """
    # Ridge regression with cross-validation
    X_train = data['train'][feat].values
    y_train = data['train']['ret_pred'].values
    X_val = data['val'][feat].values
    y_val = data['val']['ret_pred'].values

    # Fit Ridge regression across the range of lambdas
    ridge_cv = RidgeCV(alphas=lambdas, scoring='neg_mean_squared_error', store_cv_values=True)
    ridge_cv.fit(X_train, y_train)

    # Get predictions for validation set
    y_pred_val = ridge_cv.predict(X_val)

    # Hyperparameter search results
    mse_scores = -ridge_cv.cv_values_.mean(axis=0)  # Convert negative MSE to MSE
    lambda_search = pd.DataFrame({
        'lambda': lambdas,
        'mse': mse_scores
    })

    # Plot MSE vs. log(lambda)
    plt.figure(figsize=(8, 5))
    plt.plot(np.log10(lambda_search['lambda']), lambda_search['mse'], marker='o')
    plt.xlabel('log(lambda)')
    plt.ylabel('Mean Squared Error')
    plt.title('Ridge Hyperparameter Search')
    plt.grid(True)
    plt.show()

    # Optimal lambda
    lambda_opt = ridge_cv.alpha_

    # Predictions on validation set
    pred_val_op = data['val'][['id', 'eom', 'eom_pred_last']].copy()
    pred_val_op['pred'] = y_pred_val

    # Re-fit Ridge model to the full training data
    X_train_full = data['train_full'][feat].values
    y_train_full = data['train_full']['ret_pred'].values
    ridge_final = Ridge(alpha=lambda_opt)
    ridge_final.fit(X_train_full, y_train_full)

    # Predictions on test set
    X_test = data['test'][feat].values
    y_pred_test = ridge_final.predict(X_test)

    pred_op = data['test'][['id', 'eom', 'eom_pred_last']].copy()
    pred_op['pred'] = y_pred_test

    # Feature importance
    feat_imp = pd.DataFrame({
        'feature': feat,
        'estimate': ridge_final.coef_
    }).sort_values(by='estimate', key=abs, ascending=False).reset_index(drop=True)
    feat_imp['imp'] = feat_imp.index + 1

    # If vol_scale is True, adjust predictions by volatility
    if vol_scale:
        pred_val_op['pred_vol'] = pred_val_op['pred']
        pred_val_op['pred'] *= data['val']['rvol_m']
        pred_op['pred_vol'] = pred_op['pred']
        pred_op['pred'] *= data['test']['rvol_m']

    # Output results
    return {
        'fit': ridge_final,
        'hp_search': lambda_search,
        'l_opt': lambda_opt,
        'pred': pred_op,
        'pred_val': pred_val_op,
        'feat_imp': feat_imp
    }


# Not used?
def fit_xgb(train, val, params, iter, es, cores, seed):
    """
    Fit an XGBoost model using the given training and validation datasets.

    Parameters:
        train (xgb.DMatrix): Training dataset in XGBoost format.
        val (xgb.DMatrix): Validation dataset in XGBoost format.
        params (dict): Dictionary of hyperparameters for the XGBoost model.
        iter (int): Maximum number of boosting iterations.
        es (Optional[int]): Early stopping rounds. If None, no early stopping is applied.
        cores (int): Number of threads to use for training.
        seed (int): Random seed for reproducibility.

    Returns:
        xgb.Booster: Fitted XGBoost model.
    """
    # Define full parameter list
    params_all = {
        'objective': 'reg:squarederror',  # Regression with squared error loss
        'base_score': 0,                 # Initial prediction score
        'eval_metric': 'rmse',           # Root Mean Squared Error as evaluation metric
        'booster': 'gbtree',             # Use gradient boosted trees
        'max_depth': params['tree_depth'],
        'eta': params['learn_rate'],
        'gamma': params['loss_reduction'],
        'subsample': params['sample_size'],  # Row subsampling
        'colsample_bytree': params['mtry'],  # Column subsampling
        'min_child_weight': params['min_n'],
        'lambda': params['penalty'],       # L2 regularization term
        'seed': seed,                      # Seed for reproducibility
        'nthread': cores                   # Number of threads
    }

    # Watchlist for monitoring training and validation
    watchlist = [(train, 'train'), (val, 'val')]

    # Prepare arguments for xgb.train
    train_args = {
        'params': params_all,
        'dtrain': train,
        'num_boost_round': iter,
        'evals': watchlist,
        'verbose_eval': False,  # Suppress verbose output
    }
    if es is not None:
        train_args['early_stopping_rounds'] = es

    # Fit the model
    model = xgb.train(**train_args)

    return model


# Not used?
def xgb_hp_search(data, feat, vol_scale, hp_grid, iter_, es, cores, seed):
    """
    Perform hyperparameter search for XGBoost.

    Parameters:
        data (dict): Dictionary containing training, validation, and test datasets.
                     Keys should be 'train', 'val', and 'test'.
        feat (list): List of feature names.
        vol_scale (bool): Whether to scale predictions by volatility.
        hp_grid (pd.DataFrame): Hyperparameter grid for search.
        iter_ (int): Maximum number of boosting iterations.
        es (int): Early stopping rounds.
        cores (int): Number of threads for training.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: Results including the fitted model, best hyperparameters, and predictions.
    """
    # Prepare data for XGBoost
    train = xgb.DMatrix(data=data['train'][feat].values, label=data['train']['ret_pred'].values)
    val = xgb.DMatrix(data=data['val'][feat].values, label=data['val']['ret_pred'].values)
    test = xgb.DMatrix(data=data['test'][feat].values)

    # Hyperparameter search
    xgb_search = []
    for j in range(len(hp_grid)):
        print(f"Hyperparameter Set: {j + 1} / {len(hp_grid)}")
        params = hp_grid.iloc[j].to_dict()
        xgb_fit = fit_xgb(train=train, val=val, params=params, iter=iter_, es=es, cores=cores, seed=seed)
        xgb_search.append({
            "hp_no": j + 1,
            "val_rmse": xgb_fit.best_score,
            "best_iter": xgb_fit.best_iteration
        })

    # Convert search results to DataFrame
    xgb_search = pd.DataFrame(xgb_search)
    print(xgb_search)

    # Plot RMSE vs hyperparameter index
    plt.figure(figsize=(10, 6))
    plt.plot(xgb_search["hp_no"], xgb_search["val_rmse"], marker='o')
    plt.xlabel("Hyperparameter Index")
    plt.ylabel("Validation RMSE")
    plt.title("Validation RMSE vs Hyperparameter Index")
    plt.show()

    # Find the best hyperparameters
    best_hp_no = xgb_search.loc[xgb_search["val_rmse"].idxmin(), "hp_no"]
    best_hp = hp_grid.iloc[best_hp_no - 1].to_dict()
    best_iter = xgb_search.loc[xgb_search["hp_no"] == best_hp_no, "best_iter"].values[0]

    print(f"Best Hyperparameters: {best_hp}")
    print(f"Optimal Iterations: {best_iter}")

    # Re-fit on full training data
    train_full = xgb.DMatrix(data=data['train_full'][feat].values, label=data['train_full']['ret_pred'].values)
    final_fit = fit_xgb(train=train_full, val=train_full, params=best_hp, iter=best_iter, es=None, cores=cores, seed=seed)

    # Feature importance (SHAP contributions)
    shap_contrib = final_fit.predict(train_full, pred_contribs=True)
    global_imp = np.abs(shap_contrib).mean(axis=0)
    global_imp_df = pd.DataFrame({
        "char": feat + ["BIAS"],
        "value": global_imp
    }).query("char != 'BIAS'").sort_values(by="value", ascending=False).reset_index(drop=True)
    global_imp_df["rank"] = global_imp_df["value"].rank(ascending=False)

    # Plot global feature importance
    plt.figure(figsize=(10, 6))
    top_features = global_imp_df.nlargest(20, "value")
    plt.barh(top_features["char"], top_features["value"])
    plt.xlabel("Global Feature Importance")
    plt.title("Top 20 Feature Importance")
    plt.gca().invert_yaxis()
    plt.show()

    # Predictions on test data
    pred = final_fit.predict(test)
    if vol_scale:
        data['test']['pred_vol'] = pred
        data['test']['pred'] = pred * data['test']['rvol_m']
        pred_op = data['test'][["id", "eom", "pred_vol", "pred"]]
    else:
        data['test']['pred'] = pred
        pred_op = data['test'][["id", "eom", "pred"]]

    return {
        "fit": final_fit,
        "best_hp": best_hp,
        "best_iter": best_iter,
        "hp_search": xgb_search,
        "pred": pred_op,
        "feat_imp": global_imp_df
    }
