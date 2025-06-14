import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm
from scipy.linalg import sqrtm, pinv
from itertools import product
from collections import defaultdict
from joblib import Parallel, delayed
from i1_Main import settings
from a_general_functions import create_cov, create_lambda, sigma_gam_adj, initial_weights_new, pf_ts_fun
from a_return_prediction_functions import rff
from multiprocessing import cpu_count
import multiprocessing
import matplotlib.pyplot as plt
from datetime import datetime


def m_func(w, mu, rf, sigma_gam, gam, lambda_mat, iterations):
    """
    Computes the m matrix.

    Parameters:
        w (float): Weight scalar.
        mu (float): Expected return.
        rf (float): Risk-free rate.
        sigma_gam (ndarray or DataFrame): Covariance matrix scaled by gamma.
        gam (float): Risk aversion coefficient.
        lambda_mat (ndarray or DataFrame): Diagonal matrix of lambdas.
        iterations (int): Number of iterations.

    Returns:
        ndarray: Computed m matrix.
    """
    if isinstance(sigma_gam, pd.DataFrame):
        sigma_gam = sigma_gam.to_numpy()
    if isinstance(lambda_mat, pd.DataFrame):
        lambda_mat = lambda_mat.to_numpy()

    # Ensure lambda_mat is diagonal & avoid division by zero
    n = lambda_mat.shape[0]
    lamb_neg05 = np.diag(np.diag(lambda_mat) ** -0.5)

    g_bar = np.ones(n)
    mu_bar_vec = np.ones(n) * (1 + rf + mu)

    sigma_gr = (1 / (1 + rf + mu) ** 2) * (np.outer(mu_bar_vec, mu_bar_vec) + sigma_gam / gam)

    x = (1 / w) * lamb_neg05 @ sigma_gam @ lamb_neg05
    y = np.diag(1 + np.diag(sigma_gr))

    sigma_hat = x + np.diag(1 + g_bar)
    sigma_hat_squared = sigma_hat @ sigma_hat

    sqrt_term = sqrtm(sigma_hat_squared - 4 * np.eye(n))
    m_tilde = 0.5 * (sigma_hat - np.real(sqrt_term))

    for i in range(iterations):
        A = x + y - m_tilde * sigma_gr
        try:
            m_tilde = np.linalg.solve(A, np.eye(A.shape[0]))
            # print(f"  → Iter {i+1}: solved via `solve()`")
        except np.linalg.LinAlgError:
            print(f"Matrix is singular, using pseudo-inverse.")
            m_tilde = np.linalg.pinv(A)

    return lamb_neg05 @ m_tilde @ np.sqrt(lambda_mat)


def m_func_2(w, mu, rf, sigma_gam, gam, lambda_mat, iterations, plot_path='/work/frontier_ml/m_plots'):
    """
    This is an additional m function to be run when convergence is to be check.
    Computes the m matrix and plots convergence over iterations, saving the plot to a specified directory.

    Parameters:
        w (float): Weight scalar.
        mu (float): Expected return.
        rf (float): Risk-free rate.
        sigma_gam (ndarray or DataFrame): Covariance matrix scaled by gamma.
        gam (float): Risk aversion coefficient.
        lambda_mat (ndarray or DataFrame): Diagonal matrix of lambdas.
        iterations (int): Number of iterations.
        plot_path (str): Path to save the plot (default is '/work/frontier_ml_eu/m_plots').

    Returns:
        ndarray: Computed m matrix.
    """
    # Ensure the plot path exists, if not create it
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    if isinstance(sigma_gam, pd.DataFrame):
        sigma_gam = sigma_gam.to_numpy()
    if isinstance(lambda_mat, pd.DataFrame):
        lambda_mat = lambda_mat.to_numpy()

    n = lambda_mat.shape[0]
    lamb_neg05 = np.diag(np.diag(lambda_mat) ** -0.5)
    lamb_pos05 = np.sqrt(lambda_mat)

    mu_bar_vec = np.ones(n) * (1 + rf + mu)
    sigma_gr = (1 / (1 + rf + mu) ** 2) * (np.outer(mu_bar_vec, mu_bar_vec) + sigma_gam / gam)

    x = (1 / w) * lamb_neg05 @ sigma_gam @ lamb_neg05
    y = np.diag(1 + np.diag(sigma_gr))

    sigma_hat = x + np.diag(1 + np.ones(n))
    sigma_hat_squared = sigma_hat @ sigma_hat
    sqrt_term = sqrtm(sigma_hat_squared - 4 * np.eye(n))
    m_tilde = 0.5 * (sigma_hat - np.real(sqrt_term))

    # Track convergence
    norms = []

    for i in range(iterations):
        A = x + y - m_tilde @ sigma_gr
        try:
            m_next = np.linalg.solve(A, np.eye(n))
        except np.linalg.LinAlgError:
            print("Matrix is singular, using pseudo-inverse.")
            m_next = np.linalg.pinv(A)

        diff_norm = np.linalg.norm(m_next - m_tilde, ord='fro')
        print(diff_norm)
        norms.append(diff_norm)
        m_tilde = m_next

    # Only plot from iteration 3 onwards
    norms_to_plot = norms[2:]  # Start from iteration 3 (index 2)

    # Plot convergence and save it
    plt.figure(figsize=(8, 5))
    plt.plot(range(3, iterations + 1), norms_to_plot, marker='o')  # Adjust for the new starting point
    plt.title("Convergence of $\\tilde{m}$ Updates")
    plt.xlabel("Iteration")
    plt.ylabel("Frobenius Norm of Update")
    plt.grid(True)
    plt.tight_layout()

    # Create a unique filename using the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"convergence_plot_{timestamp}_{iterations}_iterations.png"
    plot_filepath = os.path.join(plot_path, plot_filename)

    # Save plot to the specified directory
    plt.savefig(plot_filepath)
    print(f"Plot saved to {plot_filepath}")

    plt.show()

    return lamb_neg05 @ m_tilde @ lamb_pos05


def w_fun(data, dates, w_opt, wealth):
    """
    Computes portfolio weights based on input data and desired optimal weights.

    Parameters:
        data (pd.DataFrame): Portfolio dataset containing ['id', 'eom', 'tr_ld1'].
        dates (list): List of timestamps for portfolio calculations.
        w_opt (pd.DataFrame): Optimal weights with ['id', 'eom', 'w'].
        wealth (pd.DataFrame): Wealth data with ['eom', 'mu_ld1'].

    Returns:
        pd.DataFrame: Dataframe with updated portfolio weights.
    """

    # Ensure 'id' is the same type across all DataFrames
    data["id"] = data["id"].astype(str)
    w_opt["id"] = w_opt["id"].astype(str)

    # Ensure 'eom' is in datetime format for consistency
    wealth["eom"] = pd.to_datetime(wealth["eom"])
    data["eom"] = pd.to_datetime(data["eom"])
    w_opt["eom"] = pd.to_datetime(w_opt["eom"])

    # Step 1: Map previous dates
    dates_prev = pd.DataFrame({"eom": dates, "eom_prev": pd.Series(dates).shift(1)})

    # Step 2: Initialize weights (Equivalent to R's `initial_weights_new(w_type = "vw")`)
    w = data[['id', 'eom']].drop_duplicates()  # Ensures no duplicate entries
    w = w.merge(w_opt, on=['id', 'eom'], how='left')  # Merge optimal weights
    w = w.merge(dates_prev, on='eom', how='left')  # Add previous month's reference

    # Step 3: Merge transition returns and wealth into w_opt
    w_opt = w_opt.merge(data[['id', 'eom', 'tr_ld1']], on=['id', 'eom'], how='left')
    w_opt = w_opt.merge(wealth[['eom', 'mu_ld1']], on='eom', how='left')

    # Step 4: Adjust weights based on transition returns and wealth
    w_opt['w_prev'] = w_opt['w'] * (1 + w_opt['tr_ld1']) / (1 + w_opt['mu_ld1'])
    w_opt = w_opt[['id', 'eom', 'w_prev']].rename(columns={'eom': 'eom_prev'})

    # Step 5: Merge adjusted weights back into `w`
    w = w.merge(w_opt, on=['id', 'eom_prev'], how='left')

    # Step 6: Initialize `w_start` weights
    min_eom = w['eom'].min()
    w['w_start'] = np.where(w['eom'] != min_eom, w['w_prev'],
                            0)  # Aligns exactly with R's `w[eom != min(eom), w_start := w_prev]`

    # Step 7: Handle missing values and drop unnecessary columns
    w['w_start'].fillna(0, inplace=True)
    w.drop(columns=['eom_prev', 'w_prev'], inplace=True)

    return w


def tpf_implement(data, cov_list, wealth, dates, gam):
    """
    Implements the Tangency Portfolio (Markowitz-ML) using the correct covariance matrix creation.

    Parameters:
        data (pd.DataFrame): Dataset containing portfolio information.
        cov_list (dict): A dictionary of covariance matrices keyed by date.
        wealth (pd.DataFrame): Wealth information with columns ['eom', 'wealth'].
        dates (list): List of dates for the portfolio.
        gam (float): Risk-aversion parameter.

    Returns:
        dict: A dictionary containing portfolio weights ('w') and performance ('pf').
    """
    data_rel = data[(data['valid'] == True) & (data['eom'].isin(dates))].copy()
    data_rel = data_rel[['id', 'eom', 'me', 'tr_ld1', 'pred_ld1']].sort_values(by=['id', 'eom'])

    data_split = {date: group for date, group in data_rel.groupby('eom')}

    tpf_opt = []
    for d in tqdm(dates, desc="Processing Dates"):

        data_sub = data_split.get(d, pd.DataFrame())
        if data_sub.empty:
            continue  # Skip if no data available

        sigma_data = cov_list.get(d, None)
        if sigma_data is None or "fct_cov" not in sigma_data:
            print(f"Warning: Missing 'fct_cov' for {d}")
            continue

        # Use create_cov to properly construct sigma
        ids = data_sub["id"].astype(str).unique()
        sigma = create_cov(sigma_data, ids)

        if sigma is None or sigma.shape[0] != sigma.shape[1]:
            print(f"Warning: Invalid covariance matrix for {d}")
            continue  # Skip if covariance matrix is missing or invalid

        pred_ld1 = data_sub.set_index("id")["pred_ld1"].dropna()  # Ensure pred_ld1 aligns with sigma
        sigma = pd.DataFrame(sigma)
        sigma.index = sigma.index.astype(str)
        pred_ld1.index = pred_ld1.index.astype(str)

        if not set(pred_ld1.index).issubset(sigma.index):
            print(f"Warning: Some IDs in pred_ld1 are missing from sigma for {d}")
            continue

        # Ensure pred_ld1 is not empty
        if pred_ld1.empty:
            print(f"Warning: pred_ld1 is empty for date {d}, skipping.")
            continue

        # Align pred_ld1 with sigma
        common_ids = pred_ld1.index.intersection(sigma.index)

        # If no common IDs exist, skip
        if common_ids.empty:
            print(f"Warning: No common IDs between pred_ld1 and sigma for {d}, skipping.")
            continue

        # Ensure pred_ld1 is a Pandas Series before using .loc
        if isinstance(pred_ld1, np.ndarray):
            pred_ld1 = pd.Series(pred_ld1, index=sigma.index)

        # Now, safely select only matching IDs and convert to NumPy
        pred_ld1 = pred_ld1.loc[common_ids].to_numpy()

        # Align sigma matrix to common IDs
        sigma = sigma.loc[common_ids, common_ids]

        # Compute optimal weights
        try:
            w_opt = np.dot(np.linalg.pinv(sigma), pred_ld1) / gam  # Use pseudo-inverse for robustness
        except np.linalg.LinAlgError:
            continue  # Skip if numerical issue

        data_sub["id"] = data_sub["id"].astype(str)
        data_sub = data_sub.loc[data_sub["id"].isin(sigma.index)].assign(w=w_opt)
        tpf_opt.append(data_sub[['id', 'eom', 'w']])

    if not tpf_opt:
        return {"w": pd.DataFrame(), "pf": pd.DataFrame()}  # Return empty results if no valid weights

    tpf_opt = pd.concat(tpf_opt, ignore_index=True)

    # Compute final portfolio weights and performance
    data_filtered = data[(data['eom'].isin(dates)) & (data['valid'] == True)]
    tpf_w = w_fun(data_filtered, dates, tpf_opt, wealth)
    tpf_pf = pf_ts_fun(tpf_w, data, wealth)

    tpf_pf['type'] = "Markowitz-ML"

    return {"w": tpf_w, "pf": tpf_pf}


def tpf_cf_fun(data, cf_cluster, er_models, cluster_labels, wealth,
               gamma_rel, cov_list, dates, seed, features):
    """
    Implements a counterfactual tangency portfolio by randomizing feature identifiers
    within a specified cluster and generating predicted returns using pre-fitted models.

    Parameters:
        data (pd.DataFrame): Input DataFrame containing stock-level characteristics.
        cf_cluster (str): Name of the cluster to shuffle features within.
        er_models (dict): Dictionary of expected return models per horizon.
        cluster_labels (pd.DataFrame): Mapping of characteristics to clusters.
        wealth (pd.DataFrame): Wealth time series used for return scaling.
        gamma_rel (float): Relative risk aversion parameter.
        cov_list (dict): Dictionary of covariance matrices per date.
        dates (list): List of evaluation dates.
        seed (int): Random seed for reproducibility.
        features (list): List of all feature names used in the model.

    Returns:
        pd.DataFrame: Portfolio returns and weights from the counterfactual implementation.
    """
    np.random.seed(seed)

    if cf_cluster != "bm":
        cf = data.drop(columns=[col for col in data.columns if col.startswith("pred_ld")], errors='ignore')
        cf['id_shuffle'] = cf.groupby('eom')['id'].transform(np.random.permutation)

        chars_sub = cluster_labels.query("cluster == @cf_cluster and characteristic in @features")["characteristic"]
        if chars_sub.empty:
            raise ValueError(f"Cluster '{cf_cluster}' has no matching characteristics.")

        # Shuffle characteristics
        shuffle_cols = ['id_shuffle', 'eom'] + chars_sub.tolist()
        chars_data = cf[shuffle_cols].rename(columns={'id_shuffle': 'id'})
        cf.drop(columns=chars_sub, inplace=True)
        cf = cf.merge(chars_data, on=['id', 'eom'], how='left')

        # Predict expected returns using pre-fitted models
        pred_map = {}
        for model_dict in er_models.values():
            for m_sub in model_dict.values():
                if not all(k in m_sub for k in ['fit', 'W', 'opt_hps', 'pred']):
                    continue

                sub_dates = m_sub['pred']['eom'].unique()
                cf_subset = cf[cf['eom'].isin(sub_dates)]
                if cf_subset.empty:
                    continue

                cf_x = cf_subset[features].to_numpy()
                if cf_x.size == 0:
                    continue

                rff_out = rff(cf_x, W=m_sub['W'])
                p = m_sub['opt_hps']['p']
                X = p ** -0.5 * np.hstack([rff_out['X_cos'], rff_out['X_sin']])
                pred_vals = m_sub['fit'].predict(X)

                cf.loc[cf['eom'].isin(sub_dates), 'pred_ld1'] = pred_vals

    else:
        cf = data[data['valid'] == True].copy()

    cf['pred_ld1'] = cf['pred_ld1'].fillna(0)

    # Run the tangency portfolio implementation
    op = tpf_implement(cf, cov_list, wealth, dates, gamma_rel)
    if 'pf' not in op:
        raise ValueError("tpf_implement did not return expected portfolio data.")

    op['pf']['cluster'] = cf_cluster
    return op['pf']



def factor_ml_implement(data, wealth, dates, n_pfs):
    """
    Implements a High Minus Low (HML) portfolio strategy using predicted returns.

    Parameters:
        data (pd.DataFrame): Contains portfolio data, including `id`, `eom`, `me` (market equity),
                             `pred_ld1` (predicted returns), and `valid` (filter column).
        wealth (pd.DataFrame or list): Wealth data used for weight adjustments.
        dates (list): List of dates for portfolio construction.
        n_pfs (int): Number of portfolios for ranking stocks (e.g., quintiles or deciles).

    Returns:
        dict: A dictionary with:
            - "w": Portfolio weights.
            - "pf": Portfolio performance metrics.
    """

    # Filter data to valid observations within the selected dates
    data_rel = data[(data["valid"] == True) & (data["eom"].isin(dates))].copy()
    data_rel = data_rel[['id', 'eom', 'me', 'pred_ld1']].sort_values(by=['id', 'eom'])

    # Split data by date
    data_split = {date: group.copy() for date, group in data_rel.groupby("eom")}

    # Store optimal weights
    hml_opt = []

    for d in dates:
        if d not in data_split:
            continue  # Skip missing dates

        data_sub = data_split[d].copy()  # Ensure modification safety

        # Compute percentile ranks of predicted returns
        er = data_sub['pred_ld1'].values
        er_prank = pd.Series(er).rank(pct=True).values  # Percentile rank of predicted returns

        # Assign portfolio positions
        data_sub['pos'] = np.where(er_prank >= 1 - (1 / n_pfs), 1,  # Top X% (long)
                                   np.where(er_prank <= 1 / n_pfs, -1, 0))  # Bottom X% (short)

        # Ensure no division by zero (skip if no stocks in group)
        grouped_me = data_sub.groupby("pos")["me"].transform("sum")
        data_sub["w"] = np.where(grouped_me != 0, (data_sub["me"] / grouped_me) * data_sub["pos"], 0)

        hml_opt.append(data_sub[['id', 'eom', 'w']])

    # Combine all computed weights
    hml_opt = pd.concat(hml_opt, ignore_index=True)

    # Filter the calculated weights to only include the relevant dates
    hml_opt_filtered = hml_opt[hml_opt['eom'].isin(dates)]

    # Compute actual portfolio weights
    hml_w = w_fun(data[data["eom"].isin(dates) & data["valid"]], dates, hml_opt_filtered, wealth)

    # Compute portfolio performance
    hml_pf = pf_ts_fun(hml_w, data, wealth)
    hml_pf["type"] = "Factor-ML"

    # Output dictionary
    return {"w": hml_w, "pf": hml_pf}


def ew_implement(data, wealth, dates):
    """
    Equal-weighted (1/N) portfolio implementation.

    Parameters:
        data (pd.DataFrame): DataFrame containing portfolio data with columns such as `id`, `eom` (end of month),
                             and `valid` column indicating valid rows.
        wealth (pd.DataFrame or list): Wealth data required for weight adjustments.
        dates (list): List of dates for which the portfolio is to be constructed.

    Returns:
        dict: A dictionary containing:
            - "w": DataFrame with the calculated portfolio weights.
            - "pf": DataFrame with the portfolio performance metrics.
    """

    # Filter valid stocks for relevant dates
    data_valid = data[(data['valid'] == True) & (data['eom'].isin(dates))].copy()

    # Compute equal-weighted portfolio (1/N for each stock in the month)
    data_valid['n'] = data_valid.groupby('eom')['id'].transform('count')
    ew_opt = data_valid[['id', 'eom']].copy()
    ew_opt['w'] = 1 / data_valid['n']

    # Compute actual weights
    ew_w = w_fun(data_valid, dates, ew_opt[['id', 'eom', 'w']], wealth)

    # Compute portfolio performance
    ew_pf = pf_ts_fun(ew_w, data, wealth)
    ew_pf['type'] = "1/N"

    # Return weights and portfolio performance
    return {"w": ew_w, "pf": ew_pf}


def mkt_implement(data, wealth, dates):
    """
    Market-capitalization weighted portfolio implementation.

    Parameters:
        data (pd.DataFrame): DataFrame containing portfolio data with columns such as `id`, `eom` (end of month),
                             `me` (market equity), and `valid` column indicating valid rows.
        wealth (pd.DataFrame or list): Wealth data required for weight adjustments.
        dates (list): List of dates for which the portfolio is to be constructed.

    Returns:
        dict: A dictionary containing:
            - "w": DataFrame with the calculated portfolio weights.
            - "pf": DataFrame with the portfolio performance metrics.
    """

    # Filter valid stocks within the specified dates
    data_valid = data[(data['valid'] == True) & (data['eom'].isin(dates))].copy()

    # Ensure `eom` is treated correctly (align to last trading day if necessary)
    data_valid['eom'] = pd.to_datetime(data_valid['eom'])

    # Compute market-cap weights
    mkt_opt = (
        data_valid.groupby('eom', group_keys=False)
        .apply(lambda group: group.assign(w=group['me'] / group['me'].sum()))
        .reset_index(drop=True)
    )

    # Compute actual weights
    mkt_w = w_fun(data_valid, dates, mkt_opt[['id', 'eom', 'w']], wealth)

    # Compute portfolio performance
    mkt_pf = pf_ts_fun(mkt_w, data, wealth)
    mkt_pf['type'] = "Market"

    # Return weights and portfolio performance
    return {"w": mkt_w, "pf": mkt_pf}


def rw_implement(data, wealth, dates):
    """
    Rank-weighted portfolio implementation.

    Parameters:
        data (pd.DataFrame): DataFrame containing portfolio data with columns such as `id`, `eom` (end of month),
                             `pred_ld1` (predicted returns), and `valid` column indicating valid rows.
        wealth (pd.DataFrame or list): Wealth data required for weight adjustments.
        dates (list): List of dates for which the portfolio is to be constructed.

    Returns:
        dict: A dictionary containing:
            - "w": DataFrame with the calculated portfolio weights.
            - "pf": DataFrame with the portfolio performance metrics.
    """

    # Filter valid stocks for relevant dates
    data_valid = data[(data['valid'] == True) & (data['eom'].isin(dates))].copy()

    # Compute rank-based weights
    rw_opt = []
    for d in dates:
        data_sub = data_valid[data_valid['eom'] == d][['id', 'eom', 'pred_ld1']].copy()

        # Compute ranks (equivalent to `frank()` in R)
        data_sub['rank'] = data_sub['pred_ld1'].rank(method="first").astype(float)

        # compute centered “pos” vector
        pos = data_sub['rank'] - data_sub['rank'].mean()
        # scale so that sum(abs(w)) == 2
        data_sub['w'] = pos * (2.0 / pos.abs().sum())

        rw_opt.append(data_sub[['id', 'eom', 'w']])

    rw_opt = pd.concat(rw_opt, ignore_index=True)

    # Compute actual weights
    rw_w = w_fun(data_valid, dates, rw_opt, wealth)

    # Compute portfolio performance
    rw_pf = pf_ts_fun(rw_w, data, wealth)
    rw_pf['type'] = "Rank-ML"

    # Return weights and portfolio performance
    return {"w": rw_w, "pf": rw_pf}


def mv_implement(data, cov_list, wealth, dates):
    """
    Minimum-variance portfolio implementation.

    Parameters:
        data (pd.DataFrame): Portfolio data.
        cov_list (dict): Dictionary of covariance matrices indexed by dates.
        wealth (pd.DataFrame or list): Wealth data.
        dates (list): List of portfolio construction dates.

    Returns:
        dict: A dictionary containing:
            - "w": Portfolio weights DataFrame.
            - "pf": Portfolio performance DataFrame.
    """

    # Split data by end-of-month (eom) dates
    data_split = {
        date: sub_df.copy() for date, sub_df in data[
            (data['valid'] == True) & (data['eom'].isin(dates))
            ][['id', 'eom', 'me']].groupby('eom')
    }

    mv_opt = []
    for d in dates:
        if d not in data_split:
            continue  # Skip if no data for the date

        data_sub = data_split[d]
        ids = data_sub['id'].astype(str).values

        # Retrieve covariance data and construct sigma using create_cov
        sigma_data = cov_list.get(d, None)
        if sigma_data is None:
            print(f"Warning: No covariance data found for {d}")
            continue  # Skip if no valid covariance data

        sigma = create_cov(sigma_data, ids)  # Use new covariance function

        # Ensure sigma is a DataFrame and aligned with valid IDs
        sigma = pd.DataFrame(sigma, index=ids, columns=ids)

        # Check if sigma is valid
        if sigma.shape[0] != sigma.shape[1] or sigma.shape[0] == 0:
            print(f"Warning: Invalid covariance matrix for {d}")
            continue  # Skip invalid covariance matrix

        # Use pseudo-inverse for numerical stability
        # sigma_inv = np.linalg.pinv(sigma, rcond=1e-5)  # Adjust rcond as needed
        sigma_inv = np.linalg.pinv(sigma)

        # Compute minimum variance weights
        ones = np.ones((sigma_inv.shape[0], 1))
        weights = (sigma_inv @ ones) / (ones.T @ sigma_inv @ ones)

        data_sub = data_sub[data_sub["id"].astype(str).isin(sigma.index)]  # Ensure valid IDs
        data_sub['w'] = weights.flatten()

        mv_opt.append(data_sub[['id', 'eom', 'w']])

    if not mv_opt:
        return {"w": pd.DataFrame(), "pf": pd.DataFrame()}  # Return empty results safely

    mv_opt = pd.concat(mv_opt, ignore_index=True)

    # Compute final portfolio weights and performance
    mv_w = w_fun(data[data['eom'].isin(dates) & data['valid']], dates, mv_opt, wealth)
    mv_pf = pf_ts_fun(mv_w, data, wealth)
    mv_pf['type'] = "Minimum Variance"

    return {"w": mv_w, "pf": mv_pf}


def mv_risky_fun(data, cov_list, wealth, dates, gam, u_vec):
    """
    Constructs mean-variance efficient portfolios of risky assets across different utility levels.

    Parameters:
        data (pd.DataFrame): Input DataFrame with asset-level returns, predictions, and metadata.
        cov_list (dict): Dictionary of covariance matrices by date.
        wealth (pd.DataFrame): Time series of portfolio wealth statistics.
        dates (list): List of evaluation dates for portfolio construction.
        gam (float): Not used (included for interface consistency).
        u_vec (list): List of scaled utility targets for portfolio optimization.

    Returns:
        pd.DataFrame: Combined portfolio performance for each utility level.
    """
    data_rel = data[(data['valid']) & (data['eom'].isin(dates))].copy()
    data_rel = data_rel[['id', 'eom', 'me', 'tr_ld1', 'pred_ld1', 'ret_ld1', 'lambda']].sort_values(by=['id', 'eom'])

    data_split = {date: group for date, group in data_rel.groupby('eom')}
    mv_opt_all = []

    for date in dates:
        data_sub = data_split.get(date, pd.DataFrame())
        if data_sub.empty:
            continue

        sigma_data = cov_list.get(date)
        if sigma_data is None or "fct_cov" not in sigma_data:
            continue

        ids = data_sub["id"].astype(str).unique()
        sigma = create_cov(sigma_data, ids)

        if sigma is None or sigma.shape[0] != sigma.shape[1]:
            continue

        pred_ld1 = data_sub.set_index("id")["pred_ld1"].dropna()
        sigma = pd.DataFrame(sigma, index=ids, columns=ids)
        pred_ld1.index = pred_ld1.index.astype(str)

        common_ids = pred_ld1.index.intersection(sigma.index)
        if common_ids.empty:
            continue

        pred_ld1 = pred_ld1.loc[common_ids].to_numpy()
        sigma = sigma.loc[common_ids, common_ids].to_numpy()

        sigma_inv = np.linalg.pinv(sigma)
        a = pred_ld1.T @ sigma_inv @ pred_ld1
        b = sigma_inv @ pred_ld1
        c = sigma_inv @ np.ones_like(pred_ld1)
        d = a * np.sum(c) - np.dot(b, b)

        for u in u_vec:
            w = ((np.sum(c) * u - np.sum(b)) / d) * (sigma_inv @ pred_ld1) + \
                ((a - np.sum(b) * u) / d) * (sigma_inv @ np.ones_like(pred_ld1))

            df_w = pd.DataFrame({
                'id': common_ids,
                'eom': date,
                'u': u * 12,
                'w': w
            })
            mv_opt_all.append(df_w)

    mv_opt_all_df = pd.concat(mv_opt_all, ignore_index=True)

    mv_results = []
    for u_val in mv_opt_all_df['u'].unique():
        w_sel = mv_opt_all_df[mv_opt_all_df['u'] == u_val].copy()
        w_sel = w_sel.drop(columns=['u'])

        data_subset = data_rel[data_rel['id'].astype(str).isin(w_sel['id'].astype(str))]
        w = w_fun(data_subset, dates, w_sel, wealth)
        pf = pf_ts_fun(w, data_subset, wealth)

        pf['u'] = u_val
        mv_results.append(pf)

    return pd.concat(mv_results, ignore_index=True)


def static_val_fun(data, dates, cov_list, lambda_list, wealth, cov_type,
                   gamma_rel, k=None, g=None, u=None, hps=None):
    """
    Static portfolio validation for ML-based portfolios.

    Parameters:
        data (pd.DataFrame): DataFrame containing portfolio data with columns like `id`, `eom`, `tr_ld1`, `pred_ld1`, etc.
        dates (list): List of dates for which the validation is performed.
        cov_list (dict): Dictionary of covariance matrices indexed by dates.
        lambda_list (dict): Dictionary of lambda matrices indexed by dates.
        wealth (pd.DataFrame): DataFrame containing wealth information with columns like `eom`, `mu_ld1`, and `wealth`.
        cov_type (str): Covariance adjustment type (e.g., "cov_mult", "cov_add", or "cor_shrink").
        gamma_rel (float): Relative risk aversion parameter.
        k (float, optional): Hyperparameter for lambda adjustment.
        g (float, optional): Hyperparameter for covariance adjustment.
        u (float, optional): Utility parameter.
        hps (pd.DataFrame, optional): DataFrame of hyperparameters indexed by `eom_ret`.

    Returns:
        pd.DataFrame: DataFrame containing static weights for the portfolio.
    """
    # Initialize weights with value-weighted (VW) as default
    static_weights = initial_weights_new(data, w_type="vw")

    # Merge only new columns to avoid duplicate column names
    static_weights = static_weights.merge(
        data[['id', 'eom', 'tr_ld1', 'pred_ld1']],
        on=['id', 'eom'],
        how='left'
    )

    static_weights = static_weights.merge(
        wealth[['eom', 'mu_ld1', 'wealth']],
        on='eom',
        how='left'
    )

    # Resolve duplicate column issue
    for col in ["tr_ld1", "pred_ld1", "mu_ld1"]:
        x_col, y_col = f"{col}_x", f"{col}_y"
        if x_col in static_weights.columns and y_col in static_weights.columns:
            static_weights[col] = static_weights[[x_col, y_col]].bfill(axis=1).iloc[:, 0]
            static_weights.drop(columns=[x_col, y_col], inplace=True)

    # Iterate over each date
    for d in dates:
        # If hyperparameters are provided, use them
        if hps is not None:
            recent_hps = hps[(hps['eom_ret'] < d) & (hps['eom_ret'] == hps[hps['eom_ret'] < d]['eom_ret'].max())]
            g = recent_hps['g'].iloc[0]
            u = recent_hps['u'].iloc[0]
            k = recent_hps['k'].iloc[0]

        # Extract inputs
        wealth_t = static_weights.loc[static_weights['eom'] == d, 'wealth'].iloc[0]
        ids = static_weights.loc[static_weights['eom'] == d, 'id'].astype(str).values

        # Retrieve and process sigma_gam
        sigma_data = cov_list.get(d)
        if sigma_data is None or "fct_cov" not in sigma_data:
            print(f"Warning: Missing 'fct_cov' for {d}")
            continue

        # Call the updated create_cov function to compute the covariance matrix
        sigma_gam = create_cov(sigma_data, ids)
        sigma_gam *= gamma_rel
        sigma_gam = sigma_gam_adj(sigma_gam, g=g, cov_type=cov_type)

        # Retrieve and process lambda_matrix
        lambda_data = lambda_list.get(d)
        if not lambda_data:
            print(f"Warning: No lambda values found for {d}")
            continue

        lambda_mat = create_lambda(lambda_data, ids)
        lambda_mat *= k

        # Extract weights and ensure pred_ld1 contains zeros for missing predictions
        pred_ld1 = static_weights.loc[static_weights['eom'] == d, 'pred_ld1'].fillna(0).values.reshape(-1,
                                                                                                       1)  # Replace NaN with 0
        static_weights.loc[static_weights['eom'] == d, 'w_start'] = static_weights.loc[
            static_weights['eom'] == d, 'w_start'].fillna(0)  # Replace NaN with 0 for w_start
        w_start = static_weights.loc[static_weights['eom'] == d, 'w_start'].values.reshape(-1, 1)
        pred_ld1 = np.nan_to_num(pred_ld1, nan=0.0)

        # Calculate RHS and A matrix
        rhs = (pred_ld1 * u) + wealth_t * (lambda_mat @ w_start)
        A = sigma_gam + wealth_t * lambda_mat

        try:
            # Try solving the matrix equation
            w_solution = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            # Handle singular matrix case
            print(f"Warning: Matrix inversion failed for date {d}. Returning zero weights instead.")
            w_solution = np.zeros_like(rhs)  # Return zero weights as fallback

        # Assign weights back to static_weights
        static_weights.loc[static_weights['eom'] == d, 'w'] = w_solution.flatten()

        # Update weights for the next month
        next_month_idx = list(dates).index(d) + 1
        if next_month_idx < len(dates):
            next_month = dates[next_month_idx]

            # 1) Create transition_weights from static_weights (rows for the date d),
            #    then shift 'eom' to the next_month and compute w_start:
            transition_weights = static_weights.loc[static_weights['eom'] == d].copy()
            transition_weights['eom'] = next_month
            transition_weights['w_start'] = (
                    transition_weights['w'] * (1 + transition_weights['tr_ld1']) /
                    (1 + transition_weights['mu_ld1'])
            )

            # 2) Merge, but keep BOTH w_start columns by giving the incoming one a suffix:
            #    - The current static_weights['w_start'] remains 'w_start'
            #    - The newly merged transition_weights['w_start'] becomes 'w_start_new'
            static_weights = static_weights.merge(
                transition_weights[['id', 'eom', 'w_start']],
                on=['id', 'eom'],
                how='left',
                suffixes=('', '_new')
            )

            # 3) Fill old w_start from the new w_start_new, only where it's NaN:
            mask = (static_weights['w_start'].isna()) & (static_weights['w_start_new'].notna())
            static_weights.loc[mask, 'w_start'] = static_weights.loc[mask, 'w_start_new']

            # 4) Drop the extra w_start_new column if you no longer need it:
            static_weights.drop(columns='w_start_new', inplace=True)

    # 5) **Now fill any remaining NaNs with zero at the very end of the entire loop**
    static_weights['w_start'] = static_weights['w_start'].fillna(0)

    return static_weights


def static_implement(data_tc, cov_list, lambda_list,
                     wealth, gamma_rel, dates_oos, dates_hp,
                     k_vec, u_vec, g_vec, cov_type,
                     validation=None, seed=None):
    """
    Full implementation of Static-ML portfolios.

    Parameters:
        data_tc (pd.DataFrame): Data containing relevant portfolio information.
                                Must have columns ['id','eom','me','tr_ld1','pred_ld1','valid'].
        cov_list (dict): Dictionary of covariance matrices indexed by date.
        lambda_list (dict): Dictionary of lambda matrices indexed by date.
        wealth (pd.DataFrame): Wealth data with columns ['eom', 'wealth'].
        gamma_rel (float): Relative risk-aversion parameter.
        dates_oos (list): List of out-of-sample dates.
        dates_hp (list): List of hyperparameter search dates.
        k_vec (list): List of k hyperparameter values to evaluate.
        u_vec (list): List of u hyperparameter values to evaluate.
        g_vec (list): List of g hyperparameter values to evaluate.
        cov_type (str): Covariance adjustment type.
        validation (pd.DataFrame, optional): Precomputed validation results. Defaults to None.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        dict: Dictionary containing:
            - "hps": Full validation DataFrame of hyperparameter trials.
            - "best_hps": DataFrame of the best hyperparameters per period.
            - "w": Final portfolio weights for out‐of‐sample dates.
            - "pf": Final portfolio performance.
    """

    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Create a grid of hyperparameters
    static_hps = pd.DataFrame(product(k_vec, u_vec, g_vec), columns=['k', 'u', 'g'])

    # Filter relevant data for hyperparameter dates
    data_rel = data_tc[(data_tc['valid'] == True) & (data_tc['eom'].isin(dates_hp))]
    data_rel = data_rel[['id', 'eom', 'me', 'tr_ld1', 'pred_ld1']].sort_values(by=['id', 'eom'])

    # Perform validation if not provided
    if validation is None:
        validation_list = []
        # Enumerate over rows so 'i' is a proper integer
        for i, hprow in tqdm(static_hps.iterrows(), total=len(static_hps), desc="Processing Hyperparameters"):
            k, u, g = float(hprow['k']), float(hprow['u']), float(hprow['g'])

            # Calculate weights for each combination
            static_w = static_val_fun(
                data=data_rel,
                dates=dates_hp,
                cov_list=cov_list,
                lambda_list=lambda_list,
                wealth=wealth,
                cov_type=cov_type,
                gamma_rel=gamma_rel,
                k=k, g=g, u=u
            )

            # Evaluate portfolio performance
            pf_results = pf_ts_fun(static_w, data_tc, wealth)
            pf_results['hp_no'] = i
            pf_results['k'] = k
            pf_results['u'] = u
            pf_results['g'] = g
            validation_list.append(pf_results)

        validation = pd.concat(validation_list, ignore_index=True)

    # Compute cumulative metrics
    validation = validation.sort_values(by=['hp_no', 'eom_ret'])
    validation['cum_mean_r'] = validation.groupby('hp_no')['r'].transform(lambda x: x.expanding().mean())
    validation['cum_mean_r2'] = validation.groupby('hp_no')['r'].transform(lambda x: (x ** 2).expanding().mean())
    validation['cum_var'] = validation['cum_mean_r2'] - validation['cum_mean_r'] ** 2
    validation['cum_obj'] = (
            validation['r'] - validation['tc'] - 0.5 * gamma_rel * validation['cum_var']
    ).groupby(validation['hp_no']).transform(lambda x: x.expanding().mean())
    validation['rank'] = validation.groupby('eom_ret')['cum_obj'].rank(ascending=False)

    # Find the best hyperparameters (example: pick 12th month & rank==1)
    optimal_hps = validation[(validation['eom_ret'].dt.month == 12) & (validation['rank'] == 1)]
    optimal_hps = optimal_hps.sort_values(by='eom_ret')

    # Implement final portfolio using chosen hyperparameters in out-of-sample period
    w = static_val_fun(
        data=data_tc[(data_tc['eom'].isin(dates_oos)) & (data_tc['valid'] == True)],
        dates=dates_oos,
        cov_list=cov_list,
        lambda_list=lambda_list,
        wealth=wealth,
        cov_type=cov_type,
        gamma_rel=gamma_rel,
        hps=optimal_hps
    )
    pf = pf_ts_fun(w, data_tc, wealth)
    pf['type'] = "Static-ML*"

    return {
        "hps": validation,
        "best_hps": optimal_hps,
        "w": w,
        "pf": pf
    }


def scale_constant(df):
    """Scale a Series or array so that the sum of squares equals 1, unless it's all zero."""

    s = np.sum(df ** 2)
    return df * np.sqrt(1.0 / s) if s != 0 else df


def scale_features_v1(df, feat_cons):
    """Normalize each column so that sum of squares = 1 per period with tqdm tracking."""

    unique_eoms = df["eom"].unique()  # Unique periods
    total_groups = len(unique_eoms)  # Number of groups to process
    progress_bar = tqdm(total=total_groups, desc="Scaling Features", unit="month")

    def normalize(x):
        """Normalize each feature within an `eom` period."""
        sum_sq = np.sum(x ** 2)
        scaled_x = x * np.sqrt(1.0 / (sum_sq + 1e-10)) if sum_sq > 0 else x
        progress_bar.update(1)  # Update progress bar
        return scaled_x

    # Apply function and track progress
    result = df.groupby("eom")[feat_cons].transform(lambda x: normalize(x))

    progress_bar.close()  # Close progress bar
    return result


def pfml_input_fun(data_tc, cov_list, lambda_list, gamma_rel, wealth, mu, dates, lb, scale,
                   risk_free, features, rff_feat, seed, p_max, g, add_orig, iter, balanced):
    """
    Prepares input data for Portfolio-ML optimization, including signal construction,
    feature transformations (e.g., RFF), volatility scaling, and lookback-weighted transformations.

    Parameters:
        data_tc (pd.DataFrame): Full dataset with features and returns.
        cov_list (dict): Covariance matrices indexed by date.
        lambda_list (dict): Transaction cost penalty matrices indexed by date.
        gamma_rel (float): Relative risk aversion parameter.
        wealth (pd.DataFrame): Portfolio wealth time series.
        mu (float): Market return assumption for excess return computation.
        dates (list): List of evaluation dates.
        lb (int): Lookback window size (in months).
        scale (bool): Whether to apply volatility scaling.
        risk_free (pd.DataFrame): Risk-free rate time series.
        features (list): List of original feature names.
        rff_feat (bool): Whether to use Random Fourier Features (RFF).
        seed (int): Seed for reproducibility of RFF generation.
        p_max (int): Number of RFF features (must be even).
        g (float): Bandwidth parameter for RFF transformation.
        add_orig (bool): Whether to append original features to RFF features.
        iter (int): Number of iterations for M-matrix calculation.
        balanced (bool): Whether to normalize features and scale them to unit variance.

    Returns:
        dict: Dictionary with transformed signal matrices and risk-return components per date.
    """

    dates = [pd.to_datetime(d) for d in dates]

    rff_w = None
    if rff_feat:
        np.random.seed(seed)
        rff_output = rff(data_tc[features].values, p=p_max, g=g)
        rff_w = rff_output["W"]
        rff_cos = rff_output["X_cos"]
        rff_sin = rff_output["X_sin"]
        rff_features = np.hstack([rff_cos, rff_sin])
        rff_columns = [f"rff{i}_cos" for i in range(1, p_max // 2 + 1)] + \
                      [f"rff{i}_sin" for i in range(1, p_max // 2 + 1)]
        rff_df = pd.DataFrame(rff_features, columns=rff_columns, index=data_tc.index)
        data = pd.concat([data_tc[['id', 'eom', 'valid', 'ret_ld1', 'tr_ld0', 'mu_ld0']], rff_df], axis=1)
        feat_new = rff_columns.copy()
        if add_orig:
            data = pd.concat([data, data_tc[features]], axis=1)
            feat_new.extend(features)
    else:
        data = data_tc[['id', 'eom', 'valid', 'ret_ld1', 'tr_ld0', 'mu_ld0'] + features].copy()
        feat_new = features.copy()

    data['id'] = data['id'].astype(str)
    data['eom'] = pd.to_datetime(data['eom'])
    feat_cons = feat_new + ['constant']

    # --- Step 2. Add volatility scales if requested (use full data) ---
    if scale:
        start_lb = (min(dates) + pd.DateOffset(days=1)) - pd.DateOffset(months=lb + 1) - pd.DateOffset(days=1)
        end_lb = max(dates) + pd.DateOffset(days=1)
        dates_lb = pd.date_range(start=start_lb, end=end_lb, freq='M')
        scales_list = []
        for d_ in dates_lb:
            sigma_data = cov_list.get(d_, None)
            if sigma_data is not None:
                sigma_temp = create_cov(sigma_data)
                if sigma_temp is not None and isinstance(sigma_temp, pd.DataFrame) and sigma_temp.shape[0] > 0:
                    diag_vol = np.sqrt(np.diag(sigma_temp))
                    scales_list.append(pd.DataFrame({'id': sigma_temp.index, 'eom': d_, 'vol_scale': diag_vol}))
        if scales_list:
            scales_df = pd.concat(scales_list, ignore_index=True)

            chunk_size = 100000
            chunked_data = []

            for i in range(0, len(data), chunk_size):
                data_chunk = data.iloc[i:i + chunk_size]
                merged_chunk = data_chunk.merge(scales_df, on=['id', 'eom'], how='left')
                chunked_data.append(merged_chunk)

            # Combine all chunks back into one DataFrame
            data = pd.concat(chunked_data, ignore_index=True)
            data['vol_scale'] = data.groupby('eom')['vol_scale'].transform(lambda x: x.fillna(x.median(skipna=True)))

    if balanced:
        data[feat_new] = data.groupby("eom")[feat_new].transform(lambda x: x - x.mean())
        data["constant"] = 1

        def scale_func(x):
            s = np.sum(x ** 2)
            return x * np.sqrt(1 / s) if s != 0 else x

        data[feat_cons] = data.groupby("eom")[feat_cons].transform(scale_func)

    inputs = {}
    for d in dates:
        d = pd.to_datetime(d)
        if (d.year % 10 == 0) and (d.month == 1):
            print(f"--> PF-ML inputs: {d.date()}")
        data_ret = data[(data["valid"] == True) & (data["eom"] == d)]
        ids = data_ret["id"].values
        id_to_index = {str(id_val): idx for idx, id_val in enumerate(ids)}
        r_vec = data_ret["ret_ld1"].values
        sigma = create_cov(cov_list[d], ids=[str(i) for i in ids])

        lambda_val = create_lambda(lambda_list[d], ids=[str(i) for i in ids])
        w_val = wealth.loc[wealth["eom"] == d, "wealth"].values[0]
        rf_val = risk_free.loc[risk_free["date"] == d, "rf"].values[0]
        m_mat = m_func(w=w_val, mu=mu, rf=rf_val, sigma_gam=sigma * gamma_rel,
                       gam=gamma_rel, lambda_mat=lambda_val, iterations=iter)

        # Lookback selection
        start_date = (d.replace(day=1) - pd.DateOffset(months=lb)) - pd.DateOffset(days=1)
        data_sub = data[
            (data['id'].isin(ids)) & (data['eom'] >= start_date) & (data['eom'] <= d) & (data['valid'] == True)].copy()

        if not balanced:
            data_sub[feat_new] = data_sub.groupby("eom")[feat_new].transform(lambda x: x - x.mean())
            data_sub["constant"] = 1
            data_sub[feat_cons] = data_sub.groupby("eom")[feat_cons].transform(
                lambda x: x * np.sqrt(1 / np.sum(x ** 2) if np.sum(x ** 2) != 0 else 1)
            )
        data_sub = data_sub.sort_values(by=["eom", "id"], ascending=[False, True])
        groups = [(grp_date, grp) for grp_date, grp in data_sub.groupby("eom")]
        groups.sort(key=lambda x: x[0], reverse=True)

        n = len(ids)
        p_dim = len(feat_cons)
        id_to_index = {str(id_val): idx for idx, id_val in enumerate(ids)}
        signals = {}
        for grp_date, grp in groups:
            # Initialize full signal matrix with zeros (n x p_dim)
            s_full = np.zeros((n, p_dim), dtype=float)
            # Fill in rows for stocks present in the group
            for _, row in grp.iterrows():
                i = id_to_index[row['id']]
                signal_vec = row[feat_cons].values.astype(float)
                if scale:
                    signal_vec *= 1.0 / row['vol_scale']
                s_full[i, :] = signal_vec
            signals[grp_date] = s_full

        gtm_list = []
        for grp_date, grp in groups:
            # Initialize gt vector with ones (for all n stocks)
            gt = np.ones(n, dtype=float)
            # For stocks present in the group, update the corresponding entry
            for _, row in grp.iterrows():
                i = id_to_index[row['id']]
                gt[i] = (1 + row["tr_ld0"]) / (1 + row["mu_ld0"])
            # Multiply full m_mat (n×n) by diag(gt) (n×n)
            gtm_list.append(m_mat @ np.diag(gt))

        groups_dates = [grp_date for grp_date, _ in groups]
        if len(groups_dates) < lb + 2:
            raise ValueError("Not enough lookback data in data_sub for lb.")
        gtm_agg = {}
        gtm_agg_l1 = {}
        gtm_agg[groups_dates[0]] = np.eye(n)
        gtm_agg_l1[groups_dates[0]] = np.eye(n)
        for i in range(1, lb + 1):
            gtm_agg[groups_dates[i]] = gtm_agg[groups_dates[i - 1]] @ gtm_list[i - 1]
            gtm_agg_l1[groups_dates[i]] = gtm_agg_l1[groups_dates[i - 1]] @ gtm_list[i]

        n = len(ids)
        p_dim = signals[groups_dates[0]].shape[1]
        omega = np.zeros((n, p_dim))
        const_mat = np.zeros((n, n))
        omega_l1 = np.zeros((n, p_dim))
        const_l1 = np.zeros((n, n))
        for i in range(lb + 1):
            d_new = groups_dates[i]
            d_new_l1 = groups_dates[i + 1]
            omega += gtm_agg[d_new] @ signals[d_new]
            const_mat += gtm_agg[d_new]
            omega_l1 += gtm_agg_l1[d_new] @ signals[d_new_l1]
            const_l1 += gtm_agg_l1[d_new]
        omega = np.linalg.inv(const_mat) @ omega
        omega_l1 = np.linalg.inv(const_l1) @ omega_l1

        grp_d = groups[0][1]
        gt_d = (1 + grp_d["tr_ld0"].values) / (1 + grp_d["mu_ld0"].values)
        omega_chg = omega - (np.diag(gt_d) @ omega_l1)

        r_tilde = (omega.T @ r_vec)
        risk_val = gamma_rel * (omega.T @ sigma @ omega)
        tc_val = w_val * (omega_chg.T @ lambda_val @ omega_chg)
        denom_val = risk_val + tc_val
        reals = {"r_tilde": r_tilde, "denom": denom_val, "risk": risk_val, "tc": tc_val}
        signal_t = signals[d] if d in signals else signals[groups_dates[0]]
        inputs[d] = {"reals": reals, "signal_t": signal_t}

    reals_dict = {str(d): inputs[d]["reals"] for d in inputs}
    signal_t_dict = {str(d): inputs[d]["signal_t"] for d in inputs}

    return {"reals": reals_dict, "signal_t": signal_t_dict, "rff_w": rff_w}


def pfml_feat_fun(p, orig_feat, features):
    """
    Generate a list of feature names for Portfolio-ML.

    Parameters:
        p (int): Number of random Fourier features (RFF). Must be even.
        orig_feat (bool): Whether to include original features.
        features (list): List of original feature names.

    Returns:
        list: A list of feature names, including RFF and optionally original features.
    """
    feat = ["constant"]
    if p != 0:
        feat += [f"rff{i}_cos" for i in range(1, p // 2 + 1)] + [f"rff{i}_sin" for i in range(1, p // 2 + 1)]
    if orig_feat:
        feat += features
    return feat


def denom_sum_fun(train):
    """
    Compute the sum of 'denom' matrices from a list of training data.

    Parameters:
        train (list): A list of dictionaries, each containing 'denom' DataFrame.

    Returns:
        pd.DataFrame: Sum of all 'denom' matrices in the list.
    """
    if not train:
        return None

    # Extract all 'denom' DataFrames
    denom_list = [x['denom'] for x in train]

    # Sum all DataFrames element-wise
    denom_sum = sum(denom_list, start=pd.DataFrame(0, index=denom_list[0].index, columns=denom_list[0].columns))

    return denom_sum


def pfml_search_coef(pfml_input, p_vec, l_vec, hp_years, orig_feat, features):
    """
    Hyperparameter search for best coefficients in Portfolio-ML.

    Parameters:
        pfml_input (dict): Dictionary containing 'reals' with:
            - 'r_tilde': Series (vector of target values)
            - 'denom': DataFrame (square matrix of denominators)
        p_vec (list): List of feature dimensions to evaluate.
        l_vec (numpy.ndarray): Array of regularization parameters (lambdas) to evaluate.
        hp_years (list): List of years for hyperparameter selection.
        orig_feat (bool): Whether to include original features.
        features (list): List of original feature names.

    Returns:
        dict: Nested dictionary containing coefficients for each (year, p, lambda).
    """

    # Identify the last year before hyperparameter selection starts
    end_bef = min(hp_years) - 1

    # Convert dictionary keys to datetime for comparison
    reals_dates = {pd.to_datetime(k): v for k, v in pfml_input['reals'].items()}

    # Select all data strictly before (end_bef-1)-12-31
    cutoff_date = pd.to_datetime(f"{end_bef - 1}-12-31")
    train_bef = {k: v for k, v in reals_dates.items() if k < cutoff_date}

    # Compute the initial sum of r_tilde and denom before hyperparameter years
    r_tilde_sum = pd.DataFrame({k: v['r_tilde'] for k, v in train_bef.items()}).sum(axis=1)
    denom_raw_sum = denom_sum_fun(list(train_bef.values()))

    # Count of training samples used
    n = len(train_bef)

    # Sort hyperparameter years
    hp_years = sorted(hp_years)
    coef_list = defaultdict(dict)

    # Iterate through each hyperparameter evaluation year
    for year in hp_years:
        # Define training date range for this hyperparameter year
        start_dt = pd.to_datetime(f"{year - 2}-12-31")
        end_dt = pd.to_datetime(f"{year - 1}-11-30")

        # Select relevant training data
        train_new = {k: v for k, v in reals_dates.items() if start_dt <= k <= end_dt}

        # Update training sample count
        n += len(train_new)

        if train_new:
            # Update r_tilde and denom sums
            r_tilde_new = pd.DataFrame({k: v['r_tilde'] for k, v in train_new.items()}).sum(axis=1)
            denom_raw_new = denom_sum_fun(list(train_new.values()))

            r_tilde_sum += r_tilde_new
            denom_raw_sum += denom_raw_new

        # Compute the coefficients for each (p, lambda) combination
        for p in p_vec:
            # Select the feature subset based on p and orig_feat setting
            feat_p = pfml_feat_fun(p, orig_feat, features)

            # Subset r_tilde and denom, then normalize by n
            r_tilde_sub = r_tilde_sum.loc[feat_p] / n
            denom_sub = denom_raw_sum.loc[feat_p, feat_p] / n

            # Solve for each lambda in l_vec
            lambda_solutions = {}
            for l_ in l_vec:
                # Convert denom_sub to NumPy before modifying diagonal
                denom_reg = denom_sub.to_numpy().copy()
                np.fill_diagonal(denom_reg, denom_reg.diagonal() + l_)

                # Solve the system: (denom_sub + lambda * I)^{-1} * r_tilde_sub
                coef_vector = np.linalg.solve(denom_reg, r_tilde_sub.to_numpy())
                lambda_solutions[l_] = pd.Series(coef_vector, index=feat_p)

            # Store solutions for this p-value
            coef_list[year][p] = lambda_solutions

    return dict(coef_list)


def pfml_hp_reals_fun(pfml_input, hp_coef, p_vec, l_vec, hp_years, orig_feat, features):
    """
    Compute realized utility for each p and lambda in each hyperparameter evaluation year.

    Parameters:
        pfml_input (dict): Input containing 'reals' with realizations and denominators.
        hp_coef (dict): Hyperparameter coefficients.
        p_vec (list): List of numbers of random Fourier features (RFF) to evaluate.
        l_vec (list): List of regularization parameters (lambdas) to evaluate.
        hp_years (list): List of hyperparameter evaluation years.
        orig_feat (bool): Whether to include original features.
        features (list): List of original feature names.

    Returns:
        pd.DataFrame: Validation results with cumulative objectives and ranks for each combination.
    """
    validation = []

    for end in tqdm(hp_years, desc="Processing Out-of-Sample Years"):
        # Filter realizations for the current year
        reals_all = {
            k: v for k, v in pfml_input['reals'].items()
            if np.datetime64(f"{end - 1}-12-31") <= np.datetime64(k) <= np.datetime64(f"{end}-11-30")
        }

        # Fix integer indexing issue
        coef_list_yr = hp_coef[end]

        for p in tqdm(p_vec, desc=f"Processing p-values for Year {end}", leave=False):
            # Generate features for the given p
            feat_p = pfml_feat_fun(p, orig_feat, features)
            coef_list_p = coef_list_yr[p]

            # Extract relevant realizations
            reals = {
                nm: {
                    "r_tilde": x["r_tilde"].loc[feat_p],
                    "denom": x["denom"].loc[feat_p, feat_p]
                }
                for nm, x in reals_all.items()
            }

            for l in l_vec:
                coef = coef_list_p[l]
                objs = []

                for nm, x in reals.items():
                    # Compute objective value
                    r = (x["r_tilde"].T @ coef - 0.5 * coef.T @ x["denom"] @ coef).item()
                    objs.append({
                        "eom": pd.Timestamp(nm),
                        "eom_ret": pd.Timestamp(nm) + pd.DateOffset(months=1),
                        "obj": r,
                        "l": l,
                        "p": p,
                        "hp_end": end
                    })

                validation.extend(objs)

    # Convert to DataFrame
    validation_df = pd.DataFrame(validation)

    # Compute cumulative objectives and ranks
    validation_df = validation_df.sort_values(["p", "l", "eom_ret"])
    validation_df["cum_obj"] = validation_df.groupby(["p", "l"])["obj"].expanding().mean().reset_index(drop=True)
    validation_df["rank"] = validation_df.groupby("eom_ret")["cum_obj"].rank(ascending=False)

    return validation_df


def pfml_aims_fun(pfml_input, validation, data_tc, hp_coef, dates_oos, orig_feat, features):
    """
    Create the optimal aim portfolio for each out-of-sample date.

    Parameters:
        pfml_input (dict): Input containing signals and realizations.
        validation (pd.DataFrame): Validation results with hyperparameter rankings.
        data_tc (pd.DataFrame): Dataframe containing stock data.
        hp_coef (dict): Coefficients for hyperparameters.
        dates_oos (list): Out-of-sample dates for portfolio implementation.
        orig_feat (bool): Whether to include original features.
        features (list): List of original feature names.

    Returns:
        dict: A dictionary of aim portfolios and coefficients for each date.
    """
    # Filter the best hyperparameters for each year
    opt_hps = validation[
        (validation['eom_ret'].dt.month == 12) & (validation['rank'] == 1)
        ].assign(hp_end=validation['eom_ret'].dt.year)[['hp_end', 'l', 'p']]

    aim_pfs_list = {}

    for d in dates_oos:
        d_ret = pd.Timestamp(d) + pd.offsets.MonthEnd(1)
        oos_year = d_ret.year
        hp_year = oos_year - 1

        # Get best hyperparameters for the previous year
        hps_d = opt_hps[opt_hps['hp_end'] == hp_year]
        if hps_d.empty:
            continue  # Skip if no hyperparameters are available for the year

        # Extract features for the aim portfolio
        feat = pfml_feat_fun(p=hps_d['p'].iloc[0], orig_feat=orig_feat, features=features)

        pfml_input['signal_t'] = {pd.Timestamp(k): v for k, v in pfml_input['signal_t'].items()}

        # Now access it normally without converting every time
        s = pfml_input['signal_t'][pd.Timestamp(d)].loc[:, feat]

        # Match lambda and extract coefficients correctly
        l_value = hps_d['l'].iloc[0]
        p_value = hps_d['p'].iloc[0]

        if l_value not in hp_coef[oos_year][p_value]:
            continue  # Skip if the lambda value is missing

        coef = hp_coef[oos_year][p_value][l_value]

        # Calculate aim portfolio weights
        data_subset = data_tc[(data_tc['valid'] == True) & (data_tc['eom'] == d)].copy()
        aim_pf = data_subset[['id', 'eom']].copy()
        aim_pf['w_aim'] = np.dot(s, coef)

        aim_pfs_list[d] = {"aim_pf": aim_pf, "coef": coef}

    return aim_pfs_list


def pfml_w(data: pd.DataFrame, dates: list, cov_list: dict, lambda_list: dict, gamma_rel: float,
           iter_: int, risk_free: pd.DataFrame, wealth: pd.DataFrame, mu: float,
           aims: pd.DataFrame = None, signal_t: dict = None, aim_coef: any = None) -> pd.DataFrame:
    """
    Compute Portfolio-ML weights.

    If 'aims' is None, an Aim Portfolio is constructed from 'signal_t' and 'aim_coef'.
    Otherwise, we simply reuse the provided aims.

    Parameters:
        data (pd.DataFrame): Portfolio data with columns at least ['id','eom','tr_ld1','mu_ld1'].
        dates (list): List of EOM dates for portfolio calculations.
        cov_list (dict): Covariance matrices for each date (key=str(date)),
                         each a matrix or data for create_cov(...).
        lambda_list (dict): Lambda matrices for each date (key=str(date)),
                            each a matrix or data for create_lambda(...).
        gamma_rel (float): Risk-aversion parameter.
        iter_ (int): Iterations for computing the m matrix.
        risk_free (pd.DataFrame): Must have ['eom','rf'] columns for each date.
        wealth (pd.DataFrame): Must have ['eom','wealth','mu_ld1'] columns for each date.
        mu (float): Portfolio drift parameter.
        aims (pd.DataFrame, optional): If provided, must have ['id','eom','w_aim'] columns.
        signal_t (dict, optional): {str(date): 2D array of signals} for each date.
        aim_coef (dict or 1D array, optional): If dict, keys are years as str; if array, used for all dates.

    Returns:
        pd.DataFrame: Final portfolio weights by date/stock, including intermediate columns.
    """

    # 1. If no 'aims', build them using signal_t and aim_coef
    if aims is None:
        aim_list = []
        for d in dates:
            d_str = str(pd.Timestamp(d))
            data_d = data.loc[data['eom'] == pd.Timestamp(d), ['id', 'eom']].copy()
            if (signal_t is not None) and (d_str in signal_t):
                s = signal_t[d_str]
            else:
                # If signals are not available, default to zero
                s = np.zeros((len(data_d), 1))

            # Extract the correct coefficient
            if isinstance(aim_coef, dict):
                # Dictionary keyed by year
                year_str = str(pd.Timestamp(d).year)
                coef = aim_coef.get(year_str, 0)
            else:
                # Single array or None
                coef = aim_coef if aim_coef is not None else 0

            # If coef is an array, do matrix multiplication
            # If it's a single float, broadcast multiply
            if isinstance(coef, (list, np.ndarray)):
                coef = np.array(coef)
                data_d['w_aim'] = s @ coef
            else:
                data_d['w_aim'] = s * coef

            aim_list.append(data_d)
        aims = pd.concat(aim_list, ignore_index=True)

    # 2. Initialize weights with a "vw" (value-weighted) strategy
    fa_weights = initial_weights_new(data, w_type="vw")

    # 3. Compute weights iteratively for each date
    for idx, d in enumerate(dates):
        cur_date = pd.Timestamp(d)
        # Filter to the stocks for this date
        ids_d = data.loc[data['eom'] == cur_date, 'id'].astype(str).unique()

        # Covariance & Lambda
        cov_list = {pd.Timestamp(k): v for k, v in cov_list.items()}
        lambda_list = {pd.Timestamp(k): v for k, v in lambda_list.items()}
        sigma = create_cov(cov_list[d], ids=ids_d)
        lambda_mat = create_lambda(lambda_list[d], ids=ids_d)

        # Wealth & risk-free
        w_val = wealth.loc[wealth['eom'] == cur_date, 'wealth'].iloc[0]
        rf_val = risk_free.loc[risk_free['date'] == cur_date, 'rf'].iloc[0]

        # m matrix
        m = m_func(
            w=w_val, mu=mu, rf=rf_val,
            sigma_gam=sigma * gamma_rel,
            gam=gamma_rel,
            lambda_mat=lambda_mat,
            iterations=iter_
        )
        iden = np.eye(m.shape[0])

        # Merge aims with the existing weights for this date
        aims['id'] = aims['id'].astype(str)
        fa_weights['id'] = fa_weights['id'].astype(str)

        w_cur = aims.loc[aims['eom'] == cur_date, ['id', 'eom', 'w_aim']].merge(
            fa_weights.loc[fa_weights['eom'] == cur_date],
            on=['id', 'eom'], how='left'
        )

        # w_opt = m * w_start + (I - m) * w_aim
        w_cur['w_opt'] = (
                (m @ w_cur['w_start'].fillna(0).values) +
                ((iden - m) @ w_cur['w_aim'].values)
        )

        # Save the optimized weight for the current month
        fa_weights.loc[
            (fa_weights['eom'] == cur_date) & (fa_weights['id'].isin(w_cur['id'])),
            'w'
        ] = w_cur['w_opt'].values

        # If we have a next month, carry forward the weight
        if idx + 1 < len(dates):
            next_date = pd.Timestamp(dates[idx + 1])
            w_cur['w_start_next'] = (
                    w_cur['w_opt'] * (1 + w_cur['tr_ld1']) / (1 + w_cur['mu_ld1'])
            ).fillna(0)

            # Assign w_start for next_date
            for row_i, row in w_cur.iterrows():
                fa_weights.loc[
                    (fa_weights['eom'] == next_date) & (fa_weights['id'] == row['id']),
                    'w_start'
                ] = row['w_start_next']

    return fa_weights


def pfml_implement(data_tc, cov_list, lambda_list, risk_free, features, wealth, mu, gamma_rel,
                   dates_full, dates_oos, lb, hp_years, rff_feat, scale, g_vec=None, p_vec=None,
                   l_vec=None, orig_feat=None, iter=100, hps=None, balanced=False, seed=None):
    """
    Portfolio-ML implementation with hyperparameter search and portfolio computation.

    Parameters:
        data_tc (pd.DataFrame): Time-series portfolio data.
        cov_list (dict): Covariance matrices for each date.
        lambda_list (dict): Lambda matrices for each date.
        risk_free (pd.DataFrame): Risk-free rate data.
        features (list): Feature names.
        wealth (pd.DataFrame): Wealth data.
        mu (float): Drift parameter.
        gamma_rel (float): Risk-aversion parameter.
        dates_full (list): Full set of dates.
        dates_oos (list): Out-of-sample dates.
        lb (int): Lookback window in months.
        hp_years (list): Years for hyperparameter search.
        rff_feat (bool): Use random Fourier features or not.
        scale (bool): Scale features or not.
        g_vec, p_vec, l_vec (list): Hyperparameter grids for g, p, and lambda.
        orig_feat (bool): Whether to include original features alongside RFF.
        iter (int): Iteration count for matrix computations.
        hps (dict): Precomputed hyperparameters; if None, we perform the search.
        balanced (bool): Use balanced data transformations.
        seed (int): Random seed.

    Returns:
        dict: A dictionary containing:
            - "hps": All hyperparameter results.
            - "best_hps": Best hyperparameters at year-end.
            - "best_hps_list": Picks of g, p, lambda for each out-of-sample date.
            - "aims": Combined aim portfolios.
            - "w": Final portfolio weights.
            - "pf": Final performance DataFrame.
            - "rff_w_list": RFF weight objects.
    """

    if not hps:
    	hps = {}
    	for g in g_vec:
    		print(f"Processing g: {g}")

    		if settings["multi_process"]:
    			pfml_input = pfml_input_fun_multi(
    				data_tc, cov_list, lambda_list,
    				gamma_rel, wealth, mu, dates_full,
    				lb, scale, risk_free, features,
    				rff_feat, seed, p_max=max(p_vec),
    				g=g, add_orig=orig_feat,
    				iter=iter, balanced=balanced
    			)
    		else:
    			pfml_input = pfml_input_fun(
    				data_tc, cov_list, lambda_list,
    				gamma_rel, wealth, mu, dates_full,
    				lb, scale, risk_free, features,
    				rff_feat, seed, p_max=max(p_vec),
    				g=g, add_orig=orig_feat,
    				iter=iter, balanced=balanced
    			)

    		# with open(f"/work/frontier_ml_eu/pfml_input_g{g:.4f}.pkl", "wb") as f:
    		# 	pickle.dump(pfml_input, f)

    		rff_w = pfml_input.get("rff_w")

    		feat_all = pfml_feat_fun(max(p_vec), orig_feat, features)
    		reals_adj = {}
    		for date_key, val in pfml_input["reals"].items():
    			reals_adj[date_key] = {
    				"r_tilde": val["r_tilde"].loc[feat_all]
    			}
    			for subk, mat_ in val.items():
    				if subk != "r_tilde":
    					reals_adj[date_key][subk] = mat_.loc[feat_all, feat_all]

    		signals_adj = {
    			date_key: mat_.loc[:, feat_all]
    			for date_key, mat_ in pfml_input["signal_t"].items()
    		}

    		pfml_input = {
    			"reals": reals_adj,
    			"signal_t": signals_adj
    		}

    		pfml_hp_coef = pfml_search_coef(
    			pfml_input, p_vec, l_vec, hp_years,
    			orig_feat, features
    		)

    		validation = pfml_hp_reals_fun(
    			pfml_input, pfml_hp_coef, p_vec, l_vec,
    			hp_years, orig_feat, features
    		)
    		validation["g"] = g

    		aims = pfml_aims_fun(
    			pfml_input, validation, data_tc,
    			pfml_hp_coef, dates_oos,
    			orig_feat, features
    		)
    		hps[g] = {
    			"aim_pfs_list": aims,
    			"validation": validation,
    			"rff_w": rff_w
    		}

    		# with open(f"/work/frontier_ml_g5/hps_g{g:.4f}.pkl", "wb") as f:
    		# 	pickle.dump(hps[g], f)

    # # Load both pfml_input and hps files
    # if not hps:
    #     hps = {}
    #     for g in g_vec:
    #         print(f"Loading precomputed data for g: {g}")
    #
    #         # Load hps data
    #         with open(f"/work/frontier_ml/hps_g{g:.4f}.pkl", "rb") as f:
    #             hps[g] = pickle.load(f)

    # 2. Find best hyperparameters at end of year
    best_hps = pd.concat([hps[g]["validation"] for g in hps], ignore_index=True)
    best_hps["rank"] = best_hps.groupby("eom_ret")["cum_obj"].rank(ascending=False)
    best_hps = best_hps[
        (best_hps["rank"] == 1) & (best_hps["eom_ret"].dt.month == 12)
        ]

    # 3. Build aim portfolios for OOS dates
    best_hps_list = []
    for d in dates_oos:
        d_ret = d + pd.offsets.MonthEnd(1)
        oos_year = d_ret.year
        hp_sel = best_hps[best_hps["eom_ret"].dt.year == (oos_year - 1)]
        g_val = hp_sel["g"].iloc[0]
        p_val = hp_sel["p"].iloc[0]

        aim_pf = hps[g_val]["aim_pfs_list"][d]["aim_pf"]
        coef_ = hps[g_val]["aim_pfs_list"][d]["coef"]
        best_hps_list.append({"g": g_val, "p": p_val, "aim": aim_pf, "coef": coef_})

    aims = pd.concat([x["aim"] for x in best_hps_list], ignore_index=True)

    data_tc_filtered = data_tc[(data_tc['eom'].isin(dates_oos)) & (data_tc['valid'] == True)]

    w = pfml_w(
        data=data_tc_filtered,
        dates=dates_oos,
        cov_list=cov_list,
        lambda_list=lambda_list,
        gamma_rel=gamma_rel,
        iter_=iter,
        risk_free=risk_free,
        wealth=wealth,
        mu=mu,
        signal_t=None,
        aim_coef=None,
        aims=aims
    )

    pf = pf_ts_fun(w, data_tc_filtered, wealth)
    pf["type"] = "Portfolio-ML"

    # 5. RFF weight list
    rff_w_list = {g: hps[g]["rff_w"] for g in hps}

    # Return all components
    return {
        "hps": hps,
        "best_hps": best_hps,
        "best_hps_list": best_hps_list,
        "aims": aims,
        "w": w,
        "pf": pf,
        "rff_w_list": rff_w_list
    }


# Counterfactual
def pfml_cf_fun(data, cf_cluster, pfml_base, dates, cov_list, lambda_list, scale,
                orig_feat, gamma_rel, wealth, risk_free, mu, iter, seed,
                features, cluster_labels):
    """
    Implements Portfolio-ML with counterfactual feature permutations based on cluster membership,
    and evaluates resulting portfolios using previously fitted hyperparameters.

    Parameters:
        data (pd.DataFrame): Dataset with stock-level returns and features.
        cf_cluster (str): Cluster name for counterfactual permutation ('bm' for baseline).
        pfml_base (dict): Dictionary containing fitted PF-ML models and hyperparameters.
        dates (list): List of evaluation dates.
        cov_list (dict): Covariance matrices per date.
        lambda_list (dict): Transaction cost penalty matrices per date.
        scale (bool): Whether to apply volatility scaling to features.
        orig_feat (bool): Whether original features are included in model fitting.
        gamma_rel (float): Relative risk aversion parameter.
        wealth (pd.DataFrame): Time series of wealth data.
        risk_free (pd.DataFrame): Risk-free rate data.
        mu (float): Market return assumption for M-function.
        iter (int): Number of M-function iterations.
        seed (int): Random seed for reproducibility.
        features (list): List of features used in the model.
        cluster_labels (pd.DataFrame): Mapping of characteristics to clusters.

    Returns:
        pd.DataFrame: Portfolio performance results from the counterfactual analysis.
    """

    np.random.seed(seed)
    cf = data.copy()

    if scale:
        scales = []
        for d in dates:
            sigma_arr = create_cov(cov_list[d])
            sigma = pd.DataFrame(sigma_arr)
            diag_vol = np.sqrt(np.diag(sigma.values))

            scales.append(pd.DataFrame({'id': sigma.index, 'eom': d, 'vol_scale': diag_vol}))

        scales = pd.concat(scales, ignore_index=True)
        cf['id'] = cf['id'].astype(str)  # Ensure 'id' in 'cf' is of type string
        scales['id'] = scales['id'].astype(str)  # Ensure 'id' in 'scales' is of type string
        cf = cf.merge(scales, on=['id', 'eom'], how='left')

        cf['vol_scale'] = cf.groupby('eom')['vol_scale'].transform(lambda x: x.fillna(x.median()))

    if cf_cluster != "bm":
        cf['id_shuffle'] = cf.groupby('eom')['id'].transform(lambda x: np.random.permutation(x))

        # Identify characteristics to replace for this cluster
        chars_sub = cluster_labels.loc[
            (cluster_labels['cluster'] == cf_cluster) & (cluster_labels['characteristic'].isin(features)),
            'characteristic'
        ]

        if not chars_sub.empty:
            # Replace the features for the shuffled IDs
            chars_data = cf[['id_shuffle', 'eom'] + chars_sub.tolist()].rename(columns={'id_shuffle': 'id'})
            cf.drop(columns=chars_sub, inplace=True)
            cf = pd.merge(cf, chars_data, on=['id', 'eom'], how='left')

    # Generate aim portfolios
    aim_cf = []
    for d in dates:
        stocks = cf.loc[cf['eom'] == d, 'id']

        # Find the matching index by searching the 'eom' column in the 'aim' DataFrame of each item
        matched_index = next(
            (i for i, item in enumerate(pfml_base['best_hps_list']) if d in item['aim']['eom'].values),
            None
        )

        if matched_index is None:
            raise ValueError(f"Date {d} not found in any 'aim' DataFrame in 'best_hps_list'.")

        # Accessing the values using the matched index
        best_g = pfml_base['best_hps_list'][matched_index]['g']
        best_p = pfml_base['best_hps_list'][matched_index]['p']
        aim_coef = pfml_base['best_hps_list'][matched_index]['coef']
        W = pfml_base['hps'][best_g]['rff_w'][:, :best_p // 2]

        # RFF for counterfactuals
        rff_x = rff(cf.loc[cf['eom'] == d, features].values, W=W)
        s = np.hstack([rff_x['X_cos'], rff_x['X_sin']])
        s = (s - s.mean(axis=0)) * np.sqrt(1 / np.sum(s ** 2, axis=0))
        s = np.hstack([s, np.ones((s.shape[0], 1))])

        feat = pfml_feat_fun(p=best_p, orig_feat=orig_feat, features=features)
        s = s[:, [feat.index(f) for f in feat]]

        if scale:
            scale_diag = np.diag(1 / cf.loc[cf['eom'] == d, 'vol_scale'])
            s = scale_diag @ s

        aim_cf.append(pd.DataFrame({'id': stocks, 'eom': d, 'w_aim': s @ aim_coef}))

    if not aim_cf:
        return pd.DataFrame()

    aim_cf = pd.concat(aim_cf, ignore_index=True)

    # Compute weights using the aim portfolio
    w_cf = pfml_w(data, dates, cov_list, lambda_list, gamma_rel, iter, risk_free, wealth, mu, aim_cf)
    pf_cf = pf_ts_fun(w_cf, data, wealth)
    pf_cf['type'] = "Portfolio-ML"
    pf_cf['cluster'] = cf_cluster

    # Remove rows where relevant columns are 0
    pf_cf = pf_cf[~(pf_cf[['inv', 'shorting', 'turnover', 'r', 'tc']].sum(axis=1) == 0)]

    return pf_cf


################################################################################
# ------------------------- Multiprocessing functions------------------------- #
################################################################################

# Minimum-variance
def process_mv_date(d, data_split, cov_list):
    """
    Helper function to process one date for minimum-variance weights.
    """
    if d not in data_split:
        return None

    data_sub = data_split[d]
    ids = data_sub['id'].astype(str).values

    sigma_data = cov_list.get(d, None)
    if sigma_data is None:
        print(f"Warning: No covariance data found for {d}")
        return None

    sigma = create_cov(sigma_data, ids)
    sigma = pd.DataFrame(sigma, index=ids, columns=ids)

    if sigma.shape[0] != sigma.shape[1] or sigma.shape[0] == 0:
        print(f"Warning: Invalid covariance matrix for {d}")
        return None

    try:
        sigma_inv = np.linalg.pinv(sigma)
        ones = np.ones((sigma_inv.shape[0], 1))
        weights = (sigma_inv @ ones) / (ones.T @ sigma_inv @ ones)

        data_sub = data_sub[data_sub["id"].astype(str).isin(sigma.index)]
        data_sub['w'] = weights.flatten()

        print(f"[{d}] Processed {len(weights)} assets for min var portfolio.")

        return data_sub[['id', 'eom', 'w']]
    except Exception as e:
        print(f"Error processing date {d}: {e}")
        return None


def mv_implement_multip(data, cov_list, wealth, dates):
    """
    Minimum-variance portfolio implementation with multiprocessing.

    Parameters:
        data (pd.DataFrame): Portfolio data.
        cov_list (dict): Dictionary of covariance matrices indexed by dates.
        wealth (pd.DataFrame or list): Wealth data.
        dates (list): List of portfolio construction dates.

    Returns:
        dict: Contains:
            - "w": Portfolio weights DataFrame.
            - "pf": Portfolio performance DataFrame.
    """

    n_jobs = max(1, multiprocessing.cpu_count() // 2)

    data_split = {
        date: sub_df.copy() for date, sub_df in data[
            (data['valid']) & (data['eom'].isin(dates))
            ][['id', 'eom', 'me']].groupby('eom')
    }

    # Parallel processing
    mv_opt_list = Parallel(n_jobs=n_jobs)(
        delayed(process_mv_date)(d, data_split, cov_list) for d in dates
    )

    # Remove None results and concatenate
    mv_opt = pd.concat([res for res in mv_opt_list if res is not None], ignore_index=True)

    if mv_opt.empty:
        return {"w": pd.DataFrame(), "pf": pd.DataFrame()}

    mv_w = w_fun(data[data['eom'].isin(dates) & data['valid']], dates, mv_opt, wealth)
    mv_pf = pf_ts_fun(mv_w, data, wealth)
    mv_pf['type'] = "Minimum Variance"

    return {"w": mv_w, "pf": mv_pf}


# Static
def static_implement_multip(data_tc, cov_list, lambda_list,
                            wealth, gamma_rel, dates_oos, dates_hp,
                            k_vec, u_vec, g_vec, cov_type,
                            validation=None, seed=None):
    """
    Full implementation of Static-ML portfolios with multiprocessing support.

    Parameters:
        data_tc (pd.DataFrame): Data containing relevant portfolio information.
        cov_list (dict): Dictionary of covariance matrices indexed by date.
        lambda_list (dict): Dictionary of lambda matrices indexed by date.
        wealth (pd.DataFrame): Wealth data with columns ['eom', 'wealth'].
        gamma_rel (float): Relative risk-aversion parameter.
        dates_oos (list): List of out-of-sample dates.
        dates_hp (list): List of hyperparameter search dates.
        k_vec (list): List of k hyperparameter values to evaluate.
        u_vec (list): List of u hyperparameter values to evaluate.
        g_vec (list): List of g hyperparameter values to evaluate.
        cov_type (str): Covariance adjustment type.
        validation (pd.DataFrame, optional): Precomputed validation results.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        dict: {
            "hps": Full validation DataFrame of hyperparameter trials,
            "best_hps": DataFrame of the best hyperparameters per period,
            "w": Final portfolio weights for out-of-sample dates,
            "pf": Final portfolio performance.
        }
    """
    n_jobs = max(1, multiprocessing.cpu_count() // 2)

    if seed is not None:
        np.random.seed(seed)

    # Build hyperparameter grid
    static_hps = pd.DataFrame(product(k_vec, u_vec, g_vec), columns=['k', 'u', 'g'])

    # Prepare input data for hyperparameter validation
    data_rel = data_tc[(data_tc['valid'] == True) & (data_tc['eom'].isin(dates_hp))]
    data_rel = data_rel[['id', 'eom', 'me', 'tr_ld1', 'pred_ld1']].sort_values(by=['id', 'eom'])

    # Perform validation if not already done
    if validation is None:
        def evaluate_hyperparams(i, hprow):
            k, u, g = float(hprow['k']), float(hprow['u']), float(hprow['g'])
            static_w = static_val_fun(
                data=data_rel,
                dates=dates_hp,
                cov_list=cov_list,
                lambda_list=lambda_list,
                wealth=wealth,
                cov_type=cov_type,
                gamma_rel=gamma_rel,
                k=k, g=g, u=u
            )
            pf_results = pf_ts_fun(static_w, data_tc, wealth)
            pf_results['hp_no'] = i
            pf_results['k'] = k
            pf_results['u'] = u
            pf_results['g'] = g
            return pf_results

        print("Running hyperparameter search with multiprocessing...")
        validation_list = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_hyperparams)(i, hprow)
            for i, hprow in tqdm(static_hps.iterrows(), total=len(static_hps))
        )

        validation = pd.concat(validation_list, ignore_index=True)

    # Compute cumulative metrics
    validation = validation.sort_values(by=['hp_no', 'eom_ret'])
    validation['cum_mean_r'] = validation.groupby('hp_no')['r'].transform(lambda x: x.expanding().mean())
    validation['cum_mean_r2'] = validation.groupby('hp_no')['r'].transform(lambda x: (x ** 2).expanding().mean())
    validation['cum_var'] = validation['cum_mean_r2'] - validation['cum_mean_r'] ** 2
    validation['cum_obj'] = (
            validation['r'] - validation['tc'] - 0.5 * gamma_rel * validation['cum_var']
    ).groupby(validation['hp_no']).transform(lambda x: x.expanding().mean())
    validation['rank'] = validation.groupby('eom_ret')['cum_obj'].rank(ascending=False)

    # Choose best hyperparams per year-end
    optimal_hps = validation[(validation['eom_ret'].dt.month == 12) & (validation['rank'] == 1)]
    optimal_hps = optimal_hps.sort_values(by='eom_ret')

    # Final out-of-sample implementation
    w = static_val_fun(
        data=data_tc[(data_tc['eom'].isin(dates_oos)) & (data_tc['valid'] == True)],
        dates=dates_oos,
        cov_list=cov_list,
        lambda_list=lambda_list,
        wealth=wealth,
        cov_type=cov_type,
        gamma_rel=gamma_rel,
        hps=optimal_hps
    )

    pf = pf_ts_fun(w, data_tc, wealth)
    pf['type'] = "Static-ML*"

    return {
        "hps": validation,
        "best_hps": optimal_hps,
        "w": w,
        "pf": pf
    }

# Portfolio ML
def pfml_input_fun_multi(data_tc, cov_list, lambda_list, gamma_rel, wealth, mu, dates, lb, scale,
                         risk_free, features, rff_feat, seed, p_max, g, add_orig, iter, balanced):
    dates = [pd.to_datetime(d) for d in dates]

    rff_w = None
    if rff_feat:
        np.random.seed(seed)
        rff_output = rff(data_tc[features].values, p=p_max, g=g)
        rff_w = rff_output["W"]
        rff_cos = rff_output["X_cos"]
        rff_sin = rff_output["X_sin"]
        rff_features = np.hstack([rff_cos, rff_sin])
        rff_columns = [f"rff{i}_cos" for i in range(1, p_max // 2 + 1)] + \
                      [f"rff{i}_sin" for i in range(1, p_max // 2 + 1)]
        rff_df = pd.DataFrame(rff_features, columns=rff_columns, index=data_tc.index)
        data = pd.concat([data_tc[['id', 'eom', 'valid', 'ret_ld1', 'tr_ld0', 'mu_ld0']], rff_df], axis=1)
        feat_new = rff_columns.copy()
        if add_orig:
            data = pd.concat([data, data_tc[features]], axis=1)
            feat_new.extend(features)
    else:
        data = data_tc[['id', 'eom', 'valid', 'ret_ld1', 'tr_ld0', 'mu_ld0'] + features].copy()
        feat_new = features.copy()

    data['id'] = data['id'].astype(str)
    data['eom'] = pd.to_datetime(data['eom'])
    feat_cons = feat_new + ['constant']

    if scale:
        start_lb = (min(dates) + pd.DateOffset(days=1)) - pd.DateOffset(months=lb + 1) - pd.DateOffset(days=1)
        end_lb = max(dates) + pd.DateOffset(days=1)
        dates_lb = pd.date_range(start=start_lb, end=end_lb, freq='M')
        scales_list = []
        for d_ in dates_lb:
            sigma_data = cov_list.get(d_, None)
            if sigma_data is not None:
                sigma_temp = create_cov(sigma_data)
                if sigma_temp is not None and isinstance(sigma_temp, pd.DataFrame) and sigma_temp.shape[0] > 0:
                    diag_vol = np.sqrt(np.diag(sigma_temp))
                    scales_list.append(pd.DataFrame({'id': sigma_temp.index, 'eom': d_, 'vol_scale': diag_vol}))
        if scales_list:
            scales_df = pd.concat(scales_list, ignore_index=True)

            chunk_size = 25000
            chunked_data = []
            for i in range(0, len(data), chunk_size):
                data_chunk = data.iloc[i:i + chunk_size]
                merged_chunk = data_chunk.merge(scales_df, on=['id', 'eom'], how='left')
                chunked_data.append(merged_chunk)

            data = pd.concat(chunked_data, ignore_index=True)
            data['vol_scale'] = data.groupby('eom')['vol_scale'].transform(lambda x: x.fillna(x.median(skipna=True)))

    if balanced:
        data[feat_new] = data.groupby("eom")[feat_new].transform(lambda x: x - x.mean())
        data["constant"] = 1

        def scale_func(x):
            s = np.sum(x ** 2)
            return x * np.sqrt(1 / s) if s != 0 else x

        data[feat_cons] = data.groupby("eom")[feat_cons].transform(scale_func)

    def process_date(d):
        d = pd.to_datetime(d)
        if (d.year % 10 == 0) and (d.month == 1):
            print(f"--> PF-ML inputs: {d.date()}")
        data_ret = data[(data["valid"] == True) & (data["eom"] == d)]
        ids = data_ret["id"].values
        id_to_index = {str(id_val): idx for idx, id_val in enumerate(ids)}
        r_vec = data_ret["ret_ld1"].values
        sigma = create_cov(cov_list[d], ids=[str(i) for i in ids])
        lambda_val = create_lambda(lambda_list[d], ids=[str(i) for i in ids])
        w_val = wealth.loc[wealth["eom"] == d, "wealth"].values[0]
        rf_val = risk_free.loc[risk_free["date"] == d, "rf"].values[0]
        m_mat = m_func(w=w_val, mu=mu, rf=rf_val, sigma_gam=sigma * gamma_rel,
                       gam=gamma_rel, lambda_mat=lambda_val, iterations=iter)

        start_date = (d.replace(day=1) - pd.DateOffset(months=lb)) - pd.DateOffset(days=1)
        data_sub = data[
            (data['id'].isin(ids)) & (data['eom'] >= start_date) & (data['eom'] <= d) & (data['valid'] == True)].copy()

        if not balanced:
            data_sub[feat_new] = data_sub.groupby("eom")[feat_new].transform(lambda x: x - x.mean())
            data_sub["constant"] = 1
            data_sub[feat_cons] = data_sub.groupby("eom")[feat_cons].transform(
                lambda x: x * np.sqrt(1 / np.sum(x ** 2) if np.sum(x ** 2) != 0 else 1)
            )

        data_sub = data_sub.sort_values(by=["eom", "id"], ascending=[False, True])
        groups = [(grp_date, grp) for grp_date, grp in data_sub.groupby("eom")]
        groups.sort(key=lambda x: x[0], reverse=True)

        n = len(ids)
        p_dim = len(feat_cons)
        signals = {}
        for grp_date, grp in groups:
            s_full = np.zeros((n, p_dim), dtype=float)
            for _, row in grp.iterrows():
                i = id_to_index[row['id']]
                signal_vec = row[feat_cons].values.astype(float)
                if scale:
                    signal_vec *= 1.0 / row['vol_scale']
                s_full[i, :] = signal_vec
            signals[grp_date] = s_full

        gtm_list = []
        for grp_date, grp in groups:
            gt = np.ones(n, dtype=float)
            for _, row in grp.iterrows():
                i = id_to_index[row['id']]
                gt[i] = (1 + row["tr_ld0"]) / (1 + row["mu_ld0"])
            gtm_list.append(m_mat @ np.diag(gt))

        groups_dates = [grp_date for grp_date, _ in groups]
        if len(groups_dates) < lb + 2:
            raise ValueError("Not enough lookback data in data_sub for lb.")
        gtm_agg = {groups_dates[0]: np.eye(n)}
        gtm_agg_l1 = {groups_dates[0]: np.eye(n)}
        for i in range(1, lb + 1):
            gtm_agg[groups_dates[i]] = gtm_agg[groups_dates[i - 1]] @ gtm_list[i - 1]
            gtm_agg_l1[groups_dates[i]] = gtm_agg_l1[groups_dates[i - 1]] @ gtm_list[i]

        omega = np.zeros((n, p_dim))
        const_mat = np.zeros((n, n))
        omega_l1 = np.zeros((n, p_dim))
        const_l1 = np.zeros((n, n))
        for i in range(lb + 1):
            d_new = groups_dates[i]
            d_new_l1 = groups_dates[i + 1]
            omega += gtm_agg[d_new] @ signals[d_new]
            const_mat += gtm_agg[d_new]
            omega_l1 += gtm_agg_l1[d_new] @ signals[d_new_l1]
            const_l1 += gtm_agg_l1[d_new]

        omega = np.linalg.inv(const_mat) @ omega
        omega_l1 = np.linalg.inv(const_l1) @ omega_l1

        grp_d = groups[0][1]
        gt_d = (1 + grp_d["tr_ld0"].values) / (1 + grp_d["mu_ld0"].values)
        omega_chg = omega - (np.diag(gt_d) @ omega_l1)

        r_tilde = (omega.T @ r_vec)
        risk_val = gamma_rel * (omega.T @ sigma @ omega)
        tc_val = w_val * (omega_chg.T @ lambda_val @ omega_chg)
        denom_val = risk_val + tc_val

        # Convert and set indices/columns directly
        r_tilde_pd = pd.Series(r_tilde, index=feat_cons)

        denom_val.index = feat_cons
        denom_val.columns = feat_cons
        denom_pd = denom_val

        risk_val.index = feat_cons
        risk_val.columns = feat_cons
        risk_pd = risk_val

        tc_pd = pd.DataFrame(tc_val, index=feat_cons, columns=feat_cons)

        reals = {"r_tilde": r_tilde_pd, "denom": denom_pd, "risk": risk_pd, "tc": tc_pd}

        signal_t = signals[d] if d in signals else signals[groups_dates[0]]
        signal_t = pd.DataFrame(signal_t, index=id_to_index, columns=feat_cons)

        return d, {"reals": reals, "signal_t": signal_t}

    num_cores = max(1, int(cpu_count() - 1))
    print(f"🔧 Using all available CPU cores: {num_cores}")
    results = Parallel(n_jobs=num_cores)(
        delayed(process_date)(d) for d in tqdm(dates, desc="Step 4. Computing signals and realizations")
    )

    inputs = {d: res for d, res in results}

    return {
        "reals": {str(d): inputs[d]["reals"] for d in inputs},
        "signal_t": {str(d): inputs[d]["signal_t"] for d in inputs},
        "rff_w": rff_w if rff_feat else None
    }