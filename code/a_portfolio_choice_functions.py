import numpy as np
import pandas as pd
from scipy.linalg import sqrtm, solve
from itertools import product
from collections import defaultdict
from functools import reduce
from a_general_functions import create_cov, create_lambda, sigma_gam_adj, initial_weights_new, pf_ts_fun
from a_return_prediction_functions import rff
from b_prepare_data import load_cluster_labels


def m_func(w, mu, rf, sigma_gam, gam, lambda_mat, iterations):
    """
    Computes the m matrix using the iterative process described in the R function.

    Parameters:
        w (float): Weight scalar.
        mu (float): Expected return.
        rf (float): Risk-free rate.
        sigma_gam (ndarray): Covariance matrix scaled by gamma.
        gam (float): Risk aversion coefficient.
        lambda_mat (ndarray): Diagonal matrix of lambdas.
        iterations (int): Number of iterations.

    Returns:
        ndarray: Computed m matrix.
    """
    n = lambda_mat.shape[0]
    g_bar = np.ones(n)
    mu_bar_vec = np.ones(n) * (1 + rf + mu)

    # Compute sigma_gr
    sigma_gr = (1 / (1 + rf + mu) ** 2) * (np.outer(mu_bar_vec, mu_bar_vec) + sigma_gam / gam)

    # Compute lamb_neg05
    lamb_neg05 = np.diag(np.diag(lambda_mat) ** -0.5)

    # Compute x and y
    x = (1 / w) * lamb_neg05 @ sigma_gam @ lamb_neg05
    y = np.diag(1 + np.diag(sigma_gr))

    # Initial sigma_hat and m_tilde
    sigma_hat = x + np.diag(1 + g_bar)
    sigma_hat_squared = sigma_hat @ sigma_hat
    m_tilde = 0.5 * (sigma_hat - sqrtm(sigma_hat_squared - 4 * np.eye(n)))

    # Iterative process
    for _ in range(iterations):
        m_tilde = solve(x + y - m_tilde @ sigma_gr, np.eye(n))

    # Output
    return lamb_neg05 @ m_tilde @ np.sqrt(lambda_mat)


def m_static(sigma_gam, w, lambda_mat, phi):
    """
    Computes the static m matrix.

    Parameters:
        sigma_gam (ndarray): Covariance matrix scaled by gamma.
        w (float): Weight scalar.
        lambda_mat (ndarray): Diagonal matrix of lambdas.
        phi (float): Scaling factor.

    Returns:
        ndarray: Computed static m matrix.
    """
    # Compute the static m matrix
    term = sigma_gam + (w / phi) * lambda_mat
    return solve(term, (w / phi) * lambda_mat)


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
    Implements the Tangency Portfolio (Markowitz-ML).

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
    for d in dates:
        print(d)

        data_sub = data_split.get(d, pd.DataFrame())
        if data_sub.empty:
            continue  # Skip if no data available

        sigma_data = cov_list.get(d, None)
        sigma = sigma_data["fct_cov"] if sigma_data and "fct_cov" in sigma_data else None

        if sigma is None or not isinstance(sigma, (pd.DataFrame, np.ndarray)) or sigma.shape[0] != sigma.shape[1]:
            continue  # Skip if covariance matrix is missing or invalid

        ids = data_sub["id"].unique()
        sigma = sigma.loc[ids, ids] if isinstance(sigma, pd.DataFrame) else None  # Filter sigma to match IDs

        if sigma is None or sigma.shape[0] == 0:
            continue  # Skip if no valid covariance matrix

        pred_ld1 = data_sub.set_index("id")["pred_ld1"].dropna()  # Ensure pred_ld1 aligns with sigma
        sigma = pd.DataFrame(sigma)
        sigma.index = sigma.index.astype(str)
        pred_ld1.index = pred_ld1.index.astype(str)
        pred_ld1 = pred_ld1.loc[sigma.index].to_numpy()  # Align shapes

        if pred_ld1 is None or pred_ld1.size == 0 or sigma.shape[0] != pred_ld1.shape[0]:
            continue  # Skip if sizes don't match

        try:
            w_opt = np.dot(np.linalg.pinv(sigma), pred_ld1) / gam  # Use pseudo-inverse for robustness
        except np.linalg.LinAlgError:
            continue  # Skip if there's a numerical issue

        data_sub["id"] = data_sub["id"].astype(str)
        data_sub = data_sub.loc[data_sub["id"].isin(sigma.index)].assign(w=w_opt)  # Assign weights correctly
        tpf_opt.append(data_sub[['id', 'eom', 'w']])

    if not tpf_opt:
        return {"w": pd.DataFrame(), "pf": pd.DataFrame()}  # Return empty results if no valid weights

    tpf_opt = pd.concat(tpf_opt, ignore_index=True)
    tpf_w = w_fun(data, dates, tpf_opt, wealth)
    tpf_pf = pf_ts_fun(tpf_w, data, wealth, gam)
    tpf_pf['type'] = "Markowitz-ML"

    return {"w": tpf_w, "pf": tpf_pf}


def tpf_cf_fun(data, cf_cluster, er_models, cluster_labels, wealth, gamma_rel, cov_list, dates, seed, features):
    """
    Counterfactual Tangency Portfolio Implementation.

    Parameters:
        data (pd.DataFrame): Main dataset.
        cf_cluster (str): Counterfactual cluster type.
        er_models (list): List of expected return models.
        cluster_labels (pd.DataFrame): Cluster labels for characteristics.
        wealth (pd.DataFrame): Wealth data.
        gamma_rel (float): Risk-aversion parameter.
        cov_list (dict): Dictionary of covariance matrices indexed by date.
        dates (list): List of dates for portfolio implementation.
        seed (int): Random seed for reproducibility.
        features (list): List of feature names to be used.

    Returns:
        pd.DataFrame: Portfolio performance with cluster information.
    """
    np.random.seed(seed)  # Ensures reproducibility

    # Shuffle characteristics and predict counterfactual ER
    if cf_cluster != "bm":
        cf = data.drop(columns=[col for col in data.columns if col.startswith("pred_ld")]).copy()
        cf['id_shuffle'] = cf.groupby('eom')['id'].transform(lambda x: np.random.permutation(x))

        # Select characteristics for the cluster
        chars_sub = cluster_labels.loc[(cluster_labels['cluster'] == cf_cluster) &
                                       (cluster_labels['characteristic'].isin(features)), 'characteristic']

        chars_data = cf[['id_shuffle', 'eom'] + chars_sub.tolist()].rename(columns={'id_shuffle': 'id'})
        cf = cf.drop(columns=chars_sub).merge(chars_data, on=['id', 'eom'], how='left')

        # Predict expected returns
        for m_sub in er_models:
            sub_dates = m_sub['pred']['eom'].unique()
            cf_x = cf.loc[cf['eom'].isin(sub_dates), features].to_numpy()

            # Random Fourier Feature Transformation
            cf_new_x = rff(cf_x, W=m_sub['W'])
            cf_new_x = m_sub['opt_hps']['p'] ** -0.5 * np.hstack([cf_new_x['X_cos'], cf_new_x['X_sin']])

            cf.loc[cf['eom'].isin(sub_dates), 'pred_ld1'] = m_sub['fit'].predict(cf_new_x,
                                                                                 m_sub['opt_hps']['lambda'])
    else:
        cf = data[data['valid'] == True]

    # Implement the tangency portfolio on counterfactual data
    op = tpf_implement(cf, cov_list, wealth, dates, gamma_rel)

    # Add cluster information
    op['pf']['cluster'] = cf_cluster
    return op['pf']


def mv_risky_fun(data, cov_list, wealth, dates, gam, u_vec):
    """
    Mean-variance efficient portfolios of risky assets.

    Parameters:
        data (pd.DataFrame): DataFrame containing data for portfolio computation.
        cov_list (dict): Dictionary of covariance matrices indexed by date.
        wealth (pd.DataFrame): DataFrame containing wealth data.
        dates (list): List of dates for portfolio computation.
        gam (float): Risk-aversion parameter.
        u_vec (list): List of utility values for portfolio construction.

    Returns:
        pd.DataFrame: DataFrame containing portfolio performance across utility values.
    """
    # Get the relevant data subset
    data_rel = data[(data['valid'] == True) & (data['eom'].isin(dates))]
    data_rel = data_rel[['id', 'eom', 'me', 'tr_ld1', 'pred_ld1']].sort_values(['id', 'eom'])

    # Desired weights
    data_split = {date: sub_df for date, sub_df in data_rel.groupby('eom')}

    mv_opt_all = []
    for d in dates:
        data_sub = data_split[d]
        ids = data_sub['id'].values
        sigma_inv = np.linalg.inv(cov_list[str(d)].create_cov(ids=ids))
        er = data_sub['pred_ld1'].values

        # Auxiliary constants
        ones = np.ones(len(er))
        a = float(er.T @ sigma_inv @ er)
        b = float(np.sum(sigma_inv @ er))
        c = float(np.sum(sigma_inv @ ones))
        d_const = a * c - b ** 2

        # Compute weights for each utility value
        for u in u_vec:
            weights = ((c * u - b) / d_const * sigma_inv @ er +
                       (a - b * u) / d_const * sigma_inv @ ones)
            weights_df = pd.DataFrame({'id': data_sub['id'], 'eom': data_sub['eom'],
                                       'u': u * 12, 'w': weights})
            mv_opt_all.append(weights_df)

    mv_opt_all = pd.concat(mv_opt_all, ignore_index=True)

    # Compute portfolios for each utility value
    def portfolio_performance(u_sel):
        w = mv_opt_all[mv_opt_all['u'] == u_sel].drop(columns='u')
        portfolio = pf_ts_fun(w, data, wealth, gam)
        portfolio['u'] = u_sel
        return portfolio

    portfolio_results = pd.concat([portfolio_performance(u) for u in u_vec], ignore_index=True)

    return portfolio_results


def factor_ml_implement(data, wealth, dates, n_pfs, gam):
    """
    Implements a High Minus Low (HML) portfolio strategy using predicted returns.

    Parameters:
        data (pd.DataFrame): Contains portfolio data, including `id`, `eom`, `me` (market equity),
                             `pred_ld1` (predicted returns), and `valid` (filter column).
        wealth (pd.DataFrame or list): Wealth data used for weight adjustments.
        dates (list): List of dates for portfolio construction.
        n_pfs (int): Number of portfolios for ranking stocks (e.g., quintiles or deciles).
        gam (float): Risk-aversion parameter.

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

    # Compute actual portfolio weights
    hml_w = w_fun(data[data["eom"].isin(dates) & data["valid"]], dates, hml_opt, wealth)

    # Compute portfolio performance
    hml_pf = pf_ts_fun(hml_w, data, wealth, gam)
    hml_pf["type"] = "Factor-ML"

    # Output dictionary
    return {"w": hml_w, "pf": hml_pf}


def ew_implement(data, wealth, dates, gamma_rel):
    """
    Equal-weighted (1/N) portfolio implementation.

    Parameters:
        data (pd.DataFrame): DataFrame containing portfolio data with columns such as `id`, `eom` (end of month),
                             and `valid` column indicating valid rows.
        wealth (pd.DataFrame or list): Wealth data required for weight adjustments.
        dates (list): List of dates for which the portfolio is to be constructed.
        gamma_rel (float): Risk-aversion parameter.

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
    ew_pf = pf_ts_fun(ew_w, data, wealth, gamma_rel)
    ew_pf['type'] = "1/N"

    # Return weights and portfolio performance
    return {"w": ew_w, "pf": ew_pf}


def mkt_implement(data, wealth, dates, gamma_rel):
    """
    Market-capitalization weighted portfolio implementation.

    Parameters:
        data (pd.DataFrame): DataFrame containing portfolio data with columns such as `id`, `eom` (end of month),
                             `me` (market equity), and `valid` column indicating valid rows.
        wealth (pd.DataFrame or list): Wealth data required for weight adjustments.
        dates (list): List of dates for which the portfolio is to be constructed.
        gamma_rel (float): Risk-aversion parameter.

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
    mkt_pf = pf_ts_fun(mkt_w, data, wealth, gamma_rel)
    mkt_pf['type'] = "Market"

    # Return weights and portfolio performance
    return {"w": mkt_w, "pf": mkt_pf}


def rw_implement(data, wealth, dates, gamma_rel):
    """
    Rank-weighted portfolio implementation.

    Parameters:
        data (pd.DataFrame): DataFrame containing portfolio data with columns such as `id`, `eom` (end of month),
                             `pred_ld1` (predicted returns), and `valid` column indicating valid rows.
        wealth (pd.DataFrame or list): Wealth data required for weight adjustments.
        dates (list): List of dates for which the portfolio is to be constructed.
        gamma_rel (float): Risk-aversion parameter.

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

        # Center ranks around 0 and normalize sum(abs(weights)) = 2
        data_sub['w'] = (data_sub['rank'] - data_sub['rank'].mean()) * (2 / data_sub['rank'].abs().sum())

        rw_opt.append(data_sub[['id', 'eom', 'w']])

    rw_opt = pd.concat(rw_opt, ignore_index=True)

    # Compute actual weights
    rw_w = w_fun(data_valid, dates, rw_opt, wealth)

    # Compute portfolio performance
    rw_pf = pf_ts_fun(rw_w, data, wealth, gamma_rel)
    rw_pf['type'] = "Rank-ML"

    # Return weights and portfolio performance
    return {"w": rw_w, "pf": rw_pf}


def mv_implement(data, cov_list, wealth, dates, gamma_rel):
    """
    Minimum-variance portfolio implementation.

    Parameters:
        data (pd.DataFrame): Portfolio data.
        cov_list (dict): Dictionary of covariance matrices indexed by dates.
        wealth (pd.DataFrame or list): Wealth data.
        dates (list): List of portfolio construction dates.
        gamma_rel (float): Risk-aversion parameter.

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
        ids = data_sub['id'].astype(str).values  # Convert to str

        # Ensure `d` is in the correct format for `cov_list`
        sigma_data = cov_list.get(d, None)
        if sigma_data is None or "fct_cov" not in sigma_data:
            print(f"Warning: No valid covariance matrix found for {d}")
            continue  # Skip if no valid covariance matrix

        sigma = sigma_data["fct_cov"]

        # Ensure sigma is a DataFrame
        if isinstance(sigma, pd.DataFrame):
            sigma.index = sigma.index.astype(str)  # Ensure index is str
            sigma.columns = sigma.columns.astype(str)  # Ensure columns are str
            sigma = sigma.loc[sigma.index.intersection(ids), sigma.columns.intersection(ids)]  # Align with ids
        elif isinstance(sigma, np.ndarray):
            sigma = pd.DataFrame(sigma, index=ids, columns=ids)  # Convert ndarray to DataFrame

        # Check if sigma is still valid
        if sigma is None or sigma.shape[0] != sigma.shape[1] or sigma.shape[0] == 0:
            print(f"Warning: Invalid covariance matrix for {d}")
            continue  # Skip invalid covariance matrix

        # Use pseudo-inverse for numerical stability
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
    mv_pf = pf_ts_fun(mv_w, data, wealth, gamma_rel)
    mv_pf['type'] = "Minimum Variance"

    return {"w": mv_w, "pf": mv_pf}


def static_val_fun(data, dates, cov_list, lambda_list, wealth, cov_type, gamma_rel, k=None, g=None, u=None, hps=None):
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
    static_weights = static_weights.merge(data[['id', 'eom', 'tr_ld1', 'pred_ld1']], on=['id', 'eom'], how='left')
    static_weights = static_weights.merge(wealth[['eom', 'mu_ld1', 'wealth']], on='eom', how='left')

    # Iterate over each date
    for d in dates:
        # If hyperparameters are provided, use them
        if hps is not None:
            recent_hps = hps[(hps['eom_ret'] < d) & (hps['eom_ret'] == hps[hps['eom_ret'] < d]['eom_ret'].max())]
            g = recent_hps['g'].iloc[0]
            u = recent_hps['u'].iloc[0]
            k = recent_hps['k'].iloc[0]

        # Extract required inputs for the current date
        wealth_t = static_weights.loc[static_weights['eom'] == d, 'wealth'].iloc[0]
        ids = static_weights.loc[static_weights['eom'] == d, 'id'].astype(str).values

        # Retrieve sigma_gam from cov_list
        sigma_data = cov_list.get(d)
        if sigma_data is None or "fct_cov" not in sigma_data:
            print(f"Warning: Missing 'fct_cov' for {d}")
            continue

        # Convert fct_cov to DataFrame and filter by ids
        sigma_gam = pd.DataFrame(sigma_data["fct_cov"])
        sigma_gam.index = sigma_gam.index.astype(str)
        sigma_gam.columns = sigma_gam.columns.astype(str)
        sigma_gam = sigma_gam.loc[sigma_gam.index.intersection(ids), sigma_gam.columns.intersection(ids)] * gamma_rel

        # Apply adjustments
        sigma_gam = sigma_gam_adj(sigma_gam, g=g, cov_type=cov_type)

        # Retrieve lambda_matrix from lambda_list
        lambda_data = lambda_list.get(d)
        if lambda_data is None or "lambda" not in lambda_data:
            print(f"Warning: Missing 'lambda' for {d}")
            continue

        # Convert lambda to DataFrame and filter by ids
        lambda_matrix = pd.DataFrame(lambda_data["lambda"])
        lambda_matrix.index = lambda_matrix.index.astype(str)
        lambda_matrix.columns = lambda_matrix.columns.astype(str)
        lambda_matrix = lambda_matrix.loc[
                            lambda_matrix.index.intersection(ids), lambda_matrix.columns.intersection(ids)] * k

        # Calculate weights
        pred_ld1 = static_weights.loc[static_weights['eom'] == d, 'pred_ld1'].values
        w_start = static_weights.loc[static_weights['eom'] == d, 'w_start'].values
        weights = np.linalg.solve(
            sigma_gam + wealth_t * lambda_matrix,
            pred_ld1 * u + wealth_t * lambda_matrix @ w_start
        )
        static_weights.loc[static_weights['eom'] == d, 'w'] = weights

        # Update weights for the next month
        next_month_idx = dates.index(d) + 1
        if next_month_idx < len(dates):
            next_month = dates[next_month_idx]
            transition_weights = static_weights.loc[static_weights['eom'] == d].copy()
            transition_weights['eom'] = next_month
            transition_weights['w_start'] = (
                transition_weights['w'] * (1 + transition_weights['tr_ld1']) /
                (1 + transition_weights['mu_ld1'])
            )
            static_weights = static_weights.merge(
                transition_weights[['id', 'eom', 'w_start']],
                on=['id', 'eom'],
                how='left'
            )
            static_weights['w_start'].fillna(0, inplace=True)

    return static_weights


def static_implement(data_tc, cov_list, lambda_list,
                     wealth, gamma_rel,
                     dates_oos, dates_hp,
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
        for i, hprow in static_hps.iterrows():
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
            pf_results = pf_ts_fun(static_w, data_tc, wealth, gamma_rel)
            pf_results['hp_no'] = i
            pf_results['k'] = k
            pf_results['u'] = u
            pf_results['g'] = g
            validation_list.append(pf_results)

        validation = pd.concat(validation_list, ignore_index=True)

    # Sort and compute cumulative metrics
    validation = validation.sort_values(by=['hp_no', 'eom_ret'])
    validation['cum_var'] = validation.groupby('hp_no')['r'].transform(lambda x: x.expanding().var())
    validation['cum_obj'] = (validation.groupby('hp_no')
                                      .apply(lambda df: (df['r'] - df['tc'] - 0.5*df['cum_var']*gamma_rel)
                                                      .expanding().mean())
                                      .reset_index(level=0, drop=True))
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
        hps=optimal_hps  # your static_val_fun presumably supports passing a DataFrame of best HPs
    )
    pf = pf_ts_fun(w, data_tc, wealth, gamma_rel)
    pf['type'] = "Static-ML*"

    return {
        "hps": validation,
        "best_hps": optimal_hps,
        "w": w,
        "pf": pf
    }


def pfml_input_fun(data_tc, cov_list, lambda_list, gamma_rel, wealth, mu, dates, lb, scale,
                   risk_free, features, rff_feat, seed, p_max, g, add_orig, iter_, balanced):
    """
    Prepares inputs for Portfolio-ML computations.

    Parameters:
        data_tc (pd.DataFrame): Data containing features and target variables for portfolio construction.
        cov_list (dict): Dictionary of covariance matrices indexed by date.
        lambda_list (dict): Dictionary of lambda matrices indexed by date.
        gamma_rel (float): Relative risk aversion parameter.
        wealth (pd.DataFrame): Wealth data containing columns ['eom', 'wealth'].
        mu (float): Expected return parameter.
        dates (list): List of dates for portfolio computation.
        lb (int): Lookback window for data preparation.
        scale (bool): Whether to scale features by volatility.
        risk_free (pd.DataFrame): Risk-free rate data with columns ['eom', 'rf'].
        features (list): List of feature names for portfolio prediction.
        rff_feat (bool): Whether to use Random Fourier Features.
        seed (int): Random seed for reproducibility.
        p_max (int): Maximum number of random Fourier features.
        g (float): RFF Gaussian kernel width.
        add_orig (bool): Whether to include original features with RFF.
        iter_ (int): Iterations for weight computation.
        balanced (bool): Whether to balance data by scaling columns.

    Returns:
        dict: A dictionary containing:
            - "reals": Realizations for r_tilde, denominator, risk, and transaction cost.
            - "signal_t": Signals for each time period.
            - "rff_w" (optional): Random Fourier feature weights (if rff_feat=True).
    """
    # Define lookback dates
    dates_lb = pd.date_range(
        start=min(dates) - pd.DateOffset(months=lb + 1),
        end=max(dates),
        freq='M'
    )

    # Define rff_x as None by default so it is always in scope
    rff_x = None

    # Create random Fourier features if required
    if rff_feat:
        np.random.seed(seed)
        rff_x = np.random.randn(len(features), p_max)
        rff_cos = np.cos(np.dot(data_tc[features], rff_x) / g)
        rff_sin = np.sin(np.dot(data_tc[features], rff_x) / g)

        rff_features = np.hstack([rff_cos, rff_sin])
        rff_columns = [f"rff{i}_cos" for i in range(1, p_max // 2 + 1)] + \
                      [f"rff{i}_sin" for i in range(1, p_max // 2 + 1)]
        rff_df = pd.DataFrame(rff_features, columns=rff_columns)

        data = pd.concat(
            [data_tc[['id', 'eom', 'valid', 'ret_ld1', 'tr_ld0', 'mu_ld0']], rff_df],
            axis=1
        )
        feat_new = rff_columns

        if add_orig:
            data = pd.concat([data, data_tc[features]], axis=1)
            feat_new.extend(features)
    else:
        data = data_tc[['id', 'eom', 'valid', 'ret_ld1', 'tr_ld0', 'mu_ld0'] + features]
        feat_new = features

    # We'll add 'constant' so we can do a group-based transformation
    feat_cons = feat_new + ['constant']

    # Scale features if requested
    if scale:
        scales = []
        # Build a monthly DataFrame of vol scaling
        for d in dates_lb:
            sigma = cov_list.get(str(d), None)
            if sigma is not None:
                diag_vol = np.sqrt(np.diag(sigma))
                scales.append(
                    pd.DataFrame({'id': np.arange(len(diag_vol)), 'eom': d, 'vol_scale': diag_vol})
                )
        scales_df = pd.concat(scales, ignore_index=True) if len(scales) > 0 else pd.DataFrame()
        data = data.merge(scales_df, on=['id', 'eom'], how='left')
        # Fill missing scales with the median for that eom group
        data['vol_scale'] = data.groupby('eom')['vol_scale'].apply(lambda x: x.fillna(x.median()))

    # If balanced, we want to demean within each date and then scale so norm=1
    if balanced:
        data[feat_new] = data.groupby('eom')[feat_new].transform(lambda x: x - x.mean())
        data['constant'] = 1
        data[feat_cons] = data.groupby('eom')[feat_cons].transform(
            lambda x: x * np.sqrt(1.0 / (x**2).sum())
        )

    # Prepare signals and realizations
    inputs = {}
    for d in dates:
        if pd.Timestamp(d).year % 10 == 0 and pd.Timestamp(d).month == 1:
            print(f"--> PF-ML inputs: {d}")

        # Subset for returns
        data_ret = data[(data['valid']) & (data['eom'] == d)][['id', 'ret_ld1']]
        ids = data_ret['id'].values
        r = data_ret['ret_ld1'].values

        sigma = cov_list[str(d)]
        lambda_mat = lambda_list[str(d)]
        w = wealth.loc[wealth['eom'] == d, 'wealth'].iloc[0]
        rf = risk_free.loc[risk_free['eom'] == d, 'rf'].iloc[0]

        # Some function m_func used for weighting, presumably defined elsewhere
        m = m_func(
            w=w, mu=mu, rf=rf,
            sigma_gam=sigma * gamma_rel,
            gam=gamma_rel, lambda_mat=lambda_mat,
            iterations=iter_
        )

        # Subset data for signals over lookback period
        data_sub = data[
            (data['id'].isin(ids)) &
            (data['eom'] >= pd.Timestamp(d) - pd.DateOffset(months=lb)) &
            (data['eom'] <= pd.Timestamp(d))
        ]
        # If not balanced, do the same transformations that balanced does
        if not balanced:
            data_sub[feat_new] = data_sub.groupby('eom')[feat_new].transform(lambda x: x - x.mean())
            data_sub['constant'] = 1
            data_sub[feat_cons] = data_sub.groupby('eom')[feat_cons].transform(
                lambda x: x * np.sqrt(1.0 / (x**2).sum())
            )

        data_sub = data_sub.sort_values(['eom', 'id'], ascending=[False, True])

        # Build signals dictionary by date
        signals = {}
        for d_new in data_sub['eom'].unique():
            s = data_sub[data_sub['eom'] == d_new][feat_cons].values
            if scale:
                # Scale by vol_scale
                s_scales = np.diag(1.0 / data_sub[data_sub['eom'] == d_new]['vol_scale'].values)
                s = s_scales @ s
            signals[d_new] = s

        # Weighted signals (omega)
        gtm = {}
        for d_new in signals.keys():
            gt = (1.0 + data_sub[data_sub['eom'] == d_new]['tr_ld0'].values) / \
                 (1.0 + data_sub[data_sub['eom'] == d_new]['mu_ld0'].values)
            gt[np.isnan(gt)] = 1.0
            gtm[d_new] = m @ np.diag(gt)

        omega = np.sum([gtm[d_new] @ signals[d_new] for d_new in signals.keys()], axis=0)

        # Realizations
        r_tilde = omega.T @ r
        risk = gamma_rel * (omega.T @ sigma @ omega)
        tc = w * (omega.T @ lambda_mat @ omega)
        denom = risk + tc

        inputs[d] = {
            'reals': {
                'r_tilde': r_tilde,
                'denom': denom,
                'risk': risk,
                'tc': tc
            },
            'signal_t': signals
        }

    return {
        'reals':  {str(d): inputs[d]['reals']   for d in dates},
        'signal_t': {str(d): inputs[d]['signal_t'] for d in dates},
        # rff_x is guaranteed to be defined here (either None or the RFF array)
        'rff_w': rff_x if rff_feat else None
    }



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
        train (list): A list of dictionaries, each containing a 'denom' matrix.

    Returns:
        numpy.ndarray: Sum of all 'denom' matrices in the list.
    """
    denom_sum = sum(x['denom'] for x in train)
    return denom_sum


def pfml_search_coef(pfml_input, p_vec, l_vec, hp_years, orig_feat, features):
    """
    Hyperparameter search for best coefficients in Portfolio-ML.

    Parameters:
        pfml_input (dict): Input containing 'reals' with realizations and denominators.
        p_vec (list): List of numbers of random Fourier features (RFF) to evaluate.
        l_vec (list): List of regularization parameters (lambdas) to evaluate.
        hp_years (list): List of hyperparameter evaluation years.
        orig_feat (bool): Whether to include original features.
        features (list): List of original feature names.

    Returns:
        dict: Coefficients for each hyperparameter combination across years.
    """
    from collections import defaultdict

    # Extract all dates and initialize variables
    d_all = np.array([np.datetime64(k) for k in pfml_input['reals'].keys()])
    end_bef = min(hp_years) - 1
    train_bef = {
        k: v for k, v in pfml_input['reals'].items()
        if np.datetime64(k) < np.datetime64(f"{end_bef - 1}-12-31")
    }

    # Compute initial sums
    r_tilde_sum = np.sum([v['r_tilde'] for v in train_bef.values()], axis=0)
    denom_raw_sum = denom_sum_fun(list(train_bef.values()))
    n = len(train_bef)

    # Prepare output container
    coef_list = defaultdict(dict)
    hp_years = sorted(hp_years)

    for year in hp_years:
        # Extract training data for current year
        train_new = {
            k: v for k, v in pfml_input['reals'].items()
            if np.datetime64(f"{year - 2}-12-31") <= np.datetime64(k) <= np.datetime64(f"{year - 1}-11-30")
        }

        # Update counts and sums
        n += len(train_new)
        r_tilde_sum += np.sum([v['r_tilde'] for v in train_new.values()], axis=0)
        denom_raw_sum += denom_sum_fun(list(train_new.values()))

        # Compute coefficients for each p and lambda
        for p in p_vec:
            feat_p = pfml_feat_fun(p, orig_feat, features)
            r_tilde_sub = r_tilde_sum[feat_p] / n
            denom_sub = denom_raw_sum[np.ix_(feat_p, feat_p)] / n

            coef_list[year][p] = {
                l: np.linalg.solve(denom_sub + l * np.eye(len(feat_p)), r_tilde_sub)
                for l in l_vec
            }

    return coef_list


def pfml_hp_reals_fun(pfml_input, hp_coef, p_vec, l_vec, hp_years, orig_feat, features):
    """
    Compute realized utility for each p and lambda in each hyperparameter evaluation year.

    Parameters:
        pfml_input (dict): Input containing 'reals' with realizations and denominators.
        hp_coef (dict): Hyperparameter coefficients from `pfml_search_coef`.
        p_vec (list): List of numbers of random Fourier features (RFF) to evaluate.
        l_vec (list): List of regularization parameters (lambdas) to evaluate.
        hp_years (list): List of hyperparameter evaluation years.
        orig_feat (bool): Whether to include original features.
        features (list): List of original feature names.

    Returns:
        pd.DataFrame: Validation results with cumulative objectives and ranks for each combination.
    """
    validation = []

    for end in hp_years:
        # Filter realizations for the current year
        reals_all = {
            k: v for k, v in pfml_input['reals'].items()
            if np.datetime64(f"{end - 1}-12-31") <= np.datetime64(k) <= np.datetime64(f"{end}-11-30")
        }

        coef_list_yr = hp_coef[str(end)]

        for p in p_vec:
            # Generate features for the given p
            feat_p = pfml_feat_fun(p, orig_feat, features)
            coef_list_p = coef_list_yr[str(p)]

            # Extract relevant realizations
            reals = {
                nm: {
                    "r_tilde": x["r_tilde"][feat_p],
                    "denom": x["denom"][np.ix_(feat_p, feat_p)]
                }
                for nm, x in reals_all.items()
            }

            for i, l in enumerate(l_vec):
                coef = coef_list_p[l]
                objs = []

                for nm, x in reals.items():
                    # Compute objective value
                    r = (x["r_tilde"].T @ coef - 0.5 * coef.T @ x["denom"] @ coef).item()
                    objs.append({
                        "eom": np.datetime64(nm),
                        "eom_ret": np.datetime64(nm) + np.timedelta64(1, 'M'),
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



def pfml_aims_fun(pfml_input, validation, data_tc, hp_coef, hp_years, dates_oos, l_vec, orig_feat, features):
    """
    Create the optimal aim portfolio for each out-of-sample date.

    Parameters:
        pfml_input (dict): Input containing signals and realizations.
        validation (pd.DataFrame): Validation results with hyperparameter rankings.
        data_tc (pd.DataFrame): Dataframe containing stock data.
        hp_coef (dict): Coefficients for hyperparameters.
        hp_years (list): List of hyperparameter evaluation years.
        dates_oos (list): Out-of-sample dates for portfolio implementation.
        l_vec (list): List of lambda values.
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
        s = pfml_input['signal_t'][str(d)][:, feat]

        # Match lambda and extract coefficients
        l_no = l_vec.index(hps_d['l'].iloc[0])
        coef = hp_coef[str(oos_year)][str(hps_d['p'].iloc[0])][l_no]

        # Calculate aim portfolio weights
        data_subset = data_tc[(data_tc['valid'] == True) & (data_tc['eom'] == d)]
        aim_pf = data_subset[['id', 'eom']].copy()
        aim_pf['w_aim'] = np.dot(s, coef)

        aim_pfs_list[d] = {"aim_pf": aim_pf, "coef": coef}

    return aim_pfs_list


def pfml_w(
    data: pd.DataFrame,
    dates: list,
    cov_list: dict,
    lambda_list: dict,
    gamma_rel: float,
    iter_: int,
    risk_free: pd.DataFrame,
    wealth: pd.DataFrame,
    mu: float,
    aims: pd.DataFrame = None,
    signal_t: dict = None,
    aim_coef: any = None
) -> pd.DataFrame:
    """
    Compute Portfolio-ML weights, mirroring the R code logic.

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

    # 1. If no 'aims', build them using signal_t and aim_coef (like the R code).
    if aims is None:
        aim_list = []
        for d in dates:
            d_str = str(pd.Timestamp(d))  # string key for dictionaries
            data_d = data.loc[data['eom'] == pd.Timestamp(d), ['id','eom']].copy()
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

    # Merge required columns from data and wealth
    # So we can update weights across months
    fa_weights = fa_weights.merge(
        data[['id','eom','tr_ld1']], on=['id','eom'], how='left'
    )
    fa_weights = fa_weights.merge(
        wealth[['eom','mu_ld1']], on='eom', how='left'
    )

    # Ensure we have a starting weight column
    if 'w_start' not in fa_weights.columns:
        # Assume w_start = w if we already have 'w'
        # or zero if neither exist
        if 'w' in fa_weights.columns:
            fa_weights['w_start'] = fa_weights['w'].fillna(0)
        else:
            fa_weights['w_start'] = 0.0

    # 3. Compute weights iteratively for each date
    for idx, d in enumerate(dates):
        cur_date = pd.Timestamp(d)
        # Filter to the stocks for this date
        ids_d = data.loc[data['eom'] == cur_date, 'id']

        # Covariance & Lambda
        sigma = create_cov(cov_list[str(d)], ids=ids_d)
        lambda_mat = create_lambda(lambda_list[str(d)], ids=ids_d)

        # Wealth & risk-free
        w_val = wealth.loc[wealth['eom'] == cur_date, 'wealth'].iloc[0]
        rf_val = risk_free.loc[risk_free['eom'] == cur_date, 'rf'].iloc[0]

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
        w_cur = aims.loc[aims['eom'] == cur_date, ['id','eom','w_aim']].merge(
            fa_weights.loc[fa_weights['eom'] == cur_date],
            on=['id','eom'], how='left'
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



def pfml_implement(
    data_tc,
    cov_list,
    lambda_list,
    risk_free,
    features,
    wealth,
    mu,
    gamma_rel,
    dates_full,
    dates_oos,
    lb,
    hp_years,
    rff_feat,
    scale,
    g_vec=None,
    p_vec=None,
    l_vec=None,
    orig_feat=None,
    iter=100,
    hps=None,
    balanced=False,
    seed=None
):
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

    # 1. Hyperparameter search if none provided
    if hps is None:
        hps = {}
        for g in g_vec:
            print(f"Processing g: {g}")
            pfml_input = pfml_input_fun(
                data_tc, cov_list, lambda_list,
                gamma_rel, wealth, mu, dates_full,
                lb, scale, risk_free, features,
                rff_feat, seed, p_max=max(p_vec),
                g=g, add_orig=orig_feat,
                iter_=iter, balanced=balanced
            )
            rff_w = pfml_input.get("rff_w")

            # Restrict "reals" and "signal_t" to final set of features
            feat_all = pfml_feat_fun(max(p_vec), orig_feat, features)
            reals_adj = {}
            for date_key, val in pfml_input["reals"].items():
                reals_adj[date_key] = {
                    "r_tilde": val["r_tilde"][feat_all]
                }
                for subk, mat_ in val.items():
                    if subk != "r_tilde":
                        reals_adj[date_key][subk] = mat_[feat_all][:, feat_all]

            signals_adj = {
                date_key: mat_[feat_all]
                for date_key, mat_ in pfml_input["signal_t"].items()
            }
            pfml_input = {
                "reals": reals_adj,
                "signal_t": signals_adj
            }

            # Hyperparam search and validation
            pfml_hp_coef = pfml_search_coef(
                pfml_input, p_vec, l_vec, hp_years,
                orig_feat, features
            )
            validation = pfml_hp_reals_fun(
                pfml_input, pfml_hp_coef, p_vec, l_vec,
                hp_years, orig_feat, features
            )
            validation["g"] = g

            # Aim portfolios
            aims = pfml_aims_fun(
                pfml_input, validation, data_tc,
                pfml_hp_coef, hp_years, dates_oos,
                l_vec, orig_feat, features
            )
            hps[g] = {
                "aim_pfs_list": aims,
                "validation": validation,
                "rff_w": rff_w
            }

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
        l_val = hp_sel["l"].iloc[0]

        aim_pf = hps[g_val]["aim_pfs_list"][d]["aim_pf"]
        coef_ = hps[g_val]["aim_pfs_list"][d]["coef"]
        best_hps_list.append({"g": g_val, "p": p_val, "aim": aim_pf, "coef": coef_})

    aims = pd.concat([x["aim"] for x in best_hps_list], ignore_index=True)

    # 4. Final portfolio weights
    # Make sure we pass all arguments that pfml_w expects:
    w = pfml_w(
        data=data_tc,
        dates=dates_oos,
        cov_list=cov_list,
        lambda_list=lambda_list,
        gamma_rel=gamma_rel,
        iter_=iter,
        risk_free=risk_free,
        wealth=wealth,
        mu=mu,
        # If pfml_w requires these:
        signal_t=None,   # or your actual signals if needed
        aim_coef=None,   # or your actual aim coefficients if needed
        aims=aims
    )

    pf = pf_ts_fun(w, data_tc, wealth, gamma_rel)
    pf["type"] = "Portfolio-ML"

    # 5. RFF weight list
    rff_w_list = {g: hps[g]["rff_w"] for g in hps}

    return {
        "hps": hps,
        "best_hps": best_hps,
        "best_hps_list": best_hps_list,
        "aims": aims,
        "w": w,
        "pf": pf,
        "rff_w_list": rff_w_list
    }


def pfml_cf_fun(data, cf_cluster, pfml_base, dates, cov_list, lambda_list, scale, orig_feat, gamma_rel, wealth,
                risk_free, mu, iter, seed, features, cluster_labels):
    """
    Portfolio-ML implementation with counterfactual inputs.

    Parameters:
        data (pd.DataFrame): Portfolio dataset with required features and returns.
        cf_cluster (str): Counterfactual cluster identifier ("bm" for benchmark or cluster name).
        pfml_base (dict): Base Portfolio-ML implementation containing hyperparameters and aim coefficients.
        dates (list): List of dates for the portfolio implementation.
        cov_list (dict): Covariance matrices indexed by date.
        lambda_list (dict): Lambda matrices indexed by date.
        scale (bool): Whether to scale features by their volatility.
        orig_feat (bool): Whether to include original features in the implementation.
        gamma_rel (float): Relative risk aversion parameter.
        wealth (pd.DataFrame): Wealth data by date.
        risk_free (pd.DataFrame): Risk-free rate data by date.
        mu (float): Drift parameter for portfolio adjustment.
        iter (int): Number of iterations for matrix computations.
        seed (int): Seed for reproducibility.
        features (list): List of feature names for the input data.
        cluster_labels (pd.DataFrame): Pre-loaded cluster labels, including clusters and their respective characteristics.

    Returns:
        pd.DataFrame: Counterfactual portfolio performance metrics.
    """

    # Step 1: Handle scaling if required
    if scale:
        scales = []
        for d in dates:
            sigma_arr = create_cov(cov_list[str(d)])
            sigma = pd.DataFrame(sigma_arr)

            diag_vol = np.sqrt(np.diag(sigma.values))

            scales.append(
                pd.DataFrame(
                    {'id': sigma.index, 'eom': d, 'vol_scale': diag_vol}
                )
            )
        scales = pd.concat(scales, ignore_index=True)
        data = data.merge(scales, on=['id', 'eom'], how='left')
        data['vol_scale'] = data.groupby('eom')['vol_scale'].apply(lambda x: x.fillna(x.median()))

    # Step 2: Prepare counterfactual data
    np.random.seed(seed)
    cf = data[['id', 'eom', 'vol_scale'] + features].copy()
    if cf_cluster != "bm":
        cf['id_shuffle'] = cf.groupby('eom')['id'].transform(lambda x: np.random.permutation(x.values))
        chars_sub = cluster_labels.loc[
            (cluster_labels['cluster'] == cf_cluster) & (cluster_labels['characteristic'].isin(features)),
            'characteristic'
        ]
        cf = cf.drop(columns=chars_sub).merge(
            cf[['id_shuffle', 'eom'] + list(chars_sub)],
            left_on=['id_shuffle', 'eom'],
            right_on=['id', 'eom'],
            suffixes=('', '_shuffle')
        ).drop(columns=['id_shuffle'])

    # Step 3: Compute counterfactual aim portfolios
    aim_cf = []
    for d in dates:
        stocks = cf.loc[cf['eom'] == d, 'id']
        best_g = pfml_base['best_hps_list'][d]['g']
        best_p = pfml_base['best_hps_list'][d]['p']
        aim_coef = pfml_base['best_hps_list'][d]['coef']
        W = pfml_base['hps'][str(best_g)]['rff_w'][:, :best_p // 2]

        # Random Fourier features for counterfactual data
        rff_x = rff(cf.loc[cf['eom'] == d, features], W=W)
        s = np.hstack([rff_x['X_cos'], rff_x['X_sin']])
        s = (s - s.mean(axis=0)) * np.sqrt(1 / np.sum(s ** 2, axis=0))
        s = np.hstack([s, np.ones((s.shape[0], 1))])

        # Reorder to align with coefficients
        feat = pfml_feat_fun(p=best_p, orig_feat=orig_feat, features=features)
        s = s[:, [feat.index(f) for f in feat]]

        if scale:
            scale_diag = np.diag(1 / cf.loc[cf['eom'] == d, 'vol_scale'])
            s = scale_diag @ s

        # Compute weights
        aim_cf.append(pd.DataFrame({
            'id': stocks,
            'eom': d,
            'w_aim': s @ aim_coef
        }))
    aim_cf = pd.concat(aim_cf, ignore_index=True)

    # Step 4: Compute counterfactual portfolio weights
    w_cf = pfml_w(data, dates, cov_list, lambda_list, gamma_rel, iter, risk_free, wealth, mu, aim_cf)
    pf_cf = pf_ts_fun(w_cf, data, wealth, gamma_rel)
    pf_cf['type'] = "Portfolio-ML"
    pf_cf['cluster'] = cf_cluster

    return pf_cf


def mp_aim_fun(preds, sigma_gam, lambda_, m, rf, mu, w, K):
    """
    Compute the multiperiod-ML aim portfolio.

    Parameters:
        preds (dict): Predictions for different horizons, keyed by "pred_ld[tau]".
        sigma_gam (np.ndarray): Adjusted covariance matrix scaled by gamma.
        lambda_ (np.ndarray): Regularization matrix.
        m (np.ndarray): Transition matrix for weights.
        rf (float): Risk-free rate.
        mu (float): Drift parameter.
        w (float): Current wealth.
        K (int): Prediction horizon.

    Returns:
        np.ndarray: Multiperiod-ML aim portfolio weights.
    """
    iden = np.eye(sigma_gam.shape[0])
    sigma_inv = np.linalg.inv(sigma_gam)
    g_bar = np.eye(lambda_.shape[0])
    m_gbar = m @ g_bar
    c = (w**-1) * m @ np.linalg.inv(lambda_) @ sigma_gam
    c_tilde = np.linalg.inv(iden - m_gbar) @ (c @ sigma_inv)

    aim_pf = 0
    m_gbar_pow = iden
    for tau in range(1, K + 1):
        if tau > 1:
            m_gbar_pow = m_gbar_pow @ m_gbar
        aim_pf += m_gbar_pow @ c_tilde @ preds[f"pred_ld{tau}"]

    aim_pf_rescaled = np.linalg.inv(iden - m) @ (aim_pf)
    return aim_pf_rescaled


def mp_val_fun(data, dates, cov_list, lambda_list, wealth, risk_free, mu, gamma_rel, cov_type, iter_, K,
               k=None, g=None, u_vec=None, hps=None, verbose=True):
    """
    Multiperiod-ML validation function.

    Parameters:
        data (pd.DataFrame): Input dataset with required fields.
        dates (list): List of dates for validation.
        cov_list (dict): Covariance matrices indexed by date.
        lambda_list (dict): Lambda matrices indexed by date.
        wealth (pd.DataFrame): Wealth data by date.
        risk_free (pd.DataFrame): Risk-free rate data by date.
        mu (float): Drift parameter.
        gamma_rel (float): Relative risk-aversion parameter.
        cov_type (str): Covariance type for adjustments.
        iter_ (int): Number of iterations for convergence.
        K (int): Prediction horizon.
        k (float, optional): Regularization scaling parameter.
        g (float, optional): Covariance adjustment scaling parameter.
        u_vec (list, optional): List of utility parameters.
        hps (pd.DataFrame, optional): Hyperparameters for optimization.
        verbose (bool, optional): Whether to print progress information.

    Returns:
        pd.DataFrame or dict: Validation weights by utility parameter or combined results.
    """
    # Initialize weights for each utility parameter
    w_init = data.copy()
    w_init['w_start'] = 0  # Initialize with zero weights
    if hps is not None:
        w_list = {"opt": w_init}
    else:
        w_list = {str(u): w_init.copy() for u in u_vec}

    # Initialize last valid hyperparameters
    last_u, last_g, last_k = None, None, None

    for d in dates:
        d_ts = pd.to_datetime(d)

        # Print verbose information
        if verbose and (d_ts.year % 10 == 0) and (d_ts.month == 12):
            print(f"Processing date: {d}")

        # Extract g, u, k from hps if available, else reuse last valid
        if hps is not None:
            hps_sub = hps[hps['eom_ret'].dt.year < d_ts.year]
            if not hps_sub.empty:
                hp_row = hps_sub.loc[hps_sub['eom_ret'].idxmax()]
                g, u, k = hp_row['g'], hp_row['u'], hp_row['k']
                last_g, last_u, last_k = g, u, k  # Update last valid values
            else:
                # Reuse last valid hyperparameters
                if last_g is None or last_u is None or last_k is None:
                    raise ValueError("No valid hyperparameters found in `hps` and no previous values to retain.")
                g, u, k = last_g, last_u, last_k
        else:
            # If no hps, use last_u or raise error for the first iteration
            if last_u is None:
                raise ValueError("No `hps` provided, and `u` is not initialized.")
            u = last_u

        # Gather inputs
        wealth_t = wealth.loc[wealth['eom'] == d_ts, 'wealth'].values[0]
        rf = risk_free.loc[risk_free['eom'] == d_ts, 'rf'].values[0]
        data_d = data[data['eom'] == d_ts].copy()

        # Adjust covariance and lambda
        sigma_raw = create_cov(cov_list[str(d)], ids=data_d['id'])
        sigma_gam = sigma_raw * gamma_rel
        sigma_gam = sigma_gam_adj(sigma_gam, g, cov_type)
        lambda_raw = create_lambda(lambda_list[str(d)], ids=data_d['id'])
        lambda_ = lambda_raw * k

        # Compute m matrix
        m = m_func(w=wealth_t, mu=mu, rf=rf, sigma_gam=sigma_gam, gam=gamma_rel, lambda_mat=lambda_, iterations=iter_)
        iden = np.eye(m.shape[0])

        # Compute aim portfolio (u=1)
        w_aim_one = mp_aim_fun(data_d, sigma_gam=sigma_gam, lambda_=lambda_, m=m, rf=rf, mu=mu, w=wealth_t, K=K)

        # Update weights for each u
        for u_key, df_w in w_list.items():
            # Determine utility value
            u_val = u if hps is not None else float(u_key)

            # Build aim portfolio for this u
            fa_aims = data_d[['id', 'eom']].copy()
            fa_aims['w_aim'] = w_aim_one['w_aim'] * u_val

            # Merge and compute weights
            w_cur = pd.merge(fa_aims, df_w[df_w['eom'] == d_ts], on=['id', 'eom'], how='left')
            w_cur['w_opt'] = (
                m @ w_cur['w_start'].fillna(0).values
            ) + (
                (iden - m) @ w_cur['w_aim'].values
            )

            # Update for the next month
            next_month_idx = dates.index(d) + 1
            next_month = dates[next_month_idx] if next_month_idx < len(dates) else None

            # Insert weights for the next month
            w_update = w_cur[['id', 'w_opt']].copy()
            if next_month:
                w_cur['w_opt_ld1'] = (
                    w_cur['w_opt'] * (1 + w_cur['tr_ld1']) / (
                        1 + wealth.loc[wealth['eom'] == pd.to_datetime(next_month), 'mu_ld1'].iloc[0])
                )
                w_update['w_opt_ld1'] = w_cur['w_opt_ld1']

            # Merge updated weights back into df_w
            df_w = pd.merge(w_update, df_w, on='id', how='right')
            mask_cur = (df_w['eom'] == d_ts) & (~df_w['w_opt'].isna())
            df_w.loc[mask_cur, 'w'] = df_w.loc[mask_cur, 'w_opt']

            if next_month:
                mask_next = (df_w['eom'] == pd.to_datetime(next_month)) & (~df_w['w_opt_ld1'].isna())
                df_w.loc[mask_next, 'w_start'] = df_w.loc[mask_next, 'w_opt_ld1']
                mask_new = (df_w['eom'] == pd.to_datetime(next_month)) & df_w['w_start'].isna()
                df_w.loc[mask_new, 'w_start'] = 0.0

            # Drop temporary columns
            df_w.drop(columns=['w_opt', 'w_opt_ld1'], inplace=True, errors='ignore')

            # Update w_list
            w_list[u_key] = df_w

        # Update last_u for the next iteration
        last_u = u

    # If hps is used, return a single DataFrame
    if hps is not None:
        return list(w_list.values())[0].reset_index(drop=True)

    # Return dictionary keyed by utility values otherwise
    return w_list


def mp_implement(data_tc, cov_list, lambda_list, rf,
                 wealth, gamma_rel,
                 dates_oos, dates_hp, k_vec, u_vec, g_vec, cov_type, K,
                 iter_, validation=None, seed=None):
    """
    Multiperiod-ML full implementation.

    Parameters:
        data_tc (pd.DataFrame): Data containing relevant portfolio information.
        cov_list (dict): Covariance matrices indexed by date.
        lambda_list (dict): Lambda matrices indexed by date.
        rf (pd.DataFrame): Risk-free rate data.
        wealth (pd.DataFrame): Wealth data.
        gamma_rel (float): Relative risk-aversion parameter.
        dates_oos (list): Out-of-sample dates.
        dates_hp (list): Dates for hyperparameter tuning.
        k_vec (list): List of k hyperparameter values.
        u_vec (list): List of u hyperparameter values.
        g_vec (list): List of g hyperparameter values.
        cov_type (str): Covariance adjustment type.
        K (int): Number of predicted lead variables.
        iter_ (int): Number of iterations for optimization.
        validation (pd.DataFrame, optional): Precomputed validation results.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        dict: Dictionary containing:
            - "hps": Full validation DataFrame.
            - "best_hps": Best hyperparameters per period.
            - "w": Final portfolio weights.
            - "pf": Final portfolio performance.
    """

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Generate all hyperparameter combinations
    mp_hps = pd.DataFrame(product(k_vec, g_vec), columns=['k', 'g'])

    # Extract relevant data for the hyperparameter tuning period
    pred_columns = [f"pred_ld{i}" for i in range(1, K + 1)]
    data_rel = data_tc[
        (data_tc['eom'].isin(dates_hp)) & (data_tc['valid'])
        ][['id', 'eom', 'me', 'tr_ld1'] + pred_columns].sort_values(['id', 'eom'])

    # Validation: Compute if not provided
    if validation is None:
        validation_results = []
        for i, (k, g) in enumerate(mp_hps.itertuples(index=False), 1):
            print(f"Processing hyperparameters {i}/{len(mp_hps)}: k={k}, g={g}")

            mp_w_list = mp_val_fun(
                data_rel, dates_hp, cov_list, lambda_list, wealth, rf, gamma_rel,
                cov_type, iter_, K, k=k, g=g, u_vec=u_vec, verbose=True
            )
            for u in u_vec:
                pf_ts = pf_ts_fun(mp_w_list[str(u)], data_tc, wealth, gamma_rel)
                pf_ts['k'] = k
                pf_ts['g'] = g
                pf_ts['u'] = u
                validation_results.append(pf_ts)

        validation = pd.concat(validation_results, ignore_index=True)

    # Compute cumulative metrics for hyperparameter optimization
    validation['cum_var'] = validation.groupby(['k', 'g', 'u'])['r'].transform(lambda x: x.expanding().var())
    validation['cum_obj'] = validation.groupby(['k', 'g', 'u'])['r'].transform(
        lambda x: (x - validation['tc'] - 0.5 * validation['cum_var'] * gamma_rel).expanding().mean())
    validation['rank'] = validation.groupby('eom_ret')['cum_obj'].rank(ascending=False)

    # Display validation results
    print("Validation Results:")
    print(validation)

    # Select optimal hyperparameters
    optimal_hps = validation.query("rank == 1 and eom_ret.dt.month == 12").sort_values('eom_ret')

    # Implement the final portfolio for out-of-sample dates
    print("Implementing final portfolio...")
    data_rel_oos = data_tc[
        (data_tc['eom'].isin(dates_oos)) & (data_tc['valid'])
        ][['id', 'eom', 'me', 'tr_ld1'] + pred_columns].sort_values(['id', 'eom'])

    final_weights = mp_val_fun(
        data_rel_oos, dates_oos, cov_list, lambda_list, wealth, rf, gamma_rel,
        cov_type, iter_, K, hps=optimal_hps
    )
    portfolio_performance = pf_ts_fun(final_weights, data_tc, wealth, gamma_rel)
    portfolio_performance['type'] = "Multiperiod-ML*"

    # Return results
    return {
        "hps": validation,
        "best_hps": optimal_hps,
        "w": final_weights,
        "pf": portfolio_performance
    }

