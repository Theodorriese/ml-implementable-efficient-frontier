import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm


# Helper functions
def weighted_corrcoef(data, weights):
    """
    Compute weighted correlation matrix from 2D data (rows: observations, columns: variables)
    using aweights (1D array).
    """
    w = weights / np.sum(weights)
    mean = np.average(data, axis=0, weights=w)
    centered = data - mean
    cov = np.cov(centered.T, aweights=w, bias=False)
    stddev = np.sqrt(np.diag(cov))
    outer_std = np.outer(stddev, stddev)
    corr = cov / outer_std
    corr[outer_std == 0] = 0
    return corr


def recursive_ewma_std(series, lambda_, min_obs):
    """
    Recursively compute EWMA standard deviation with a given lambda and min_obs.
    """
    ewma_var = []
    prev_var = np.nan
    for i, x in enumerate(series):
        if i == 0:
            ewma_var.append(np.nan)
        else:
            if np.isnan(prev_var):
                prev_var = x**2
            else:
                prev_var = lambda_ * prev_var + (1 - lambda_) * x**2
            ewma_var.append(prev_var)
    ewma_std = np.sqrt(ewma_var)
    ewma_std = pd.Series(ewma_std, index=series.index)
    return ewma_std if ewma_std.count() >= min_obs else pd.Series(np.nan, index=series.index)


# Main function
def prepare_cluster_data(chars, cluster_labels, daily, settings, features):
    """
    Constructs a full cluster-based risk model including factor returns, covariances, and stock-specific risk,
    using firm characteristics and daily returns.

    Parameters:
        chars (pd.DataFrame): Monthly firm characteristics, including size group and cluster features.
        cluster_labels (pd.DataFrame): Mapping of features to clusters and their expected return directions.
        daily (pd.DataFrame): Daily returns data with excess returns and end-of-month labels.
        settings (dict): Dictionary containing model configuration (e.g., half-lives, lookback window).
        features (list): List of features used to construct cluster exposures.

    Returns:
        dict: Dictionary with the following keys:
            - 'cluster_data_d': Daily returns with merged monthly cluster exposures.
            - 'fct_ret': Estimated daily factor returns from cross-sectional regressions.
            - 'factor_cov': Estimated factor covariance matrices per month-end.
            - 'spec_risk': Stock-level specific risk (EWMA residual volatility) by month-end.
            - 'barra_cov': Stock-level factor loadings, covariances, and idiosyncratic variances per date.
    """

    # 1) Monthly cluster exposures
    cluster_data_m = chars[chars["valid"]].copy()
    cluster_data_m = cluster_data_m[["id", "eom", "size_grp", "ff12"] + features]
    clusters = cluster_labels["cluster"].unique()

    # 2) Flip characteristics if direction == -1, then average
    cluster_ranks = []
    for cl in clusters:
        sub_lbl = cluster_labels[
            (cluster_labels["cluster"] == cl) &
            (cluster_labels["characteristic"].isin(features))
        ]
        data_sub = cluster_data_m[sub_lbl["characteristic"]].copy()
        for c in sub_lbl["characteristic"]:
            if sub_lbl.loc[sub_lbl["characteristic"] == c, "direction"].iloc[0] == -1:
                data_sub[c] = 1 - data_sub[c]
        cluster_ranks.append(data_sub.mean(axis=1).rename(cl))
    cluster_ranks = pd.concat(cluster_ranks, axis=1)

    # 3) Combine with main columns, define eom_ret
    cluster_data_m = pd.concat(
        [cluster_data_m[["id", "eom", "size_grp", "ff12"]], cluster_ranks],
        axis=1
    )
    cluster_data_m["eom_ret"] = cluster_data_m["eom"] + pd.offsets.MonthEnd(1)

    # 4) Industry or market dummies
    if settings["cov_set"]["industries"]:
        industries = sorted(cluster_data_m["ff12"].unique())
        for ind in industries:
            cluster_data_m[ind] = cluster_data_m["ff12"].apply(lambda x: int(x == ind))
        ind_factors = industries
    else:
        cluster_data_m["mkt"] = 1
        ind_factors = ["mkt"]

    # 5) Standardize cluster factors by eom
    for cl in clusters:
        cluster_data_m[cl] = cluster_data_m.groupby("eom")[cl].transform(
            lambda x: (x - x.mean()) / x.std()
        )

    # 6) Merge daily returns (keeping all daily dates)
    daily_sub = daily.loc[
        daily["date"] >= cluster_data_m["eom"].min(),
        ["id", "date", "ret_exc", "eom"]
    ].rename(columns={"eom": "eom_ret"})
    cluster_data_d = cluster_data_m.merge(
        daily_sub,
        how="right",
        on=["id", "eom_ret"]
    ).dropna()

    # 7) Daily cross-sectional regressions for factor returns
    factor_returns = []
    unique_dates = sorted(cluster_data_d["date"].unique())
    X_cols = ind_factors + list(clusters)

    for dt in tqdm(unique_dates, desc="Estimating factor returns"):
        data_slice = cluster_data_d[cluster_data_d["date"] == dt]
        if len(data_slice) == 0:
            continue
        X = data_slice[X_cols].values
        y = data_slice["ret_exc"].values
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X, y)
        factor_returns.append({"date": dt, **dict(zip(X_cols, reg.coef_))})
    fct_ret = pd.DataFrame(factor_returns).sort_values("date").reset_index(drop=True)

    # 8) EWMA weights for factor-cov
    w_cor = (0.5 ** (1 / settings["cov_set"]["hl_cor"])) ** np.arange(settings["cov_set"]["obs"], 0, -1) # delete?
    w_var = (0.5 ** (1 / settings["cov_set"]["hl_var"])) ** np.arange(settings["cov_set"]["obs"], 0, -1)

    # 9) Factor covariance dates
    fct_dates = sorted(fct_ret["date"].unique())
    if settings["cov_set"]["obs"] >= len(fct_dates):
        raise ValueError("Not enough historical daily factor-return observations for the chosen window.")

    pivot_dt = pd.to_datetime(fct_dates[settings["cov_set"]["obs"]])
    start_of_month = pivot_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    start_floor = start_of_month - pd.Timedelta(days=1)
    calc_dates = sorted(cluster_data_m.loc[cluster_data_m["eom"] >= start_floor, "eom"].unique())

    # 10) Factor Covariance for each calc_date
    factor_cov_est = {}
    for d in tqdm(calc_dates, desc="Calculating factor covariance"):
        if d not in fct_dates:
            # find the nearest daily factor-return date <= d
            possible = [x for x in fct_dates if x <= d]
            if len(possible) == 0:
                continue
            d_ = possible[-1]
        else:
            d_ = d

        idx_d = fct_dates.index(d_)
        first_idx = max(0, idx_d - settings["cov_set"]["obs"] + 1)
        sub_dates = fct_dates[first_idx : idx_d + 1]
        cov_data = fct_ret[fct_ret["date"].isin(sub_dates)].drop(columns="date")
        t = len(cov_data)

        # 10.1 Compute weighted correlation matrix
        w_cor_cur = w_cor[-t:]
        w_cor_norm = w_cor_cur / np.sum(w_cor_cur)
        corr_matrix = weighted_corrcoef(cov_data.values, w_cor_norm)

        # 10.2. Compute standard deviations (from EWMA variance weights)
        w_var_cur = w_var[-t:]
        w_var_norm = w_var_cur / np.sum(w_var_cur)
        weighted_mean2 = np.average(cov_data, axis=0, weights=w_var_norm)
        centered_data2 = cov_data - weighted_mean2
        var_vector = np.average(centered_data2 ** 2, axis=0, weights=w_var_norm)
        sd_diag = np.diag(np.sqrt(var_vector))

        # 10.3. Reconstruct covariance: cov = sd * cor * sd
        cov_matrix = sd_diag @ corr_matrix @ sd_diag

        # 10.4 Apply Bessel correction
        correction_factor = np.sum(w_var_cur) / (np.sum(w_var_cur) - 1)
        cov_matrix *= correction_factor

        # 10.5. Save to factor_cov_est
        factor_names = list(fct_ret.columns[1:])
        factor_cov_est[d] = pd.DataFrame(cov_matrix, index=factor_names, columns=factor_names)

    # 11) Specific Risk: residuals from daily cross-sectional regressions
    res_list = []
    for dt in tqdm(unique_dates, desc="Estimating residuals"):
        data_slice = cluster_data_d[cluster_data_d["date"] == dt]
        if len(data_slice) == 0:
            continue
        X = data_slice[X_cols].values
        y = data_slice["ret_exc"].values
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X, y)
        resids = y - reg.predict(X)
        for (i, row_id) in enumerate(data_slice["id"].values):
            res_list.append({"id": row_id, "date": dt, "residual": resids[i]})
    res_df = pd.DataFrame(res_list).sort_values(["id", "date"]).reset_index(drop=True)

    # 12) EWMA of residual -> res_vol
    halflife_lambda = settings["cov_set"]["hl_stock_var"]
    initial_obs = settings["cov_set"]["initial_var_obs"]

    lambda_ = 0.5 ** (1 / halflife_lambda)
    res_df["res_vol"] = (
        res_df.groupby("id")["residual"]
        .apply(lambda x: recursive_ewma_std(x, lambda_=lambda_, min_obs=initial_obs))
        .reset_index(drop=True)
    )

    # 13) Keep last daily residual in each month
    res_df["date_m"] = res_df["date"].dt.to_period("M")

    spec_risk = (
        res_df.groupby(["id", "date_m"])
        .last()
        .reset_index()
        .rename(columns={"date_m": "month_period"})
    )

    spec_risk["eom_ret"] = pd.to_datetime(spec_risk["month_period"].dt.to_timestamp(freq="M"))
    spec_risk["eom_ret"] = spec_risk["eom_ret"].dt.normalize()
    spec_risk = spec_risk[["id", "eom_ret", "res_vol"]]


    # 14) Stock-level data for each calc_date
    barra_cov = {}

    for d in tqdm(calc_dates, desc="Stock-level covariances"):
        # Step 1: Extract stock-level characteristics
        char_data = cluster_data_m[cluster_data_m["eom"] == d].copy()
        char_data = char_data.merge(spec_risk, on=["id", "eom_ret"], how="left")

        # Step 2: Impute missing residual volatility (res_vol)
        grp_med = char_data.groupby(["size_grp", "eom"])["res_vol"].transform("median")
        grp_all = char_data.groupby("eom")["res_vol"].transform("median")
        char_data["res_vol"] = char_data["res_vol"].fillna(grp_med).fillna(grp_all)

        # Step 3: Retrieve factor covariance and annualize it
        fct_cov = factor_cov_est.get(d)
        if fct_cov is None:
            print(f"Warning: No factor covariance matrix for date {d}")
            continue

        fct_cov_annual = fct_cov * 21  # Annualization factor

        # Step 4: Ensure factor covariance is a DataFrame with correct index/columns
        if not isinstance(fct_cov, pd.DataFrame):
            factor_names = list(fct_ret.columns[1:])  # Extract factor names from fct_ret (excluding date column)
            fct_cov_annual = pd.DataFrame(fct_cov_annual, index=factor_names, columns=factor_names)

        # Step 5: Extract factor loadings, ensuring alignment with factor covariance
        char_data = char_data.sort_values("id")  # Sort by asset ID
        asset_ids = char_data["id"].values  # Asset IDs as row index

        factor_names = fct_cov_annual.columns.tolist()  # Extract factor names from covariance matrix
        if not all(f in char_data.columns for f in factor_names):
            print(f"Warning: Missing factor names in char_data for {d}")
            continue  # Skip if factor exposures are not properly available

        # X = char_data[factor_names].fillna(0)

        char_data = char_data.dropna(subset=factor_names)
        X = char_data[factor_names]

        X.index = char_data["id"].astype(str)

        # Step 7: Store results in dictionary (barra_cov)
        barra_cov[d] = {
            "fct_load": X,
            "fct_cov": fct_cov_annual,
            "ivol_vec": pd.Series((char_data["res_vol"] ** 2 * 21).values, index=asset_ids)  # Idiosyncratic risk
        }

    return {
        "cluster_data_d": cluster_data_d,       # daily data merged with monthly factors
        "fct_ret": fct_ret,                     # daily factor returns
        "factor_cov": factor_cov_est,           # factor covariance estimates
        "spec_risk": spec_risk,                 # specific risk by eom
        "barra_cov": barra_cov                  # stock-level factor loadings & i-vol by eom
    }
