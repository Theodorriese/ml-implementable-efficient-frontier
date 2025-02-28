import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

def prepare_cluster_data(chars, cluster_labels, daily, settings, features):
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
    w_cor = (0.5 ** (1 / settings["cov_set"]["hl_cor"])) ** np.arange(settings["cov_set"]["obs"], 0, -1)
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

        # Weighted correlation
        w_cur = w_cor[-t:]
        mean_c = np.average(cov_data, axis=0, weights=w_cur)
        centered_c = cov_data - mean_c
        cor_est = (w_cur * centered_c.T) @ centered_c / (w_cur @ w_cur)

        # Weighted variance
        wv_cur = w_var[-t:]
        mean_v = np.average(cov_data, axis=0, weights=wv_cur)
        centered_v = cov_data - mean_v
        var_est = (wv_cur * centered_v.T) @ centered_v / wv_cur.sum()
        sd_diag = np.diag(np.sqrt(np.diag(var_est)))
        cov_out = sd_diag @ cor_est @ sd_diag

        factor_names = list(fct_ret.columns[1:])
        factor_cov_est[d] = pd.DataFrame(cov_out)
        factor_cov_est[d].index = factor_names
        factor_cov_est[d].columns = factor_names

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
    res_df["res_vol"] = (
        res_df.groupby("id")["residual"]
        .apply(lambda x: x.ewm(halflife=halflife_lambda).std())
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

    # Initialize dictionary for stock-level covariance results
    barra_cov = {}

    for d in tqdm(calc_dates, desc="Stock-level covariances"):
        # Step 1: Extract stock-level characteristics
        char_data = cluster_data_m[cluster_data_m["eom"] == d].copy()
        char_data = char_data.merge(spec_risk, on=["id", "eom_ret"], how="left")  # Corrected merge

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

        X = char_data[factor_names].fillna(0).values  # Extract factor exposures, fill NaNs with 0

        # Convert fct_load to a DataFrame with asset IDs as index
        fct_load_df = pd.DataFrame(X, index=asset_ids, columns=factor_names)

        # Step 6.1: Compute stock covariance matrix (asset-level covariance)
        asset_cov = X @ fct_cov_annual @ X.T + np.diag((char_data["res_vol"] ** 2 * 21).values)

        # Convert `asset_cov` to a DataFrame before storing it
        asset_cov_df = pd.DataFrame(asset_cov)
        asset_cov_df.index = asset_ids
        asset_cov_df.columns = asset_ids

        # Step 7: Store results in dictionary (barra_cov)
        barra_cov[d] = {
            "fct_load": fct_load_df,  # Factor exposures (now a DataFrame with asset IDs)
            "fct_cov": asset_cov_df,  # Asset covariance matrix
            "ivol_vec": pd.Series((char_data["res_vol"] ** 2 * 21).values, index=asset_ids)  # Idiosyncratic risk
        }

    return {
        "cluster_data_d": cluster_data_d,       # daily data merged with monthly factors
        "fct_ret": fct_ret,                     # daily factor returns
        "factor_cov": factor_cov_est,           # factor covariance estimates
        "spec_risk": spec_risk,                 # specific risk by eom
        "barra_cov": barra_cov                  # stock-level factor loadings & i-vol by eom
    }
