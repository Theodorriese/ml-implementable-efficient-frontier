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

    # R code uses: calc_dates = cluster_data_m[eom >= floor_date(fct_dates[obs], "m") - 1]$eom
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
        factor_cov_est[d] = cov_out

    # 11) Specific Risk: residuals from daily cross-sectional regressions
    #     We already have the daily fits, so we either re-run or store them. Let's re-run for clarity.
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
    # R code uses ewma_c(..., lambda=0.5^(1/hl_stock_var)), start=initial
    # We'll do a straightforward EWM with halflife
    halflife_lambda = settings["cov_set"]["hl_stock_var"]
    res_df["res_vol"] = (
        res_df.groupby("id")["residual"]
        .apply(lambda x: x.ewm(halflife=halflife_lambda).std())
        .reset_index(drop=True)
    )

    # 13) Keep last daily residual in each month
    # R code merges with trading-day range checks, etc. We'll do a simpler approach: end-of-month pivot
    res_df["date_m"] = res_df["date"].dt.to_period("M")
    spec_risk = (
        res_df.groupby(["id", "date_m"])
        .last()
        .reset_index()
        .rename(columns={"date_m": "month_period"})
    )
    spec_risk["eom_ret"] = spec_risk["month_period"].dt.to_timestamp(how="end")
    spec_risk = spec_risk[["id", "eom_ret", "res_vol"]]

    # 14) Stock-level data for each calc_date
    barra_cov = {}
    for d in tqdm(calc_dates, desc="Stock-level covariances"):
        char_data = cluster_data_m[cluster_data_m["eom"] == d].copy()
        char_data = char_data.merge(spec_risk, on=["id", "eom_ret"], how="left")

        # Fill missing residual vol by size_grp median, then overall median
        grp_med = char_data.groupby(["size_grp", "eom"])["res_vol"].transform("median")
        grp_all = char_data.groupby("eom")["res_vol"].transform("median")
        char_data["res_vol"] = char_data["res_vol"].fillna(grp_med).fillna(grp_all)

        # Get (annualized) factor covariance
        fct_cov = factor_cov_est.get(d)
        if fct_cov is None:
            continue
        fct_cov_annual = fct_cov * 21

        # Build factor loadings from monthly exposures
        X = char_data[ind_factors + list(clusters)].fillna(0).values
        ivol_vec = (char_data["res_vol"] ** 2 * 21).values

        barra_cov[d] = {
            "fct_load": X,
            "fct_cov": fct_cov_annual,
            "ivol_vec": ivol_vec
        }

    return {
        "cluster_data_d": cluster_data_d,       # daily data merged with monthly factors
        "fct_ret": fct_ret,                     # daily factor returns
        "factor_cov": factor_cov_est,           # factor covariance estimates
        "spec_risk": spec_risk,                 # specific risk by eom
        "barra_cov": barra_cov                  # stock-level factor loadings & i-vol by eom
    }
