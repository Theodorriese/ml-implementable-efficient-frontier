import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd
# from scipy.stats import rankdata
# from functools import reduce
import re


# Read configuration file
def read_config(file):
    """
    Reads a configuration file and parses its key-value pairs.

    Parameters:
        file (str): Path to the configuration file.

    Returns:
        dict: A dictionary containing the parsed configuration.
    """
    with open(file, "r") as f:
        lines = f.readlines()
    config = {}
    for line in lines:
        line = line.strip()
        if not line.startswith("#") and "=" in line:
            key, value = map(str.strip, line.split("=", 1))
            config[key] = eval(value)  # Dangerous: Consider replacing eval with safer parsing
    return config


# Create covariance matrix
def create_cov(x, ids=None):
    """
    Creates a covariance matrix.

    Parameters:
        x (dict): A dictionary containing the components for covariance calculation.
                  Expected keys: 'fct_load' (factor loadings), 'fct_cov' (factor covariance matrix),
                  'ivol_vec' (idiosyncratic volatilities).
        ids (list, optional): List of IDs to subset the components. Defaults to None.

    Returns:
        np.ndarray: The computed covariance matrix.
    """
    x["fct_load"].index = x["fct_load"].index.astype(str)
    x["ivol_vec"].index = x["ivol_vec"].index.astype(str)

    if ids is None:
        load = x["fct_load"]
        ivol = x["ivol_vec"]
    else:
        load = x["fct_load"].loc[ids]
        ivol = x["ivol_vec"].loc[ids]
    return load @ x["fct_cov"] @ load.T + np.diag(ivol)


# Create lambda
def create_lambda(x, ids):
    """
    Creates a lambda matrix for regularization.

    Parameters:
        x (dict): Dictionary containing lambda values for each ID.
        ids (list): List of IDs to construct the lambda matrix.

    Returns:
        np.ndarray: A diagonal matrix of lambda values for the given IDs.
    """
    x = {str(k): v for k, v in x.items()}  # Convert keys to strings
    ids = [str(i) for i in ids]  # Ensure IDs are also strings

    return np.diag([x[i] for i in ids])  # Now lookup works



# Compute expected risk
def expected_risk_fun(ws, dates, cov_list):
    """
    Computes the expected portfolio risk for different portfolio types over specified dates.

    Parameters:
        ws (pd.DataFrame): DataFrame containing portfolio weights with columns ['type', 'eom', 'id', 'w'].
        dates (list): List of dates for which to compute portfolio risk.
        cov_list (dict): Dictionary of covariance matrices, keyed by date.

    Returns:
        pd.DataFrame: DataFrame containing portfolio type, risk (variance), and associated date.
    """
    ws = ws.sort_values(by=["type", "eom", "id"])
    types = ws["type"].unique()
    w_list = {date: group for date, group in ws.groupby("eom")}

    results = []
    for d in dates:
        w_sub = w_list.get(d)
        if w_sub is not None:
            ids = w_sub["id"].unique()
            sigma = create_cov(cov_list[d], ids)
            for t in types:
                w = w_sub.loc[w_sub["type"] == t, "w"].values
                pf_var = w.T @ sigma @ w
                results.append({"type": t, "pf_var": pf_var, "eom": d})
    return pd.DataFrame(results)


# Long horizon returns
def long_horizon_ret(data, h, impute):
    """
    Replicates the R 'long_horizon_ret' function.

    Computes long-horizon returns over specified periods and handles missing values.

    Parameters:
        data (pd.DataFrame): Input data containing at least 'id', 'eom', 'eom_m', and 'ret_exc' columns.
        h (int): The maximum horizon for which to compute lagged returns.
        impute (str): Method to handle missing values. Options are:
                      - 'zero': Fill missing values with 0.
                      - 'mean': Fill missing values with the mean of the column (by date).
                      - 'median': Fill missing values with the median of the column (by date).

    Returns:
        pd.DataFrame: DataFrame with lagged returns (e.g., 'ret_ld1', 'ret_ld2', ...) and imputed missing values.
    """
    # Filter rows with non-NA `ret_exc` and prepare dates
    data = data.dropna(subset=["ret_exc"]).copy()
    dates = data[["eom", "eom_m"]].drop_duplicates()

    # Generate start and end for each id
    ids = data.groupby("id").agg(start=("eom", "min"), end=("eom", "max")).reset_index()

    # Create a full range of dates for each id (Cartesian join equivalent)
    full_ret = (
        ids.merge(dates, how="cross")
        .query("start <= eom <= end")  # Filter only valid date ranges
        .merge(data[["id", "eom", "eom_m", "ret_exc"]], on=["id", "eom"], how="left")
    )

    # Ensure only one eom_m column exists
    if "eom_m_x" in full_ret.columns and "eom_m_y" in full_ret.columns:
        full_ret["eom_m"] = full_ret["eom_m_x"].combine_first(full_ret["eom_m_y"])  # Prefer non-null values
        full_ret.drop(columns=["eom_m_x", "eom_m_y"], inplace=True)
    elif "eom_m_x" in full_ret.columns:
        full_ret.rename(columns={"eom_m_x": "eom_m"}, inplace=True)
    elif "eom_m_y" in full_ret.columns:
        full_ret.rename(columns={"eom_m_y": "eom_m"}, inplace=True)

    # Sort for lagging operations
    full_ret.sort_values(by=["id", "eom"], inplace=True)

    # Generate lagged return columns
    for l in range(1, h + 1):
        full_ret[f"ret_ld{l}"] = full_ret.groupby("id")["ret_exc"].shift(-l)

    # Remove rows where all lagged returns are missing
    lagged_cols = [f"ret_ld{l}" for l in range(1, h + 1)]
    full_ret["all_missing"] = full_ret[lagged_cols].isna().all(axis=1)
    print(f"All missing excludes {full_ret['all_missing'].mean() * 100:.2f}% of the observations")
    full_ret = full_ret[~full_ret["all_missing"]]
    full_ret.drop(columns=["all_missing"], inplace=True)

    # Impute missing returns
    if impute == "zero":
        full_ret[lagged_cols] = full_ret[lagged_cols].fillna(0)
    elif impute == "mean":
        full_ret[lagged_cols] = full_ret.groupby("eom")[lagged_cols].transform(lambda x: x.fillna(x.mean()))
    elif impute == "median":
        full_ret[lagged_cols] = full_ret.groupby("eom")[lagged_cols].transform(lambda x: x.fillna(x.median()))

    # Keep `eom_m` and drop unnecessary columns
    full_ret.drop(columns=["start", "end"], inplace=True)

    return full_ret



# Sigma adjustment
def sigma_gam_adj(sigma_gam, g, cov_type):
    """
    Adjusts the covariance matrix based on the specified covariance adjustment type.

    Parameters:
        sigma_gam (np.ndarray): The covariance matrix scaled by gamma (risk-aversion).
        g (float): Adjustment factor, varies based on `cov_type`.
        cov_type (str): Type of adjustment to apply:
                        - "cov_mult": Multiplies the covariance by g.
                        - "cov_add": Adds a diagonal adjustment to the covariance matrix.
                        - "cor_shrink": Shrinks the correlation matrix towards the identity matrix.

    Returns:
        np.ndarray: Adjusted covariance matrix.

    Raises:
        ValueError: If `cov_type` is unknown or not implemented.
    """
    if cov_type == "cov_mult":
        return sigma_gam * g
    elif cov_type == "cov_add":
        return sigma_gam + np.diag(np.diag(sigma_gam) * g)
    elif cov_type == "cor_shrink":
        assert abs(g) <= 1, "g must be between -1 and 1 for correlation shrinkage."
        sd_vec = np.sqrt(np.diag(sigma_gam))
        sd_vec_inv = np.diag(1 / sd_vec)
        cor_mat = sd_vec_inv @ sigma_gam @ sd_vec_inv
        cor_mat_adj = cor_mat * (1 - g) + np.eye(cor_mat.shape[0]) * g
        return np.diag(sd_vec) @ cor_mat_adj @ np.diag(sd_vec)
    else:
        raise ValueError(f"Unknown cov_type: {cov_type}")


# Initial weights
def initial_weights_new(data, w_type, udf_weights=None):
    """
    Initializes portfolio weights based on the specified type.

    Parameters:
        data (pd.DataFrame): Input data containing at least the columns 'eom' and 'me'.
        w_type (str): Weight initialization type. Options are:
                      - 'vw': Value-weighted, using market equity ('me').
                      - 'ew': Equal-weighted, 1/N for all stocks.
                      - 'rand_pos': Random positive weights that sum to 1.
                      - 'udf': User-defined weights provided via `udf_weights`.
        udf_weights (pd.DataFrame, optional): User-defined weights with columns 'id', 'eom', and 'w_start'.

    Returns:
        pd.DataFrame: Dataframe with initial weights added to the column 'w_start'.
                      A column 'w' is also added, initialized as NaN.

    Raises:
        ValueError: If `w_type` is unknown or if `udf_weights` is required but not provided.
    """
    if w_type == "vw":
        data["w_start"] = data.groupby("eom")["me"].transform(lambda x: x / x.sum())
    elif w_type == "ew":
        data["w_start"] = 1 / len(data)
    elif w_type == "rand_pos":
        data["w_start"] = np.random.random(len(data))
        data["w_start"] /= data["w_start"].sum()
    elif w_type == "udf" and udf_weights is not None:
        data = pd.merge(data, udf_weights, on=["id", "eom"], how="left")
    else:
        raise ValueError("Unknown w_type or missing udf_weights for 'udf'.")

    # Set initial weights only for the earliest 'eom'
    data.loc[data["eom"] != data["eom"].min(), "w_start"] = np.nan

    # Initialize the final weights as NaN
    data["w"] = np.nan

    return data


# Portfolio function
def agg_func(df):
    """
    Aggregates portfolio statistics for a given end-of-month (eom) group.

    This function computes key portfolio metrics, including:
    - Total invested capital (`inv`)
    - Total short positions (`shorting`)
    - Portfolio turnover (`turnover`)
    - Portfolio returns (`r`)
    - Transaction costs (`tc`)

    Parameters:
        df (pd.DataFrame): DataFrame containing portfolio data for a single `eom` period.
                           Expected columns: ['w', 'w_start', 'ret_ld1', 'lambda', 'wealth'].

    Returns:
        pd.Series: A Series containing aggregated portfolio statistics:
            - 'inv': Sum of absolute weights (total invested capital).
            - 'shorting': Sum of absolute values of negative weights (total short positions).
            - 'turnover': Sum of absolute changes in weights (`|w - w_start|`).
            - 'r': Portfolio returns computed as `sum(w * ret_ld1)`.
            - 'tc': Transaction costs calculated as `(wealth / 2) * sum(lambda * (w - w_start)^2)`.
    """
    inv = df['w'].abs().sum(skipna=True)
    shorting = df.loc[df['w'] < 0, 'w'].abs().sum(skipna=True)
    turnover = np.nansum(np.abs(df['w'] - df['w_start']))
    r = np.nansum(df['w'] * df['ret_ld1'])

    # Get unique wealth value
    wealth_val = df['wealth'].dropna().unique()
    wealth_val = wealth_val[0] if len(wealth_val) > 0 else 0

    # Transaction cost
    tc = wealth_val / 2 * np.nansum(df['lambda'] * (df['w'] - df['w_start']) ** 2)

    return pd.Series({
        'inv': inv,
        'shorting': shorting,
        'turnover': turnover,
        'r': r,
        'tc': tc
    })


def pf_ts_fun(weights, data, wealth):
    """
    Computes time-series portfolio performance based on given weights and stock return data.

    This function calculates **monthly portfolio metrics** such as:
    - Total capital invested (`inv`).
    - Short positions (`shorting`).
    - Portfolio turnover (`turnover`).
    - Monthly returns (`r`).
    - Transaction costs (`tc`).

    Parameters:
        weights (pd.DataFrame): DataFrame containing portfolio weights for each stock per month.
                                Expected columns: ['id', 'eom', 'w', 'w_start'].
        data (pd.DataFrame): Stock return data with necessary financial metrics.
                             Expected columns: ['id', 'eom', 'ret_ld1', 'pred_ld1', 'lambda'].
        wealth (pd.DataFrame): Portfolio wealth per `eom`. Expected columns: ['eom', 'wealth'].

    Returns:
        pd.DataFrame: Aggregated portfolio statistics with the following columns:
            - 'inv': Total capital invested.
            - 'shorting': Total short positions.
            - 'turnover': Portfolio turnover.
            - 'r': Portfolio returns.
            - 'tc': Transaction costs.
            - 'eom_ret': End-of-month return date.
    """

    weights["id"] = weights["id"].astype(str)
    data["id"] = data["id"].astype(str)

    # Step 1: Merge weights with relevant columns from data
    comb = pd.merge(
        weights,
        data[['id', 'eom', 'ret_ld1', 'pred_ld1', 'lambda']],
        on=['id', 'eom'],
        how='left'
    )

    # Resolve duplicate column issue dynamically
    for col in ["ret_ld1", "pred_ld1", "lambda"]:
        x_col, y_col = f"{col}_x", f"{col}_y"
        if x_col in comb.columns and y_col in comb.columns:
            comb[col] = comb[[x_col, y_col]].bfill(axis=1).iloc[:, 0]
            comb.drop(columns=[x_col, y_col], inplace=True)

    # Step 2: Merge with wealth data only if needed
    if 'wealth' not in comb.columns:
        comb = pd.merge(
            comb,
            wealth[['eom', 'wealth']],
            on='eom',
            how='left'
        )

    # Step 3: Ensure w_start is not NaN
    comb['w_start'] = comb['w_start'].fillna(0)

    # Step 4: Apply the aggregation function
    grouped = comb.groupby('eom').apply(agg_func).reset_index()

    # Step 5: Compute end-of-month return date using MonthEnd
    grouped['eom_ret'] = grouped['eom'] + MonthEnd(1)

    return grouped


# Size-based screen
def size_screen_fun(chars, screen_type):
    """
    Filters data based on size criteria (e.g., all stocks, top N, bottom N, specific size groups, or percentile ranges).

    Parameters:
        chars (pd.DataFrame): Stock characteristics with 'eom', 'me', and 'valid_data'.
                              'me' refers to market equity (size).
        screen_type (str): Type of size screen to apply:
                           - 'all': Includes all valid stocks.
                           - 'topN': Includes top N stocks by market equity.
                           - 'bottomN': Includes bottom N stocks by market equity.
                           - 'size_grp_X': Includes stocks in the specified size group X.
                           - 'perc_lowX_highY_minZ': Includes stocks within the given percentile range.

    Returns:
        pd.DataFrame: Data with an additional 'valid_size' column indicating stocks that pass the screen.
    """
    count = 0  # Ensure at least one screen is applied

    # Include all valid stocks
    if screen_type == "all":
        print("No size screen applied.")
        chars["valid_size"] = chars["valid_data"]
        count += 1

    # Select top N stocks by market equity
    elif screen_type.startswith("top"):
        top_n = int(re.search(r"\d+", screen_type).group())  # Extract N from "topN"
        chars["me_rank"] = chars.groupby("eom")["me"].rank(method="first", ascending=False)
        chars["valid_size"] = chars["me_rank"] <= top_n
        chars.drop(columns=["me_rank"], inplace=True)
        count += 1

    # Select bottom N stocks by market equity
    elif screen_type.startswith("bottom"):
        bottom_n = int(re.search(r"\d+", screen_type).group())  # Extract N from "bottomN"
        chars["me_rank"] = chars.groupby("eom")["me"].rank(method="first", ascending=True)
        chars["valid_size"] = chars["me_rank"] <= bottom_n
        chars.drop(columns=["me_rank"], inplace=True)
        count += 1

    # Filter stocks based on size group
    elif screen_type.startswith("size_grp_"):
        size_grp_value = screen_type.replace("size_grp_", "")
        chars["valid_size"] = (chars["size_grp"] == size_grp_value) & chars["valid_data"]
        count += 1

    # Percentile-based screening
    elif screen_type.startswith("perc"):
        # Extract percentile range and minimum stock count
        low_p = int(re.search(r"low(\d+)", screen_type).group(1)) # extract a number following the substring "low" in screen_type
        high_p = int(re.search(r"high(\d+)", screen_type).group(1)) # extract a number following the substring "high" in screen_type
        min_n = int(re.search(r"min(\d+)", screen_type).group(1)) # extract a number following the substring "min" in screen_type

        print(f"Percentile-based screening: Range {low_p}% - {high_p}%, min {min_n} stocks")

        # This line calculates the percentile rank of the me (market equity) column within each eom (end-of-month)
        # group and assigns it to a new column called me_perc. Converts the rank to a value between 0 and 1.
        # chars["me_perc"] = chars.groupby("eom")["me"].rank(pct=True) # already computed

        # Filter based on percentile range
        chars["valid_size"] = (chars["market_equity"] > low_p / 100) & (chars["market_equity"] <= high_p / 100)

        # Compute stock counts
        # Each row will now have a new column n_tot: the total count of valid stocks for the respective eom group.
        chars["n_tot"] = chars.groupby("eom")["valid_data"].transform("sum")

        # The total number of stocks within each eom group that meet the criteria defined by the valid_size column.
        chars["n_size"] = chars.groupby("eom")["valid_size"].transform("sum")

        # Counts valid stocks per eom where `market_equity` <= low_p / 100.
        chars["n_less"] = chars.groupby("eom")["valid_data"].transform(
            lambda x: (x & (chars.loc[x.index, "market_equity"] <= low_p / 100)).sum()
        )

        # Counts valid stocks per eom where `me_perc` > high_p / 100.
        chars["n_more"] = chars.groupby("eom")["valid_data"].transform(
            lambda x: (x & (chars.loc[x.index, "market_equity"] > high_p / 100)).sum()
        )

        # Compute missing stocks needed
        chars["n_miss"] = np.maximum(min_n - chars["n_size"], 0)
        chars["n_below"] = np.ceil(np.minimum(chars["n_miss"] / 2, chars["n_less"]))
        chars["n_above"] = np.ceil(np.minimum(chars["n_miss"] / 2, chars["n_more"]))

        # Adjust `n_below` and `n_above` when their combined value is less than the required `n_miss`.
        # If more stocks can be added to `n_above`, increment it by the remaining difference (`n_miss` - current total).
        # Similarly, if `n_below` can be incremented, adjust it to make up the shortfall.
        # This ensures that the required number of missing stocks (`n_miss`) is distributed appropriately.
        adjust_mask = (chars["n_below"] + chars["n_above"] < chars["n_miss"])
        chars.loc[adjust_mask & (chars["n_above"] > chars["n_below"]), "n_above"] += chars["n_miss"] - chars[
            "n_above"] - chars["n_below"]
        chars.loc[adjust_mask & (chars["n_below"] > chars["n_above"]), "n_below"] += chars["n_miss"] - chars[
            "n_above"] - chars["n_below"]

        # Adjust valid_size based on additional stocks needed
        chars["valid_size"] = (
                (chars["market_equity"] > (low_p / 100 - chars["n_below"] / chars["n_tot"])) &
                (chars["market_equity"] <= (high_p / 100 + chars["n_above"] / chars["n_tot"]))
        )

        # Drop intermediate columns
        chars.drop(columns=["n_tot", "n_size", "n_less", "n_more", "n_miss", "n_below", "n_above"],
                   inplace=True)
        count += 1

    # Raise error if more than one or no screen is applied
    if count != 1:
        raise ValueError("Invalid size screen applied! Please check the `screen_type` parameter.")

    return chars


def investment_universe(add_series, delete_series):
    """
    Mimics the R investment_universe function.
    Iterates over the add and delete signals:
      - When an add signal is TRUE, it turns the valid state ON.
      - When a delete signal is TRUE, it turns the valid state OFF.
    Returns a boolean Series representing the investment universe.
    """
    state = False
    valid = []
    for a, d in zip(add_series, delete_series):
        if a:
            state = True
        if d:
            state = False
        valid.append(state)
    return pd.Series(valid, index=add_series.index)


def apply_investment_universe(group):
    """
    Applies the investment_universe logic on a group.
    If the group has more than one row, compute valid as per investment_universe;
    otherwise, set valid to False.
    """
    if len(group) > 1:
        group = group.copy()
        group['valid'] = investment_universe(group['add'], group['delete'])
    else:
        group = group.copy()
        group['valid'] = False
    return group


def agg_turnover(df):
    """
    Aggregates turnover statistics for a single group of rows (one eom).
    Computes:
      - sum_valid_temp: sum of 'valid_temp' (raw number of valid_temp=TRUE)
      - sum_valid: sum of 'valid' (raw number of valid=TRUE)
      - raw: ratio of changes in valid_temp to sum_valid_temp
      - adj: ratio of changes in valid to sum_valid
    Returns a Series with these values.
    """
    sum_valid_temp = df["valid_temp"].sum()
    sum_valid = df["valid"].sum()
    raw = np.nan
    adj = np.nan

    if sum_valid_temp != 0:
        raw = df["chg_raw"].sum() / sum_valid_temp
    if sum_valid != 0:
        adj = df["chg_adj"].sum() / sum_valid

    return pd.Series({
        "raw_n": sum_valid_temp,
        "adj_n": sum_valid,
        "raw": raw,
        "adj": adj
    })


def compute_bool_changes(df, group_col, bool_col):
    """
    Faster version using `.transform()` instead of `.apply()`.

    This method avoids the overhead of `apply()` by using `transform()`,
    which is optimized for element-wise operations.

    Returns:
        A boolean Series where True indicates a change from the previous row within a group.
    """
    return df[bool_col] != df.groupby(group_col)[bool_col].shift(1)


def addition_deletion_fun(chars, addition_n, deletion_n):
    """
    Replicates the R addition_deletion_fun logic with no nested function definitions.

    Steps:
      1. Create temporary validity flag: valid_temp = (valid_data & valid_size).
      2. Sort the data by 'id' and 'eom'.
      3. For each 'id', compute rolling sums on valid_temp with windows addition_n and deletion_n.
      4. Flag add (if rolling sum equals addition_n) and delete (if rolling sum equals 0),
         then replace any NaN with False.
      5. Count the number of observations (n) per id.
      6. For groups with n > 1, compute valid using investment_universe (applied via helper);
         for groups with a single observation, set valid to False.
      7. Ensure that if valid_data is False, valid is also False.
      8. Compute turnover measures:
           - chg_raw: Change in valid_temp from previous row (by id).
           - chg_adj: Change in valid (from investment_universe) from previous row.
         Then replace NaN with False.
      9. Aggregate by eom and compute average turnover across months.
      10. Print turnover results.
      11. Drop intermediate columns.

    Parameters:
      chars (pd.DataFrame): Must contain columns 'id', 'eom', 'valid_data', and 'valid_size'.
      addition_n (int): Rolling window size for addition.
      deletion_n (int): Rolling window size for deletion.

    Returns:
      pd.DataFrame: Modified DataFrame with intermediate columns removed.
    """

    # Step 1: Temporary validity
    print("Step 13.1: Creating temporary validity flag")
    chars["valid_temp"] = chars["valid_data"] & chars["valid_size"]

    # Step 2: Sort by 'id' and 'eom'
    print("Step 13.2: Sorting by 'id' and 'eom'")
    chars.sort_values(by=["id", "eom"], inplace=True)

    # Step 3: Compute rolling sums
    print("Step 13.3: Computing rolling sums for addition and deletion")
    chars["addition_count"] = chars.groupby("id")["valid_temp"].transform(
        lambda x: x.rolling(window=addition_n, min_periods=addition_n).sum()
    )
    chars["deletion_count"] = chars.groupby("id")["valid_temp"].transform(
        lambda x: x.rolling(window=deletion_n, min_periods=deletion_n).sum()
    )

    # Step 4: Flag add/delete, replace NaN with False
    print("Step 13.4: Flagging 'add' and 'delete' signals")
    chars["add"] = (chars["addition_count"] == addition_n)
    chars["add"] = chars["add"].replace({np.nan: False})

    chars["delete"] = (chars["deletion_count"] == 0)
    chars["delete"] = chars["delete"].replace({np.nan: False})

    # Step 5: Count the number of rows per id
    print("Step 13.5: Counting rows per 'id'")
    chars["n"] = chars.groupby("id")["id"].transform("count")

    # Step 6: Apply the investment_universe logic
    print("Step 13.6: Applying investment_universe logic")
    chars = chars.groupby("id", group_keys=False).apply(apply_investment_universe)
    # For ids with a single observation, ensure valid is False
    chars.loc[chars["n"] == 1, "valid"] = False

    # Step 7: If valid_data is False, then valid must be False
    print("Step 13.7: Ensuring valid_data=False enforces valid=False")
    chars.loc[chars["valid_data"] == False, "valid"] = False

    # Step 8: Turnover calculations
    print("Step 13.8: Computing turnover changes (chg_raw and chg_adj)")
    # Compute raw changes in 'valid_temp'
    chars["chg_raw"] = compute_bool_changes(chars, "id", "valid_temp")

    # Compute adjusted changes in 'valid'
    chars["chg_adj"] = compute_bool_changes(chars, "id", "valid")

    # Step 9: Aggregate turnover by end-of-month (eom)
    print("Step 13.9: Aggregating turnover by end-of-month")
    agg = chars.groupby("eom").apply(agg_turnover).reset_index()

    # Filter out NaN or zero
    agg_filtered = agg[(~agg["raw"].isna()) & (~agg["adj"].isna()) & (agg["adj"] != 0)]
    turnover_stats = {
        "n_months": len(agg_filtered),
        "n_raw": agg_filtered["raw_n"].mean() if len(agg_filtered) > 0 else np.nan,
        "n_adj": agg_filtered["adj_n"].mean() if len(agg_filtered) > 0 else np.nan,
        "turnover_raw": agg_filtered["raw"].mean() if len(agg_filtered) > 0 else np.nan,
        "turnover_adjusted": agg_filtered["adj"].mean() if len(agg_filtered) > 0 else np.nan,
    }

    # Step 10: Print turnover results
    print("Step 13.10: Computing turnover statistics")
    print("Turnover wo addition/deletion rule: {}%".format(
        round(turnover_stats["turnover_raw"] * 100, 2) if not np.isnan(turnover_stats["turnover_raw"]) else np.nan))
    print("Turnover w  addition/deletion rule: {}%".format(
        round(turnover_stats["turnover_adjusted"] * 100, 2) if not np.isnan(
            turnover_stats["turnover_adjusted"]) else np.nan))

    # Step 11: Drop intermediate columns
    print("Step 13.11: Dropping intermediate columns and returning DataFrame")
    drop_cols = [
        "n", "addition_count", "deletion_count", "add", "delete",
        "valid_temp", "valid_data", "valid_size", "chg_raw", "chg_adj"
    ]
    chars.drop(columns=drop_cols, inplace=True, errors="ignore")

    return chars
