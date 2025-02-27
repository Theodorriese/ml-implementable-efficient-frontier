import pandas as pd
import numpy as np
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
    return np.diag([x[i] for i in ids])


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
def pf_ts_fun(weights, data, wealth, gam):
    """
    Computes portfolio time-series performance based on weights and returns.

    Parameters:
        weights (pd.DataFrame): Portfolio weights for each 'id' and 'eom'.
                                Expected columns: ['id', 'eom', 'w', 'w_start'].
        data (pd.DataFrame): Stock return data with 'id', 'eom', and 'ret_ld1', 'lambda'.
        wealth (pd.DataFrame): Wealth data with 'eom' and 'wealth'.
        gam (float): Risk-aversion parameter.

    Returns:
        pd.DataFrame: Time-series portfolio performance metrics including:
                      - 'inv': Total invested (absolute weight sum).
                      - 'shorting': Total short positions (absolute sum of negative weights).
                      - 'turnover': Total portfolio turnover.
                      - 'r': Portfolio return.
                      - 'tc': Transaction costs.
                      - 'eom_ret': End-of-month return date.
    """
    # Merge weights with data and wealth
    comb = data[['id', 'eom', 'ret_ld1', 'lambda']].merge(weights, on=["id", "eom"], how="left")
    comb = comb.merge(wealth[['eom', 'wealth']], on="eom", how="left")

    # Ensure 'w' is properly handled for calculations
    comb['w'] = comb['w'].fillna(0)

    # Compute portfolio metrics
    summary = comb.groupby("eom").agg(
        inv=("w", lambda x: x.abs().sum()),  # Total invested (sum of absolute weights)
        shorting=("w", lambda x: x[x < 0].abs().sum()),  # Total short positions
        turnover=("w", lambda x: np.sum(np.abs(x - comb.loc[x.index, "w_start"]))),  # Portfolio turnover
        r=("w", lambda x: np.sum(x * comb.loc[x.index, "ret_ld1"])),  # Portfolio return
        tc=("w", lambda x: (comb["wealth"].iloc[0] / 2) * np.sum(comb.loc[x.index, "lambda"] * (x - comb.loc[x.index, "w_start"]) ** 2))  # Transaction costs
    ).reset_index()

    # Adjust end-of-month return date
    summary["eom_ret"] = summary["eom"] + pd.DateOffset(months=1) - pd.DateOffset(days=1)

    return summary



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


# Addition/deletion rule
def addition_deletion_fun(chars, addition_n, deletion_n):
    """
    Applies addition and deletion rules to determine stock validity for portfolio inclusion.

    Parameters:
        chars (pd.DataFrame): DataFrame containing stock characteristics with the following columns:
                              - 'id': Stock identifier.
                              - 'eom': End-of-month date.
                              - 'valid_data': Boolean indicating if data for the stock is valid.
                              - 'valid_size': Boolean indicating if the stock passes size screening.
        addition_n (int): Number of consecutive valid periods required to add a stock to the portfolio.
        deletion_n (int): Number of consecutive invalid periods required to remove a stock from the portfolio.

    Returns:
        pd.DataFrame: The input DataFrame with additional columns:
                      - 'valid_temp': Combined validity of data and size.
                      - 'addition_count': Rolling count of consecutive valid periods for addition.
                      - 'deletion_count': Rolling count of consecutive invalid periods for deletion.
                      - 'add': Boolean indicating if the stock qualifies for addition.
                      - 'delete': Boolean indicating if the stock qualifies for deletion.
                      - 'valid': Final validity flag after applying addition and deletion rules.
    """
    # Combine 'valid_data' and 'valid_size' to determine temporary validity
    chars["valid_temp"] = chars["valid_data"] & chars["valid_size"]

    # Ensure the data is sorted by 'id' and 'eom' for rolling operations
    chars.sort_values(["id", "eom"], inplace=True)

    # Compute rolling counts for addition and deletion rules
    chars["addition_count"] = (
        chars.groupby("id")["valid_temp"]
        .rolling(window=addition_n)
        .sum()
        .reset_index(drop=True)
    )

    chars["deletion_count"] = (
        chars.groupby("id")["valid_temp"]
        .rolling(window=deletion_n)
        .sum()
        .reset_index(drop=True)
    )

    chars["add"] = chars["addition_count"] == addition_n
    chars["delete"] = chars["deletion_count"] == 0
    chars["valid"] = chars["add"] & ~chars["delete"]

    return chars
