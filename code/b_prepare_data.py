import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from a_general_functions import size_screen_fun, addition_deletion_fun, long_horizon_ret
import datetime as dt


# Function to load the risk-free rate data
def load_risk_free(data_path):
    """
    Load and preprocess risk-free rate data.

    This function reads the Fama-French 3-factor model risk-free rate dataset,
    filters valid entries, converts the risk-free rate to decimal format, and
    creates date columns for both the standard end-of-month (EOM) and the
    last business day of the same month.

    Parameters:
        data_path (str): The directory path where the risk-free data file is stored.

    Returns:
        pd.DataFrame: A DataFrame with the following columns:
            - 'eom': Standard end-of-month date.
            - 'rf': Risk-free rate in decimal format.
            - 'eom_bd': Last business day of the same month.
    """
    # Load the data, skipping the first three rows (metadata) and selecting relevant columns
    risk_free = pd.read_csv(
        os.path.join(data_path, "ff3_m.csv"),
        skiprows=3,
        usecols=["yyyymm", "RF"],
        sep=",",
    )

    risk_free = risk_free[risk_free["yyyymm"].astype(str).str.match(r"^\d{6}$")]
    risk_free["RF"] = pd.to_numeric(risk_free["RF"], errors="coerce")
    risk_free["rf"] = risk_free["RF"] / 100

    # Generate end-of-month date ('eom') using 'yyyymm'
    risk_free["eom_m"] = (pd.to_datetime(risk_free["yyyymm"].astype(str) + "01", format="%Y%m%d")
                          + pd.offsets.MonthEnd(0))

    risk_free["date"] = risk_free["eom_m"]

    return risk_free[["eom_m", "rf", "date"]]


# Function to load and preprocess market data
def load_market_data(settings):
    """
    Load and preprocess market data.

    Parameters:
        settings (dict): Settings containing 'data_path' and 'region'.

    Returns:
        pd.DataFrame: Filtered market data for the specified region (USA or EU/Germany).
    """
    data_path = settings["data_path"]
    region = settings.get("region", "USA")  # Default to US

    # Load the market data
    market = pd.read_csv(os.path.join(data_path, "market_returns.csv"))
    market["eom"] = pd.to_datetime(market["eom"])

    # Filter based on the region setting
    if region == "EU":
        market_data = market[market["excntry"] == "DEU"]
    elif region == "USA":
        market_data = market[market["excntry"] == "USA"]
    else:
        raise ValueError(f"Unsupported region: {region}. Please use 'USA' or 'EU'.")

    # Return the filtered DataFrame with only necessary columns
    return market_data[["eom", "mkt_vw_exc"]]


# Wealth function
def wealth_func(wealth_end, end, market, risk_free):
    """
    Computes cumulative wealth over time based on excess market returns and risk-free rates,
    ending at a specified wealth value.

    Parameters:
        wealth_end (float): Target wealth value at the final date.
        end (datetime.date or str): Final evaluation date for the wealth path.
        market (pd.DataFrame): DataFrame containing market excess returns (column 'mkt_vw_exc').
        risk_free (pd.DataFrame): DataFrame with monthly risk-free rates (column 'rf') and 'eom_m' date.

    Returns:
        pd.DataFrame: Time series of portfolio wealth and corresponding returns, indexed by month-end.
    """

    wealth = pd.merge(
        risk_free.rename(columns={"eom_m": "eom_ret"}),
        market,
        left_on="eom_ret",
        right_on="eom",
        how="inner"
    )
    wealth["tret"] = wealth["mkt_vw_exc"] + wealth["rf"]

    # Ensure `end` is a datetime.date object
    if isinstance(end, dt.datetime):
        end = end.date()
    elif not isinstance(end, dt.date):
        end = pd.to_datetime(end).date()

    # Convert `eom_ret` to datetime.date for comparison
    wealth["eom_ret"] = pd.to_datetime(wealth["eom_ret"]).dt.date

    # Filter rows where `eom_ret` is less than or equal to `end`
    wealth = wealth[wealth["eom_ret"] <= end].sort_values(by="eom_ret", ascending=False)

    # Calculate cumulative wealth
    wealth["wealth"] = (1 - wealth["tret"]).cumprod() * wealth_end
    wealth = wealth.assign(mu_ld1=wealth["tret"])

    # Adjust `eom` column to have consistent month-end formatting
    wealth["eom"] = pd.to_datetime(wealth["eom_ret"]) + pd.offsets.MonthEnd(0)

    # Append the final row for the `end` date
    wealth = pd.concat([
        wealth,
        pd.DataFrame({
            "eom": [pd.to_datetime(end) + pd.offsets.MonthEnd(0)],
            "wealth": [wealth_end],
            "mu_ld1": [np.nan]
        })
    ])

    wealth["eom_ret"] = pd.to_datetime(wealth["eom_ret"])

    return wealth.sort_values(by="eom")


# Function to load cluster labels
def load_cluster_labels(data_path):
    """
    Loads and processes cluster label metadata used for feature grouping and directionality.

    Parameters:
        data_path (str): Path to the directory containing 'Cluster_labels.csv' and 'Factor Details.xlsx'.

    Returns:
        pd.DataFrame: Cluster label DataFrame with associated feature direction and cleaned names.
    """

    # Load Cluster Labels
    cluster_labels = pd.read_csv(os.path.join(data_path, "Cluster_labels.csv"))
    cluster_labels["cluster"] = cluster_labels["cluster"].str.lower().str.replace(r"\s|-", "_", regex=True)

    # Load Factor Details
    factor_signs = pd.read_excel(os.path.join(data_path, "Factor Details.xlsx"))[["abr_jkp", "direction"]].dropna()
    factor_signs.columns = ["characteristic", "direction"]
    factor_signs["direction"] = factor_signs["direction"].astype(float)

    # Merge direction into cluster labels
    cluster_labels = pd.merge(cluster_labels, factor_signs, on="characteristic", how="left")

    # Append additional row
    additional_row = pd.DataFrame({
        "characteristic": ["rvol_252d"],
        "cluster": ["low_risk"],
        "direction": [-1]
    })

    cluster_labels = pd.concat([cluster_labels, additional_row], ignore_index=True)
    return cluster_labels


def load_monthly_data_USA(data_path, settings, risk_free):
    """
    Load, clean, and prepare monthly data for return calculations.

    Parameters:
        data_path (str): Path to the data directory.
        settings (dict): Configuration settings.
        risk_free (pd.DataFrame): Dataframe containing risk-free rates.

    Returns:
        pd.DataFrame: Cleaned and processed dataframe with long-horizon returns and total returns.
    """
    # Load and preprocess monthly data
    monthly_data = pd.read_csv(
        os.path.join(data_path, "world_ret_monthly.csv"),
        usecols=["PERMNO", "date", "EXCHCD", "TICKER", "COMNAM", "DLRET", "RET"]
    )
    monthly_data.rename(columns={"date": "eom", "PERMNO": "id"}, inplace=True)

    # Generate end-of-month date ('eom') using the consistent method
    monthly_data["eom"] = pd.to_datetime(monthly_data["eom"], format="%d/%m/%Y", dayfirst=True)
    monthly_data["eom_m"] = monthly_data["eom"] + pd.offsets.MonthEnd(0)  # Align with risk-free

    # Clean and filter EXCHCD and RET
    monthly_data["EXCHCD"] = monthly_data["EXCHCD"].fillna(0).astype(int)
    monthly_data = monthly_data[monthly_data["EXCHCD"].isin([1, 2, 3])]
    monthly_data["RET"] = pd.to_numeric(monthly_data["RET"], errors="coerce")
    monthly_data.dropna(subset=["RET"], inplace=True)

    # Merge risk-free rate BEFORE long_horizon_ret so it can be used for excess return calculation
    monthly_data = monthly_data.merge(risk_free, on="eom_m", how="left")
    monthly_data["ret_exc"] = monthly_data["RET"] - monthly_data["rf"]  # Compute excess return

    # Compute long-horizon returns using the new excess return column
    data_ret = long_horizon_ret(monthly_data, h=settings["pf"]["hps"]["m1"]["K"], impute="zero")

    # Add `tr_ld1` (total return) by combining `ret_ld1` with `rf`
    data_ret = data_ret.merge(risk_free, on="eom_m", how="left")
    data_ret["tr_ld1"] = data_ret["ret_ld1"] + data_ret["rf"]
    data_ret.drop(columns=["rf"], inplace=True)

    # Add total return at T-1 by shifting `tr_ld1` backwards
    data_ret["tr_ld0"] = data_ret.groupby("id")["tr_ld1"].shift(1)

    # Final formatting
    column_order = ["eom", "eom_m"] + [col for col in data_ret.columns if col not in ["eom", "eom_m"]]
    data_ret = data_ret[column_order]
    data_ret[["eom_m", "eom"]] = data_ret[["eom_m", "eom"]].apply(pd.to_datetime)

    return data_ret


def flag_extreme_ret(group):
    """
    Flags extreme return values within a group based on 0.1% and 99.9% quantiles.

    Parameters:
        group (pd.DataFrame): DataFrame containing a 'RET' column of returns.

    Returns:
        pd.DataFrame: The same group with an 'extreme_ret' boolean column indicating outliers.
    """

    lower = group["RET"].quantile(0.001)
    upper = group["RET"].quantile(0.999)
    group["extreme_ret"] = (group["RET"] < lower) | (group["RET"] > upper)
    return group


def load_monthly_data_EU(data_path, settings, risk_free):
    """
    Load, clean, and prepare monthly data for return calculations (EU version).

    Parameters:
        data_path (str): Path to the data directory.
        settings (dict): Configuration settings.
        risk_free (pd.DataFrame): Dataframe containing risk-free rates.

    Returns:
        pd.DataFrame: Cleaned and processed dataframe with long-horizon returns and total returns.
    """
    # Load and preprocess monthly data
    monthly_data = pd.read_csv(
        os.path.join(data_path, "world_ret_monthly.csv"),
        usecols=["gvkey", "datadate", "ISIN", "prccm", "ajexm", "ajpm", "iid"]
    )

    # Rename columns
    monthly_data.rename(columns={"datadate": "eom", "gvkey": "id"}, inplace=True)
    monthly_data.dropna(subset=["ISIN"], inplace=True)

    # Convert dates
    monthly_data["eom"] = pd.to_datetime(monthly_data["eom"], format="%d/%m/%Y", dayfirst=True)
    monthly_data["eom_m"] = monthly_data["eom"] + pd.offsets.MonthEnd(0)

    # Remove rows with missing `prccm`
    monthly_data = monthly_data[~monthly_data["id"].isin(
        monthly_data[monthly_data["prccm"].isna()]["id"].unique()
    )]

    # Deduplicate: keep lowest iid per ISIN-date first, then per id-date
    monthly_data.sort_values(by=["ISIN", "eom", "iid"], inplace=True)
    monthly_data = monthly_data.drop_duplicates(subset=["ISIN", "eom"], keep="first")
    monthly_data.sort_values(by=["id", "eom", "iid"], inplace=True)
    monthly_data = monthly_data.drop_duplicates(subset=["id", "eom"], keep="first")

    # Final sort before calculation
    monthly_data.sort_values(by=["id", "eom"], inplace=True)

    # Calculate RET (returns)
    monthly_data["RET"] = (
            (((monthly_data["prccm"] / monthly_data["ajexm"]) * monthly_data["ajpm"]) /
             ((monthly_data.groupby("id")["prccm"].shift(1) / monthly_data.groupby("id")["ajexm"].shift(1)) *
              monthly_data.groupby("id")["ajpm"].shift(1))) - 1
    )

    monthly_data.dropna(subset=["RET"], inplace=True)

    # Remove extreme return stocks
    monthly_data = monthly_data.groupby("eom", group_keys=False).apply(flag_extreme_ret)
    bad_ids = monthly_data.loc[monthly_data["extreme_ret"], "id"].unique()
    monthly_data = monthly_data[~monthly_data["id"].isin(bad_ids)]
    monthly_data.drop(columns=["extreme_ret"], inplace=True)

    # Merge risk-free rate
    monthly_data = monthly_data.merge(risk_free, on="eom_m", how="left")
    monthly_data["ret_exc"] = monthly_data["RET"] - monthly_data["rf"]

    # Long-horizon returns
    data_ret = long_horizon_ret(monthly_data, h=settings["pf"]["hps"]["m1"]["K"], impute="zero")

    # Total return calc
    data_ret = data_ret.merge(risk_free, on="eom_m", how="left")
    data_ret["tr_ld1"] = data_ret["ret_ld1"] + data_ret["rf"]
    data_ret["tr_ld0"] = data_ret.groupby("id")["tr_ld1"].shift(1)
    data_ret.drop(columns=["rf"], inplace=True)

    # Final formatting
    column_order = ["eom", "eom_m"] + [col for col in data_ret.columns if col not in ["eom", "eom_m"]]
    data_ret = data_ret[column_order]
    data_ret[["eom_m", "eom"]] = data_ret[["eom_m", "eom"]].apply(pd.to_datetime)

    return data_ret


# Prepare data - stock characteristics
def preprocess_chars(data_path, features, settings, data_ret_ld1, wealth):
    """
    Preprocess characteristics data for either USA or EU region.

    Parameters:
        data_path (str): Path to the data directory.
        features (list): List of feature columns to load.
        settings (dict): Configuration settings including 'region' and 'pi'.
        data_ret_ld1 (pd.DataFrame): Dataframe containing return data for merging.
        wealth (pd.DataFrame): Dataframe containing wealth data.

    Returns:
        pd.DataFrame: Processed characteristics data compatible with further processing.

    Notes:
        - For USA, reads 'usa.csv', expects 'id' column, and filters only CRSP observations (id <= 99999).
        - For EU, reads 'eu.csv', expects 'gvkey' column (renamed to 'id') and does NOT filter CRSP observations.
    """
    # Choose the correct file and columns based on region
    if settings["region"] == "USA":
        cols_to_use = ["id", "eom", "sic", "ff49", "size_grp", "me", "crsp_exchcd", "rvol_252d",
                       "dolvol_126d"] + features
        chars = pd.read_csv(os.path.join(data_path, "usa.csv"), usecols=cols_to_use)
        chars = chars[chars["id"] <= 99999]  # Filter only CRSP observations
    else:  # EU Case
        cols_to_use = ["gvkey", "eom", "sic", "ff49", "size_grp", "me", "rvol_252d", "dolvol_126d"] + features
        chars = pd.read_csv(os.path.join(data_path, "eu.csv"), usecols=cols_to_use)
        chars.rename(columns={"gvkey": "id"}, inplace=True)

    # Add useful columns
    chars["dolvol"] = chars["dolvol_126d"]
    chars["lambda"] = 2 / chars["dolvol"] * settings["pi"]
    chars["rvol_m"] = chars["rvol_252d"] * np.sqrt(21)

    # Convert columns to datetime for proper merging
    chars["eom"] = pd.to_datetime(chars["eom"])

    chars.rename(columns={"eom": "eom_m"}, inplace=True)
    chars = chars.merge(data_ret_ld1, on=["id", "eom_m"], how="left")
    chars.drop(columns=["eom"], inplace=True)
    chars.rename(columns={"eom_m": "eom_ret"}, inplace=True)
    chars = chars.merge(wealth[["eom_ret", "mu_ld1"]], on="eom_ret", how="left")
    chars.rename(columns={"eom_ret": "eom"}, inplace=True)

    # Shift `eom` forward by 1 month before merging with `chars`
    wealth_temp = wealth.copy()
    wealth_temp["eom"] = (wealth_temp["eom"] + pd.DateOffset(months=1)).apply(lambda x: x + pd.offsets.MonthEnd(0))

    # Merge mu_ld1 as mu_ld0 into chars
    chars = chars.merge(wealth_temp[["eom", "mu_ld1"]].rename(columns={"mu_ld1": "mu_ld0"}), on="eom", how="left")

    return chars


# Date screen
def filter_chars(chars, settings):
    """
    Filters stock-level characteristic data based on date range and essential data availability,
    while logging the exclusion percentages for transparency.

    Parameters:
        chars (pd.DataFrame): DataFrame containing firm characteristics and metadata.
        settings (dict): Dictionary with 'screens' sub-dict containing 'start' and 'end' date keys.

    Returns:
        tuple:
            - pd.DataFrame: Filtered DataFrame.
            - int: Number of observations before filtering.
            - float: Total market equity before filtering.
    """

    chars = chars[(chars["eom"] >= settings["screens"]["start"]) &
                  (chars["eom"] <= settings["screens"]["end"])]

    # Monitor screen impact
    n_start = len(chars)
    me_start = chars["me"].sum(skipna=True)

    # Require non-missing 'me' (market equity)
    me_exclusion_pct = round(chars["me"].isna().mean() * 100, 2)
    print(f"Non-missing 'me' excludes {me_exclusion_pct}% of the observations")
    chars = chars.dropna(subset=["me"])

    # Require non-missing return t and t+1
    return_exclusion_pct = round(chars[["tr_ld1", "tr_ld0"]].isna().any(axis=1).mean() * 100, 2)
    print(f"Valid return req excludes {return_exclusion_pct}% of the observations")
    chars = chars.dropna(subset=["tr_ld1", "tr_ld0"])

    # Require non-missing and non-zero 'dolvol'
    dolvol_exclusion_pct = round(((chars["dolvol"].isna()) | (chars["dolvol"] == 0)).mean() * 100, 2)
    print(f"Non-missing/non-zero 'dolvol' excludes {dolvol_exclusion_pct}% of the observations")
    chars = chars[chars["dolvol"].notna() & (chars["dolvol"] > 0)]

    # Require stock to have SIC code (for covariance estimation with industry)
    sic_exclusion_pct = round(chars["sic"].isna().mean() * 100, 2)
    print(f"Valid SIC code excludes {sic_exclusion_pct}% of the observations")
    chars = chars.dropna(subset=["sic"])

    return chars, n_start, me_start


# Feature screen
def feature_screen(chars, features, settings, n_start, me_start, run_sub):
    """
    Screens dataset based on feature availability threshold and optionally samples a subset of firms.

    Parameters:
        chars (pd.DataFrame): DataFrame of firm characteristics.
        features (list): List of feature column names to check for availability.
        settings (dict): Dictionary containing screen settings (including 'feat_pct' and 'seed').
        n_start (int): Initial number of observations before feature screening.
        me_start (float): Initial total market equity before feature screening.
        run_sub (bool): Whether to randomly sample a subset of firms.

    Returns:
        pd.DataFrame: Filtered (and optionally sampled) characteristic data.
    """

    # Count available features per row
    feat_available = chars[features].notna().sum(axis=1)

    # Calculate minimum required features per row
    min_feat = np.floor(len(features) * settings["screens"]["feat_pct"])

    # Filter dataset
    chars = chars[feat_available >= min_feat]

    # Summary statistics
    final_obs_pct = round((len(chars) / n_start) * 100, 2)
    final_market_cap_pct = round((chars["me"].sum() / me_start) * 100, 2)
    print(f"In total, the final dataset has {final_obs_pct}% of the observations and"
          f" {final_market_cap_pct}% of the market cap in the post {settings['screens']['start']} data")

    # Screen out if running subset
    if run_sub:
        np.random.seed(settings["seed"])
        sampled_ids = np.random.choice(chars["id"].unique(), 2500, replace=False)
        chars = chars[chars["id"].isin(sampled_ids)]

    return chars


# Feature standardization
def standardize_features(chars, features, settings):
    """
    Standardizes feature columns using empirical cumulative distribution function (ECDF) transformation.

    Parameters:
        chars (pd.DataFrame): DataFrame containing characteristics data.
        features (list): List of feature columns to be transformed.
        settings (dict): Dictionary containing standardization settings.

    Returns:
        pd.DataFrame: Updated chars DataFrame with standardized features.
    """
    if settings.get("feat_prank", False):  # Ensure standardization is enabled
        # Convert feature columns to float for consistency
        chars[features] = chars[features].astype(float)

        # Check for missing or all-NaN columns and remove them from processing
        valid_features = [f for f in features if chars[f].notna().any()]
        if len(valid_features) < len(features):
            print(f"Warning: Dropping {len(features) - len(valid_features)} all-NaN features from ECDF transformation.")

        # Create a zero-mask for all features at once to avoid fragmentation
        zero_mask = chars[valid_features] == 0

        # Apply ECDF transformation **grouped by eom** for all features at once
        chars[valid_features] = chars.groupby("eom", group_keys=False)[valid_features].transform(
            lambda x: x.rank(method="average", pct=True) if len(x) > 1 else x
        )

        # Restore exact zero values
        chars[valid_features] = chars[valid_features].where(~zero_mask, 0)

        # Ensure memory efficiency by defragmenting DataFrame
        chars = chars.copy()

    return chars


# Feature imputation
def impute_features(chars, features, settings):
    """
    Imputes missing feature values based on specified settings using either percentile rank fill or median.

    Parameters:
        chars (pd.DataFrame): DataFrame of firm characteristics.
        features (list): List of feature column names to impute.
        settings (dict): Dictionary with 'feat_impute' and 'feat_prank' flags.

    Returns:
        pd.DataFrame: DataFrame with imputed feature values.
    """

    if settings["feat_impute"]:
        if settings["feat_prank"]:
            # Replace NaNs with 0.5 for all features
            chars[features] = chars[features].fillna(0.5)
        else:
            # Replace NaNs with the median of each feature within each 'eom' group
            chars[features] = chars.groupby("eom")[features].transform(lambda x: x.fillna(x.median()))

    return chars


# Industry classification
def classify_industry(chars):
    """
    Classifies firms into Fama-French 12 industry groups based on SIC codes.

    Parameters:
        chars (pd.DataFrame): DataFrame containing a 'sic' column with SIC industry codes.

    Returns:
        None: Modifies the input DataFrame in place by adding a new 'ff12' industry classification column.
    """

    chars["ff12"] = np.select(
        [
            chars["sic"].between(100, 999) | chars["sic"].between(2000, 2399) |
            chars["sic"].between(2700, 2749) | chars["sic"].between(2770, 2799) |
            chars["sic"].between(3100, 3199) | chars["sic"].between(3940, 3989),

            chars["sic"].between(2500, 2519) | chars["sic"].between(3630, 3659) |
            chars["sic"].between(3710, 3711) | chars["sic"].between(3714, 3714) |
            chars["sic"].between(3716, 3716) | chars["sic"].between(3750, 3751) |
            chars["sic"].between(3792, 3792) | chars["sic"].between(3900, 3939) |
            chars["sic"].between(3990, 3999),

            chars["sic"].between(2520, 2589) | chars["sic"].between(2600, 2699) |
            chars["sic"].between(2750, 2769) | chars["sic"].between(3000, 3099) |
            chars["sic"].between(3200, 3569) | chars["sic"].between(3580, 3629) |
            chars["sic"].between(3700, 3709) | chars["sic"].between(3712, 3713) |
            chars["sic"].between(3715, 3715) | chars["sic"].between(3717, 3749) |
            chars["sic"].between(3752, 3791) | chars["sic"].between(3793, 3799) |
            chars["sic"].between(3830, 3839) | chars["sic"].between(3860, 3899),

            chars["sic"].between(1200, 1399) | chars["sic"].between(2900, 2999),
            chars["sic"].between(2800, 2829) | chars["sic"].between(2840, 2899),
            chars["sic"].between(3570, 3579) | chars["sic"].between(3660, 3692) |
            chars["sic"].between(3694, 3699) | chars["sic"].between(3810, 3829) |
            chars["sic"].between(7370, 7379),

            chars["sic"].between(4800, 4899),
            chars["sic"].between(4900, 4949),
            chars["sic"].between(5000, 5999) | chars["sic"].between(7200, 7299) |
            chars["sic"].between(7600, 7699),
            chars["sic"].between(2830, 2839) | chars["sic"].between(3693, 3693) |
            chars["sic"].between(3840, 3859) | chars["sic"].between(8000, 8099),
            chars["sic"].between(6000, 6999)
        ],
        [
            "NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq",
            "Telcm", "Utils", "Shops", "Hlth", "Money"
        ],
        default="Other"
    )

    return chars


# Check for valid observations and eligiblity for portfolio-ml
def validate_observations(chars, pf_set):
    """
    Check which observations are valid for portfolio-ML.

    Parameters:
        chars (pd.DataFrame): The dataset containing stock data.
        pf_set (dict): Dictionary containing portfolio settings, including `lb_hor`.

    Returns:
        pd.DataFrame: Updated dataset with a `valid_data` column.
    """
    chars = chars.sort_values(by=["id", "eom"])  # Ensure ordering
    chars["valid_data"] = True  # Initialize valid_data column
    # Compute lookback period
    lb = pf_set["lb_hor"] + 1  # Plus 1 to align with last signal of previous portfolio
    chars["eom_lag"] = chars.groupby("id")["eom"].shift(lb)  # Compute lagged `eom`

    # Compute the difference in months between `eom_lag` and `eom`
    chars["month_diff"] = ((chars["eom"] - chars["eom_lag"]) / pd.Timedelta(days=30)).round().astype("Int64")

    # Identify invalid observations
    invalid_mask = (chars["month_diff"] != lb) | chars["month_diff"].isna()
    invalid_percentage = invalid_mask.mean() * 100
    print(f"   Valid lookback observation screen excludes {invalid_percentage:.2f}% of the observations")

    # Apply validity filter and remove temporary columns
    chars["valid_data"] = chars["valid_data"] & ~invalid_mask
    chars.drop(columns=["eom_lag", "month_diff"], inplace=True)

    return chars


def apply_size_screen(chars, settings):
    """
    Apply size-based screening.

    Parameters:
        chars (pd.DataFrame): The dataset containing stock data.
        settings (dict): Dictionary containing screen settings.

    Returns:
        pd.DataFrame: Updated dataset after size screening.
    """
    chars = size_screen_fun(chars, screen_type=settings["screens"]["size_screen"])
    return chars


def apply_addition_deletion_rule(chars, settings):
    """
    Apply addition/deletion rules.

    Parameters:
        chars (pd.DataFrame): The dataset containing stock data.
        settings (dict): Dictionary containing rule settings.

    Returns:
        pd.DataFrame: Updated dataset after applying rules.
    """
    chars = addition_deletion_fun(chars, addition_n=settings["addition_n"], deletion_n=settings["deletion_n"])
    return chars


def show_investable_universe(chars, colours=None, save_path=None):
    """
    Plot the number of valid stocks over time and print summary statistics.

    Parameters:
        chars (pd.DataFrame): The dataset containing stock data (must include 'eom' and 'valid').
        colours (list of str, optional): List of colors [main_line, avg_line, _]. Defaults to standard.
    """
    if colours is None:
        colours = ["steelblue", "darkorange", "gray"]

    # Filter and group
    investable = chars[chars["valid"]].groupby("eom").size()

    # Summary stats
    print("\nInvestable stock counts over time:\n", investable)
    print("\nSummary Statistics:")
    print(f"  Minimum: {investable.min():,}")
    print(f"  Maximum: {investable.max():,}")
    print(f"  Median:  {int(investable.median()):,}")
    print(f"  Average: {int(investable.mean()):,}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.grid(False)

    # Time series line
    ax.plot(
        investable.index, investable.values,
        color=colours[0], linewidth=2,
        label="Valid Stocks in Sample"
    )

    # Average line
    ax.axhline(
        y=investable.mean(),
        color=colours[1] if len(colours) > 1 else "black",
        linestyle='--', linewidth=2,
        label=f"Sample Average = {int(investable.mean()):,}"
    )

    # Labels and formatting
    ax.set_xlabel("Date (End of Month)")
    ax.set_ylabel("Valid Stocks")
    ax.set_title("Investable Universe Over Time")
    ax.legend(loc="upper right")
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def valid_summary(chars):
    """
    Compute and print the validity summary.

    Parameters:
        chars (pd.DataFrame): The dataset containing stock data.
    """
    valid_percentage = chars["valid"].mean() * 100
    market_cap_percentage = chars.loc[chars["valid"], "me"].sum() / chars["me"].sum() * 100

    print(f"   The valid_data subset has {valid_percentage:.2f}% of the observations "
          f"and {market_cap_percentage:.2f}% of the market cap")


def load_daily_returns_USA(data_path, chars, risk_free):
    """
    Load and preprocess daily returns data for the US region from a CSV file.

    Parameters:
        data_path (str): Path to the dataset directory.
        chars (pd.DataFrame): Processed characteristics data with valid stocks.
        risk_free (pd.DataFrame): DataFrame containing risk-free rates with columns ['eom_m', 'rf'].

    Returns:
        pd.DataFrame: Preprocessed daily returns data with calculated excess returns.
    """

    # Load daily returns data
    daily = pd.read_csv(os.path.join(data_path, "usa_dsf.csv"))

    # Rename columns for consistency
    daily.rename(columns={"PERMNO": "id", "RET": "RET"}, inplace=True)

    # Filter for valid returns and IDs
    daily = daily[daily["RET"].notna() & daily["id"].le(99999)]

    # Filter to only valid IDs from characteristics
    valid_ids = chars.loc[chars["valid"], "id"].unique()
    daily = daily[daily["id"].isin(valid_ids)]

    # Convert date
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce", dayfirst=True)

    # Create a copy of risk-free to safely manipulate
    rf_temp = risk_free.copy()
    rf_temp["date"] = pd.to_datetime(rf_temp["eom_m"])
    rf_temp["days_in_month"] = rf_temp["date"].dt.daysinmonth
    rf_temp["rf_daily"] = rf_temp["rf"] / rf_temp["days_in_month"]
    rf_temp["month"] = rf_temp["date"].dt.to_period("M")

    # Merge monthly RF rate to daily data by month
    daily["month"] = daily["date"].dt.to_period("M")
    daily = daily.merge(rf_temp[["month", "rf_daily"]], on="month", how="left")
    daily.drop(columns=["month"], inplace=True)

    # Ensure `RET` is numeric
    daily["RET"] = pd.to_numeric(daily["RET"], errors="coerce")

    # Calculate excess return
    daily["ret_exc"] = daily["RET"] - daily["rf_daily"]

    # Add end-of-month
    daily["eom"] = daily["date"] + pd.offsets.MonthEnd(0)

    # Drop unused columns
    daily.drop(columns=["rf_daily"], inplace=True)

    return daily


# Function to load and preprocess daily returns data for EU from a CSV file
def load_daily_returns_pkl_EU(data_path, chars, risk_free):
    """
    Load and preprocess daily returns data for EU region from a CSV file.

    Parameters:
        data_path (str): Path to the dataset directory.
        chars (pd.DataFrame): Processed characteristics data with valid stocks.
        risk_free (pd.DataFrame): DataFrame containing risk-free rates with columns ['eom_m', 'rf'].

    Returns:
        pd.DataFrame: Preprocessed daily returns data with calculated excess returns.
    """

    # Load daily returns data
    daily = pd.read_csv(os.path.join(data_path, "eu_dsf.csv"))

    # Rename columns for consistency
    daily.rename(columns={
        "gvkey": "id", "datadate": "date", "prccd": "PRCCD",
        "ajexdi": "AJEXDI", "trfd": "TRFD"
    }, inplace=True)

    # Filter for valid price & adjustment data
    daily = daily[daily["PRCCD"].notna() & daily["AJEXDI"].notna() & daily["TRFD"].notna()]

    # Filter to only valid IDs from characteristics
    valid_ids = chars.loc[chars["valid"], "id"].unique()
    daily = daily[daily["id"].isin(valid_ids)]

    # Convert date
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce", dayfirst=True)

    # Keep only one ISIN per (id, date) with lowest iid
    daily = daily[~daily["ISIN"].isna()]
    daily.sort_values(by=["id", "date", "iid"], inplace=True)
    daily = daily.drop_duplicates(subset=["id", "date"], keep="first")

    # Sort for return calc
    daily.sort_values(by=["id", "date"], inplace=True)

    # Calculate daily total return
    daily["RET"] = (
            (((daily["PRCCD"] / daily["AJEXDI"]) * daily["TRFD"]) /
             ((daily.groupby("id")["PRCCD"].shift(1) / daily.groupby("id")["AJEXDI"].shift(1)) *
              daily.groupby("id")["TRFD"].shift(1))) - 1
    )

    daily.dropna(subset=["RET"], inplace=True)

    # Flag and remove extreme stocks
    daily = daily.groupby("date", group_keys=False).apply(flag_extreme_ret)
    bad_ids = daily.loc[daily["extreme_ret"], "id"].unique()
    daily = daily[~daily["id"].isin(bad_ids)]
    daily.drop(columns=["extreme_ret"], inplace=True)

    # Create a copy of risk-free to safely manipulate
    rf_temp = risk_free.copy()
    rf_temp["date"] = pd.to_datetime(rf_temp["eom_m"])
    rf_temp["days_in_month"] = rf_temp["date"].dt.daysinmonth
    rf_temp["rf_daily"] = rf_temp["rf"] / rf_temp["days_in_month"]
    rf_temp["month"] = rf_temp["date"].dt.to_period("M")

    # Merge monthly RF rate to daily data by month
    daily["month"] = daily["date"].dt.to_period("M")
    daily = daily.merge(rf_temp[["month", "rf_daily"]], on="month", how="left")
    daily.drop(columns=["month"], inplace=True)

    # Calculate excess return
    daily["RET"] = pd.to_numeric(daily["RET"], errors="coerce")
    daily["ret_exc"] = daily["RET"] - daily["rf_daily"]

    # Add end-of-month
    daily["eom"] = daily["date"] + pd.offsets.MonthEnd(0)

    # Drop columns you no longer need
    daily.drop(columns=["rf_daily", "PRCCD", "AJEXDI", "TRFD"], inplace=True)

    return daily
