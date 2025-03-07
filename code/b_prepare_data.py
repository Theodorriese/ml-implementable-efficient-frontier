# Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from a_general_functions import size_screen_fun, addition_deletion_fun, long_horizon_ret
import gc
import openpyxl
import datetime as dt

# data_path = r"C:\Master"


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
        skiprows=3,  # Skip metadata rows
        usecols=["yyyymm", "RF"],  # Select relevant columns
        sep=",",  # Ensure correct delimiter
    )

    risk_free = risk_free[risk_free["yyyymm"].astype(str).str.match(r"^\d{6}$")]
    risk_free["RF"] = pd.to_numeric(risk_free["RF"], errors="coerce")
    risk_free["rf"] = risk_free["RF"] / 100

    # Generate end-of-month date ('eom') using 'yyyymm'
    risk_free["eom_m"] = pd.to_datetime(risk_free["yyyymm"].astype(str) + "01", format="%Y%m%d") + pd.offsets.MonthEnd(0)

    # Calculate last business day of the same month
    # risk_free["eom_bd"] = risk_free["eom"].apply(
    #     lambda x: x - pd.offsets.BDay(0) if x.weekday() < 5 else x - pd.offsets.BDay(1))

    # return risk_free[["eom", "rf", "eom_bd"]]

    return risk_free[["eom_m", "rf"]]


# Function to load and preprocess market data
def load_market_data(data_path):
    market = pd.read_csv(os.path.join(data_path, "market_returns.csv"))
    market["eom"] = pd.to_datetime(market["eom"])
    return market[market["excntry"] == "USA"][["eom", "mkt_vw_exc"]]


# Wealth function
def wealth_func(wealth_end, end, market, risk_free):
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
    # Load Cluster Labels
    cluster_labels = pd.read_csv(os.path.join(data_path, "Cluster Labels.csv"))
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


def load_monthly_data(data_path, settings, risk_free):
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


# Prepare data - stock characteristics
def preprocess_chars(data_path, features, settings, data_ret_ld1, wealth):
    cols_to_use = ["id", "eom", "sic", "ff49", "size_grp", "me", "crsp_exchcd", "rvol_252d", "dolvol_126d"] + features
    chars = pd.read_csv(os.path.join(data_path, "usa.csv"), usecols=cols_to_use)
    chars = chars[chars["id"] <= 99999] # Filter only CRSP observations

    # Add useful columns
    chars["dolvol"] = chars["dolvol_126d"]
    chars["lambda"] = 2 / chars["dolvol"] * settings["pi"]
    chars["rvol_m"] = chars["rvol_252d"] * np.sqrt(21)

    chars["eom"] = pd.to_datetime(chars["eom"])
    # data_ret_ld1["eom_m"] = pd.to_datetime(data_ret_ld1["eom"])

    ### TEMP MERGE - FIX ALL THE EOM NAMES
    # Convert columns to datetime for proper merging
    chars["eom"] = pd.to_datetime(chars["eom"])
    # data_ret_ld1["eom_m"] = pd.to_datetime(data_ret_ld1["eom_m"])
    # wealth["eom_ret"] = pd.to_datetime(wealth["eom_ret"])

    chars.rename(columns={"eom": "eom_m"}, inplace=True)
    chars = chars.merge(data_ret_ld1, on=["id", "eom_m"], how="left")
    chars.drop(columns=["eom"], inplace=True)
    chars.rename(columns={"eom_m": "eom_ret"}, inplace=True)
    chars = chars.merge(wealth[["eom_ret", "mu_ld1"]], on="eom_ret", how="left")
    chars.rename(columns={"eom_ret": "eom"}, inplace=True)

    # # Exchange code screen - OBS this screens for only NYSE stocks - change if needed
    # if settings["screens"]["nyse_stocks"]:
    #     # Ensure 'crsp_exchcd' exists and calculate exclusion percentage
    #     exclude_pct = round(((chars["crsp_exchcd"] != 1).astype(int).mean()) * 100, 2)
    #     print(f"NYSE stock screen excludes {exclude_pct}% of the observations")
    #     chars = chars[chars["crsp_exchcd"] == 1]
    #
    # print(chars["crsp_exchcd"].head())  # Preview the first few rows

    return chars


# Date screen
def filter_chars(chars, settings):
    # date_exclusion_pct = round(((chars["eom"] < settings["screens"]["start"]) |
    #                             (chars["eom"] > settings["screens"]["end"])).mean() * 100, 2)
    # print(f"Date screen excludes {date_exclusion_pct}% of the observations")
    
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
    # Count available features per row
    feat_available = chars[features].notna().sum(axis=1)

    # Calculate minimum required features per row
    min_feat = np.floor(len(features) * settings["screens"]["feat_pct"])

    # Compute exclusion percentage
    # feat_exclusion_pct = round((feat_available < min_feat).mean() * 100, 2)
    # print(f"At least {settings['screens']['feat_pct'] * 100}% of feature excludes {feat_exclusion_pct}% of the observations")

    # Filter dataset
    chars = chars[feat_available >= min_feat]

    # Summary statistics
    final_obs_pct = round((len(chars) / n_start) * 100, 2)
    final_market_cap_pct = round((chars["me"].sum() / me_start) * 100, 2)
    print(f"In total, the final dataset has {final_obs_pct}% of the observations and {final_market_cap_pct}% of the market cap in the post {settings['screens']['start']} data")

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
    chars["valid_data"] = True # Initialize valid_data column
    # Compute lookback period
    lb = pf_set["lb_hor"] + 1  # Plus 1 to align with last signal of previous portfolio
    chars["eom_lag"] = chars.groupby("id")["eom"].shift(lb) # Compute lagged `eom`

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


def show_investable_universe(chars):
    """
    Plot the number of valid stocks over time.

    Parameters:
        chars (pd.DataFrame): The dataset containing stock data.
    """
    investable_counts = chars[chars["valid"] == True].groupby("eom").size()

    plt.figure(figsize=(10, 5))
    plt.scatter(investable_counts.index, investable_counts.values, alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='dashed')
    plt.xlabel("End of Month (eom)")
    plt.ylabel("Valid Stocks")
    plt.title("Investable Universe Over Time")
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


# Function to load and preprocess daily returns
def load_daily_returns(data_path, chars, risk_free):
    """
    Load and preprocess daily returns data.

    Parameters:
        data_path (str): Path to the dataset directory.
        chars (pd.DataFrame): Processed characteristics data with valid stocks.
        risk_free (pd.DataFrame): DataFrame containing risk-free rates with columns ['eom_m', 'rf'].

    Returns:
        pd.DataFrame: Preprocessed daily returns data with calculated excess returns.
    """
    # Load daily returns data
    daily = pd.read_csv(
        os.path.join(data_path, "usa_dsf.csv"),
        usecols=["PERMNO", "date", "RET"]
    )

    # Rename columns for consistency
    daily.rename(columns={"PERMNO": "id", "RET": "ret"}, inplace=True)

    # Filter for valid data (non-NA returns and IDs <= 99999)
    daily = daily[daily["ret"].notna() & daily["id"].le(99999)]
    valid_ids = chars.loc[chars["valid"], "id"].unique()  # Get valid stock IDs from `chars`
    daily = daily[daily["id"].isin(valid_ids)]

    # Convert `date` to datetime
    daily["date"] = pd.to_datetime(daily["date"], format="%d/%m/%Y")

    # Prepare `risk_free` data for merging
    risk_free.rename(columns={"eom_m": "date"}, inplace=True)
    risk_free["date"] = pd.to_datetime(risk_free["date"], format="%Y-%m-%d")
    risk_free["month"] = risk_free["date"].dt.to_period("M")

    # Calculate daily risk-free rate
    risk_free["days_in_month"] = risk_free["date"].dt.daysinmonth  # Number of days in each month
    risk_free["rf_daily"] = risk_free["rf"] / risk_free["days_in_month"]  # Convert to daily rate

    # Map monthly risk-free rates to daily data
    daily["month"] = daily["date"].dt.to_period("M")
    daily = daily.merge(risk_free[["month", "rf_daily"]], on="month", how="left")
    daily.drop(columns=["month"], inplace=True)  # Clean up temporary column

    # Ensure `ret` is numeric
    daily["ret"] = pd.to_numeric(daily["ret"], errors="coerce")

    # Calculate excess return
    daily["ret_exc"] = daily["ret"] - daily["rf_daily"]

    # Add `eom` column (end-of-month)
    daily["eom"] = daily["date"] + pd.offsets.MonthEnd(0)

    # Optional: Drop unnecessary columns
    daily.drop(columns=["rf_daily"], inplace=True)

    return daily


# Function to load and preprocess daily returns from a pickle file
def load_daily_returns_pkl(data_path, chars, risk_free):
    """
    Load and preprocess daily returns data from a pickle file.

    Parameters:
        data_path (str): Path to the dataset directory.
        chars (pd.DataFrame): Processed characteristics data with valid stocks.
        risk_free (pd.DataFrame): DataFrame containing risk-free rates with columns ['eom_m', 'rf'].

    Returns:
        pd.DataFrame: Preprocessed daily returns data with calculated excess returns.
    """
    # Load daily returns data from a pickle file
    daily = pd.read_pickle(os.path.join(data_path, "usa_dsf.pkl"))

    # Rename columns for consistency
    daily.rename(columns={"PERMNO": "id", "RET": "ret"}, inplace=True)

    # Filter for valid data (non-NA returns and IDs <= 99999)
    daily = daily[daily["ret"].notna() & daily["id"].le(99999)]
    valid_ids = chars.loc[chars["valid"], "id"].unique()  # Get valid stock IDs from `chars`
    daily = daily[daily["id"].isin(valid_ids)]

    # Convert `date` to datetime
    daily["date"] = pd.to_datetime(daily["date"], format="%Y-%m-%d")  # Pickle already preserves date format

    # Prepare `risk_free` data for merging
    risk_free.rename(columns={"eom_m": "date"}, inplace=True)
    risk_free["date"] = pd.to_datetime(risk_free["date"], format="%Y-%m-%d")
    risk_free["month"] = risk_free["date"].dt.to_period("M")

    # Calculate daily risk-free rate
    risk_free["days_in_month"] = risk_free["date"].dt.daysinmonth  # Number of days in each month
    risk_free["rf_daily"] = risk_free["rf"] / risk_free["days_in_month"]  # Convert to daily rate

    # Map monthly risk-free rates to daily data
    daily["month"] = daily["date"].dt.to_period("M")
    daily = daily.merge(risk_free[["month", "rf_daily"]], on="month", how="left")
    daily.drop(columns=["month"], inplace=True)  # Clean up temporary column

    # Ensure `ret` is numeric
    daily["ret"] = pd.to_numeric(daily["ret"], errors="coerce")

    # Calculate excess return
    daily["ret_exc"] = daily["ret"] - daily["rf_daily"]

    # Add `eom` column (end-of-month)
    daily["eom"] = daily["date"] + pd.offsets.MonthEnd(0)

    # Optional: Drop unnecessary columns
    daily.drop(columns=["rf_daily"], inplace=True)

    return daily
