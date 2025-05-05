import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
from typing import List


def plot_alpha_decay_cumulative_continuous(chars, features, output_path):
    """
    Plot cumulative alpha returns using cross-sectional regression (features already standardized).
    Each month's alpha is the slope from regressing returns on the standardized feature.
    Applies filter to include only stocks with at least 5 years of data.
    """
    chars = chars.sort_values(by=["eom", "id"])

    # Filter: keep only stocks with ≥ 60 months of data
    valid_ids = chars.groupby("id").size()
    valid_ids = valid_ids[valid_ids >= 60].index
    chars = chars[chars["id"].isin(valid_ids)]

    alpha_df = []

    for feature in features:
        monthly_alpha = []
        for date, group in chars.groupby("eom"):
            x = group[[feature]].values
            y = group["ret_ld1"].values
            mask = (
                np.isfinite(x).flatten() &
                np.isfinite(y) &
                (x.flatten() != 0.5)  # 0.5 indicates imputed/missing
            )
            if mask.sum() >= 10:
                model = LinearRegression().fit(x[mask], y[mask])
                alpha = model.coef_[0]
            else:
                alpha = np.nan
            monthly_alpha.append((date, alpha))

        monthly_alpha = pd.Series(dict(monthly_alpha), name=feature)
        alpha_df.append(monthly_alpha)

    alpha_matrix = pd.concat(alpha_df, axis=1)
    cumulative = (1 + alpha_matrix.fillna(0)).cumprod()

    # Plot
    plt.figure(figsize=(12, 6))
    for col in cumulative.columns:
        plt.plot(cumulative.index, cumulative[col], label=col)
    plt.title("Cumulative Alpha Return (Regression-Based)")
    plt.xlabel("End of Month")
    plt.ylabel("Cumulative Growth of $1")
    plt.axhline(1, color='black', linestyle='--')
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_path = os.path.join(output_path, "alpha_decay_regression_cont.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    # Summary table
    summary = alpha_matrix.mean().to_frame("mean_alpha")
    summary["std"] = alpha_matrix.std()
    summary["t_stat"] = summary["mean_alpha"] / summary["std"] * np.sqrt(alpha_matrix.notna().sum())
    return summary



def plot_alpha_decay_rolling_tstat(chars, features, output_path, window=24):
    """
    Plot rolling t-statistics for long-short (top vs bottom quintile) portfolios.
    """
    chars = chars.sort_values(by=["eom", "id"])
    decay_df = []

    for feature in features:
        chars["signal_rank"] = chars.groupby("eom")[feature].rank(pct=True)
        chars["qcut"] = pd.qcut(chars["signal_rank"], q=5, labels=False, duplicates="drop")
        chars["long"] = chars["qcut"] == 4
        chars["short"] = chars["qcut"] == 0

        # Monthly spread
        monthly_spread = chars.groupby("eom").apply(
            lambda x: x.loc[x["long"], "ret_ld1"].mean() - x.loc[x["short"], "ret_ld1"].mean()
            if x["long"].sum() >= 5 and x["short"].sum() >= 5 else np.nan
        )
        decay_df.append(monthly_spread.rename(feature))

    decay_matrix = pd.concat(decay_df, axis=1)

    # Rolling t-statistics
    tstat_df = decay_matrix.rolling(window=window).apply(
        lambda x: ttest_1samp(x.dropna(), 0).statistic if len(x.dropna()) >= 10 else np.nan
    )

    # Plot
    plt.figure(figsize=(12, 6))
    for col in tstat_df.columns:
        plt.plot(tstat_df.index, tstat_df[col], label=col)
    plt.axhline(0, color='black', linestyle='--')
    plt.axhline(1.96, color='grey', linestyle='--', linewidth=0.7)
    plt.axhline(-1.96, color='grey', linestyle='--', linewidth=0.7)
    plt.title(f"Rolling T-Statistic (window={window} months)")
    plt.xlabel("End of Month")
    plt.ylabel("T-statistic")
    plt.legend()
    plt.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.12)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_path = os.path.join(output_path, "alpha_tstat.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    return tstat_df


################################# NEXT #################################
def compute_signal_rank_stability(chars: pd.DataFrame, features: List[str], output_path: str, max_lag: int = 6) -> pd.DataFrame:
    """
    Computes signal rank stability for each feature across time lags.

    Parameters:
        chars (pd.DataFrame): DataFrame with ['id', 'eom'] and standardized feature columns.
        features (List[str]): List of standardized feature names.
        output_path (str): Path to save the resulting plot.
        max_lag (int): Number of months to lag when computing correlation.

    Returns:
        pd.DataFrame: DataFrame of average Spearman rank correlations per lag per feature.
    """
    chars = chars.sort_values(by=["id", "eom"]).copy()
    chars["eom"] = pd.to_datetime(chars["eom"])

    results = []

    for feature in features:
        temp = chars[["id", "eom", feature]].copy()
        temp["rank"] = temp.groupby("eom")[feature].rank(method="average")

        for lag in range(1, max_lag + 1):
            # Create lagged version
            temp["eom_lag"] = temp["eom"] + pd.DateOffset(months=lag)
            merged = temp.merge(temp[["id", "eom", "rank"]], left_on=["id", "eom_lag"], right_on=["id", "eom"], suffixes=("", "_lag"))
            mask = merged["rank"].notna() & merged["rank_lag"].notna()

            if mask.sum() > 0:
                rho, _ = spearmanr(merged.loc[mask, "rank"], merged.loc[mask, "rank_lag"])
            else:
                rho = np.nan

            results.append({"feature": feature, "lag": lag, "rank_corr": rho})

    df_result = pd.DataFrame(results)

    # Plot
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")
    for feature in df_result["feature"].unique():
        sub = df_result[df_result["feature"] == feature]
        plt.plot(sub["lag"], sub["rank_corr"], marker="o", label=feature)

    plt.title("Signal Rank Stability Over Time")
    plt.xlabel("Lag (Months)")
    plt.ylabel("Avg. Spearman Rank Correlation")
    plt.axhline(0, color='black', linestyle='--', linewidth=0.7)
    plt.legend()
    plt.tight_layout()

    os.makedirs(output_path, exist_ok=True)
    plot_path = os.path.join(output_path, "signal_rank_decay.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    return df_result.pivot(index="lag", columns="feature", values="rank_corr")