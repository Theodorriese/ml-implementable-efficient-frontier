import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp
from sklearn.linear_model import LinearRegression


def plot_alpha_decay_cumulative_continuous(chars, features, output_path):
    """
    Plot cumulative alpha returns using cross-sectional regression (standardized inputs).

    Each month's alpha is the slope from regressing returns on the z-scored feature.
    """
    chars = chars.sort_values(by=["eom", "id"])
    alpha_df = []

    for feature in features:
        # Standardize feature within month
        chars["signal_std"] = chars.groupby("eom")[feature].transform(
            lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else np.nan
        )

        # Run monthly regression: ret_ld1 ~ z(feature)
        monthly_alpha = []
        for date, group in chars.groupby("eom"):
            if group["signal_std"].notna().sum() >= 10 and group["ret_ld1"].notna().sum() >= 10:
                X = group[["signal_std"]].values
                y = group["ret_ld1"].values
                model = LinearRegression().fit(X, y)
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

    # Optional summary table
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
