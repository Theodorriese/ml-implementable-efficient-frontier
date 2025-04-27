import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# How do weigths differ for PF-ML, Market, and Markowitz

def combine_portfolio_weights(static, pfml, mkt, tpf, chars, date):
    """
    Combine portfolio weights for Static-ML, Portfolio-ML, Market, and Markowitz-ML on a given date.

    Parameters:
        static (pd.DataFrame): Static-ML portfolio data containing weights.
        pfml (pd.DataFrame): Portfolio-ML portfolio data containing weights.
        mkt (pd.DataFrame): Market portfolio data containing weights.
        tpf (pd.DataFrame): Tangency portfolio data containing weights.
        chars (pd.DataFrame): Characteristics data containing 'id', 'eom', and 'dolvol'.
        date (str): The target date for combining weights ('YYYY-MM-DD').

    Returns:
        pd.DataFrame: A combined DataFrame of portfolio weights and characteristics.
    """
    # Combine weights from all portfolio types
    date = pd.to_datetime(date)
    weights = pd.concat([
        static["w"].loc[static["w"]["eom"] == date].assign(type="Static-ML*"),
        pfml["w"].loc[pfml["w"]["eom"] == date].assign(type="Portfolio-ML"),
        mkt["w"].loc[mkt["w"]["eom"] == date].assign(type="Market"),
        tpf["w"].loc[tpf["w"]["eom"] == date].assign(type="Markowitz-ML")
    ], ignore_index=True)

    # Drop the existing 'dolvol' to make space for the new one getting merged
    weights.drop(columns=["dolvol"], inplace=True, errors="ignore")

    # Merge with characteristics data to bring in the 'dolvol' from 'chars'
    weights["id"] = weights["id"].astype(str)
    chars["id"] = chars["id"].astype(str)
    weights = weights.merge(chars[["id", "eom", "dolvol"]], on=["id", "eom"], how="left")

    # Filter for consistent observation counts
    weights["n"] = weights.groupby(["id", "eom"])["id"].transform("size")
    weights = weights[weights["n"] == weights["n"].max()]

    return weights


def calculate_and_plot_weight_differences(weights, settings, save_path=None):
    """
    Analyze and visualize weight differences across portfolio types.

    Parameters:
        weights (pd.DataFrame): Combined weights dataframe.
        settings (dict): Settings dictionary containing `seed_no`.
        save_path (str): Path to save the plot (optional).

    Returns:
        None
    """
    # Set random seed and sample 100 IDs
    np.random.seed(settings["seed_no"])
    sample_ids = np.random.choice(weights["id"].unique(), size=100, replace=False)

    # Calculate Spearman correlation for each portfolio type
    cor_results = weights.groupby("type").apply(
        lambda x: spearmanr(x["dolvol"], x["w"].abs())[0]
    )
    print(cor_results)

    # Plot absolute weights vs. dollar volume rank for sampled IDs
    subset = weights.loc[(weights["id"].isin(sample_ids)) & (weights["type"].isin(["Markowitz-ML", "Portfolio-ML"]))]

    subset = subset.sort_values("dolvol", ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=subset,
        x="id",
        y="w",
        hue="type",
        estimator=np.sum
    )
    plt.xticks(rotation=90)
    plt.xlabel("ID")
    plt.ylabel("Absolute Portfolio Weight")
    plt.title("Portfolio Weights by Dollar Volume Rank")
    plt.legend(title="Type")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    # Add dollar volume rank and plot
    weights["dolvol_rank"] = weights.groupby("type")["dolvol"].rank()
    sns.lmplot(
        data=weights,
        x="dolvol_rank",
        y="w",
        hue="type",
        col="type",
        col_wrap=3,
        facet_kws={"sharey": False},
        scatter_kws={"s": 10},
        line_kws={"color": "red"}
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path.replace(".png", "_rank.png"))
    else:
        plt.show()


# Portfolio analysis ---------------------------------------------
def portfolio_analysis(static, pfml, mkt, tpf, chars, colours_theme, save_path=None):
    """
    Perform portfolio analysis, including dollar volume sorting and weight visualization.

    Parameters:
        static (pd.DataFrame): Static-ML portfolio data containing weights.
        pfml (pd.DataFrame): Portfolio-ML portfolio data containing weights.
        mkt (pd.DataFrame): Market portfolio data containing weights.
        tpf (pd.DataFrame): Tangency portfolio data containing weights.
        chars (pd.DataFrame): Characteristics data containing 'id', 'eom', and 'dolvol'.
        colours_theme (list): List of colors for visualizations.
        save_path (str): Path to save the plot (optional).

    Returns:
        None
    """
    # Combine weights from all portfolio types
    weights = pd.concat([
        static["w"].assign(type="Static-ML*"),
        pfml["w"].assign(type="Portfolio-ML"),
        mkt["w"].assign(type="Market"),
        tpf["w"].assign(type="Markowitz-ML")
    ], ignore_index=True)

    # Drop the existing 'dolvol' to make space for the new one getting merged
    weights.drop(columns=["dolvol"], inplace=True, errors="ignore")

    # Merge with characteristics data to bring in the 'dolvol' from 'chars'
    weights = weights.merge(chars[["id", "eom", "dolvol"]], on=["id", "eom"], how="left")

    # Filter for consistent observation counts
    weights["n"] = weights.groupby(["id", "eom"])["id"].transform("size")
    weights = weights[weights["n"] == weights["n"].max()]

    # Create dollar volume portfolios
    weights["dv_pf"] = weights.groupby(["type", "eom"])["dolvol"].rank(pct=True).apply(lambda x: np.ceil(x * 10))

    # Average absolute weights by type, eom, and dv_pf
    avg_weights = (
        weights.groupby(["type", "eom", "dv_pf"])["w"]
        .apply(lambda x: np.mean(np.abs(x)))
        .reset_index(name="w_abs")
    )

    avg_weights_by_pf = (
        avg_weights.groupby(["type", "dv_pf"])["w_abs"]
        .mean()
        .reset_index()
    )

    # Filter for relevant portfolio types
    filtered_weights = avg_weights_by_pf[avg_weights_by_pf["type"].isin(["Markowitz-ML", "Portfolio-ML", "Market"])]

    # Plot average absolute weights by dollar volume portfolios
    plt.figure(figsize=(12, 6))
    sns.barplot(data=filtered_weights, x="dv_pf", y="w_abs", hue="type", palette=colours_theme[:3])
    plt.xlabel("Dollar volume sorted portfolios (1=low)")
    plt.ylabel("Average Absolute Weight")
    plt.title("Average Absolute Weights by Dollar Volume Portfolios")
    plt.legend(title="Portfolio Type")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    # Add long/short classification
    weights["long"] = np.where(weights["w"] >= 0, "Long positions", "Short positions")

    # Calculate average weights for long and short positions
    long_short_weights = (
        weights.groupby(["type", "eom", "dv_pf", "long"])["w"]
        .mean()
        .reset_index(name="w_abs")
    )

    avg_long_short_weights = (
        long_short_weights.groupby(["type", "dv_pf", "long"])["w_abs"]
        .mean()
        .reset_index()
    )

    # Filter for relevant portfolio types
    filtered_long_short_weights = avg_long_short_weights[
        avg_long_short_weights["type"].isin(["Markowitz-ML", "Portfolio-ML", "Market"])]

    # Plot long/short weights by dollar volume portfolios
    plt.figure(figsize=(12, 6))
    sns.barplot(data=filtered_long_short_weights, x="dv_pf", y="w_abs", hue="long", palette=colours_theme[:2])
    plt.xlabel("Dollar volume sorted portfolios (1=low)")
    plt.ylabel("Average Stock Weight")
    plt.title("Long/Short Weights by Dollar Volume Portfolios")
    plt.legend(title="")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path.replace(".png", "_long_short.png"))
    else:
        plt.show()
