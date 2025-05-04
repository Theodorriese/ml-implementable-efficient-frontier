import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from matplotlib.ticker import LogLocator, NullFormatter
import os

from matplotlib import rcParams
rcParams.update({
    'font.size': 14,           # Base font size
    'axes.titlesize': 15,      # Axes title
    'axes.labelsize': 14,      # Axes labels
    'xtick.labelsize': 12,     # X tick labels
    'ytick.labelsize': 12,     # Y tick labels
    'legend.fontsize': 14,     # Legend text
    'figure.titlesize': 18     # Figure title
})
colours_theme = ["steelblue", "darkorange", "gray"]


# How do weigths differ
def combine_portfolio_weights(static, pfml, mkt, tpf, chars, date):
    date = pd.to_datetime(date)

    def extract_weights(source, label):
        df = source["w"].loc[source["w"]["eom"] == date].copy()
        df["type"] = label
        return df

    # Extract and label each portfolio type
    static_df = extract_weights(static, "Static-ML*")
    pfml_df = extract_weights(pfml, "Portfolio-ML")
    mkt_df = extract_weights(mkt, "Market")
    tpf_df = extract_weights(tpf, "Markowitz-ML")

    # Ensure all have the same columns (take intersection or union as needed)
    common_cols = list(set(static_df.columns) & set(pfml_df.columns) & set(mkt_df.columns) & set(tpf_df.columns))
    weights = pd.concat([
        static_df[common_cols],
        pfml_df[common_cols],
        mkt_df[common_cols],
        tpf_df[common_cols]
    ], ignore_index=True)

    # Enrich with dolvol (merge cleanly)
    weights["id"] = weights["id"].astype(str)
    chars["id"] = chars["id"].astype(str)
    weights = weights.merge(chars[["id", "eom", "dolvol"]], on=["id", "eom"], how="left")

    # Filter only fully observed rows (optional)
    weights["n"] = weights.groupby(["id", "eom"])["id"].transform("size")
    weights = weights[weights["n"] == weights["n"].max()]

    return weights


def calculate_and_plot_weight_differences(weights, settings, save_path=None):
    """
    Analyze and visualize weight differences across portfolio types with dollar volume context.

    Parameters:
        weights (pd.DataFrame): Combined weights dataframe.
        settings (dict): Settings dictionary containing `seed_no`.
        save_path (str): Path to save the plot (optional).
    """
    # Set random seed and sample IDs
    np.random.seed(settings["seed_no"])
    sample_ids = np.random.choice(weights["id"].unique(), size=70, replace=False)

    # Calculate Spearman correlation for each portfolio type
    cor_results = weights.groupby("type").apply(
        lambda x: spearmanr(x["dolvol"], x["w"].abs())[0]
    )
    print(cor_results)

    # Filter subset for barplot
    subset = weights.loc[
        (weights["id"].isin(sample_ids)) &
        (weights["type"].isin(["Portfolio-ML", "Static-ML*"]))
        ]

    # Get average dolvol per ID and sort descending
    avg_dolvol = (
        subset.groupby("id")["dolvol"]
        .mean()
        .sort_values(ascending=False)
    )

    # Ensure consistent ID ordering
    ordered_ids = avg_dolvol.index.astype(str).tolist()
    subset["id"] = subset["id"].astype(str)
    subset["id"] = pd.Categorical(subset["id"], categories=ordered_ids, ordered=True)

    # Plot
    fig, ax1 = plt.subplots(figsize=(14, 6))

    sns.barplot(
        data=subset,
        x="id",
        y="w",
        hue="type",
        estimator=np.sum,
        ax=ax1
    )

    ax1.set_xlabel("ID")
    ax1.set_ylabel("Absolute Portfolio Weight")
    ax1.set_title("Portfolio Weights by Dollar Volume Rank")
    ax1.legend(title="Type")
    ax1.tick_params(axis="x", rotation=90)
    ax1.grid(False)

    # Secondary axis for log-scaled dollar volume
    ax2 = ax1.twinx()
    ax2.set_yscale("log")
    ax2.set_ylabel("Dollar Volume (Millions, log scale)")
    ax2.plot(
        range(len(avg_dolvol)),
        avg_dolvol.values / 1e6,
        "o",
        color="gray",
        markersize=4
    )
    ax2.yaxis.set_major_locator(LogLocator(base=10.0))
    ax2.yaxis.set_minor_formatter(NullFormatter())
    ax2.tick_params(axis="y", which="minor", length=0)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    # Add dollar volume rank and plot regression by type
    weights["dolvol_rank"] = weights.groupby("type")["dolvol"].rank()

    g = sns.lmplot(
        data=weights,
        x="dolvol_rank",
        y="w",
        hue="type",
        col="type",
        col_wrap=2,  # Change from 3 to 2 for 2x2 layout
        facet_kws={"sharey": False},
        scatter_kws={"s": 10},
        line_kws={"color": "red"}
    )
    g.set_axis_labels("Dollar Volume Rank", "Portfolio Weight")
    g.fig.tight_layout()

    if save_path:
        g.savefig(save_path.replace(".png", "_rank.png"))
    else:
        plt.show()


def calculate_and_plot_weight_differences_rel(weights, types_to_compare, settings, save_path=None, sample_ids=None):
    """
    Analyze and visualize relative weight differences across specified portfolio types,
    replacing IDs with numeric labels and exporting the ID map if saving.

    Parameters:
        weights (pd.DataFrame): Combined weights dataframe.
        types_to_compare (list): List of two portfolio types to compare (e.g. ["Portfolio-ML", "Markowitz-ML"]).
        settings (dict): Settings dictionary containing 'seed_no'.
        save_path (str): Optional path to save the figure (CSV is saved with the same base name).
        sample_ids (list): Optional list of IDs to use instead of random sampling.
    """
    import os

    # Use provided IDs or sample new ones
    if sample_ids is None:
        np.random.seed(settings["seed_no"])
        sample_ids = np.random.choice(weights["id"].unique(), size=70, replace=False)

    # Filter subset
    subset = weights.loc[
        (weights["id"].isin(sample_ids)) & (weights["type"].isin(types_to_compare))
    ].copy()

    # Calculate average dollar volume and order IDs by it
    avg_dolvol = (
        subset.groupby("id")["dolvol"]
        .mean()
        .sort_values(ascending=True)
    )
    ordered_ids = avg_dolvol.index.tolist()

    # Create numeric labels
    id_mapping = pd.DataFrame({
        "stock_number": range(1, len(ordered_ids) + 1),
        "id": ordered_ids
    })
    id_dict = dict(zip(ordered_ids, id_mapping["stock_number"]))

    # Apply numeric labels
    subset["stock_number"] = subset["id"].map(id_dict)

    # Normalize weights within each type
    subset["w_rel"] = subset.groupby("type")["w"].transform(lambda x: x / x.abs().sum())

    # Create plot
    fig, ax1 = plt.subplots(figsize=(14, 6))
    sns.barplot(
        data=subset,
        x="stock_number",
        y="w_rel",
        hue="type",
        hue_order=types_to_compare,
        estimator=np.sum,
        ax=ax1
    )

    ax1.set_xlabel("Stock")
    ax1.set_ylabel("Relative Portfolio Weight")
    ax1.set_title(f"Relative Portfolio Weights by Dollar Volume Rank: {types_to_compare[0]} vs {types_to_compare[1]}")
    ax1.tick_params(axis="x", rotation=90)
    ax1.grid(False)

    ax1.legend(
        title="Type",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        borderaxespad=0.0,
        ncol=len(types_to_compare)
    )

    # Add secondary y-axis for dollar volume
    ax2 = ax1.twinx()
    ax2.set_yscale("log")
    ax2.set_ylabel("Dollar Volume (Millions, log scale)")
    ax2.plot(
        subset.drop_duplicates("stock_number").sort_values("stock_number")["stock_number"],
        avg_dolvol.loc[ordered_ids].values / 1e6,
        "o",
        color="gray",
        markersize=4
    )

    ax2.yaxis.set_major_locator(LogLocator(base=10.0))
    ax2.yaxis.set_minor_formatter(NullFormatter())
    ax2.tick_params(axis="y", which="minor", length=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        base, _ = os.path.splitext(save_path)
        id_mapping.to_csv(base + "_id_map.csv", index=False)
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
    """

    # Combine weights from all portfolio types
    weights = pd.concat([
        static["w"].assign(type="Static-ML*"),
        pfml["w"].assign(type="Portfolio-ML"),
        mkt["w"].assign(type="Market"),
        tpf["w"].assign(type="Markowitz-ML")
    ], ignore_index=True)

    # Drop old 'dolvol' column if present
    weights.drop(columns=["dolvol"], inplace=True, errors="ignore")

    # Ensure 'id' is the same type before merging
    weights["id"] = weights["id"].astype(str)
    chars["id"] = chars["id"].astype(str)

    # Merge in dollar volume info
    weights = weights.merge(chars[["id", "eom", "dolvol"]], on=["id", "eom"], how="left")

    # Normalize weights within each portfolio (type + eom)
    weights["w_rel"] = weights.groupby(["type", "eom"])["w"].transform(lambda x: x / x.abs().sum())

    # Filter for consistent observation counts
    weights["n"] = weights.groupby(["id", "eom"])["id"].transform("size")
    weights = weights[weights["n"] == weights["n"].max()]

    # Create dollar volume sorted portfolios (deciles)
    weights["dv_pf"] = weights.groupby(["type", "eom"])["dolvol"].rank(pct=True).apply(lambda x: np.ceil(x * 10))

    # Average relative weights by type, eom, and dv_pf
    avg_weights = (
        weights.groupby(["type", "eom", "dv_pf"])["w_rel"]
        .apply(lambda x: np.mean(np.abs(x)))
        .reset_index(name="w_abs")
    )

    avg_weights_by_pf = (
        avg_weights.groupby(["type", "dv_pf"])["w_abs"]
        .mean()
        .reset_index()
    )

    # Filter for selected portfolio types
    filtered_weights = avg_weights_by_pf[avg_weights_by_pf["type"].isin(["Markowitz-ML", "Portfolio-ML", "Market"])]

    # Plot average relative weights
    plt.figure(figsize=(12, 6))
    sns.barplot(data=filtered_weights, x="dv_pf", y="w_abs", hue="type", palette=colours_theme[:3])
    plt.xlabel("Dollar volume sorted portfolios (1=low)")
    plt.ylabel("Average Relative Weight")
    plt.title("Average Relative Weights by Dollar Volume Portfolios")
    plt.legend(title="Portfolio Type")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    # Classify long vs short positions
    weights["long"] = np.where(weights["w"] >= 0, "Long positions", "Short positions")

    # Calculate average relative weights for long/short
    long_short_weights = (
        weights.groupby(["type", "eom", "dv_pf", "long"])["w_rel"]
        .mean()
        .reset_index(name="w_rel_avg")
    )

    avg_long_short_weights = (
        long_short_weights.groupby(["type", "dv_pf", "long"])["w_rel_avg"]
        .mean()
        .reset_index()
    )

    # List of portfolio types to plot separately
    long_short_types = ["Markowitz-ML", "Portfolio-ML", "Static-ML*"]

    for pf_type in long_short_types:
        pf_data = avg_long_short_weights[avg_long_short_weights["type"] == pf_type]

        plt.figure(figsize=(12, 6))
        sns.barplot(data=pf_data, x="dv_pf", y="w_rel_avg", hue="long", palette=colours_theme[:2])
        plt.xlabel("Dollar volume sorted portfolios (1=low)")
        plt.ylabel("Average Relative Weight")
        plt.title(f"Long/Short Relative Weights by Dollar Volume: {pf_type}")
        plt.legend(title="")
        plt.tight_layout()

        if save_path:
            file_tag = pf_type.lower().replace("-", "").replace("*", "").replace(" ", "_")
            plt.savefig(save_path.replace(".png", f"_{file_tag}_long_short.png"))
        else:
            plt.show()

