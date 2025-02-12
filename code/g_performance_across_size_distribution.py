import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Download files --------------------------
def process_size_folder(size_category, size_folders, size_path):
    """
    Process a specific size category folder.

    Parameters:
        size_category (str): The size category (e.g., "all", "low80_high100").
        size_folders (list): List of folders in the size directory.
        size_path (str): Path to the size portfolios directory.

    Returns:
        pd.DataFrame: Combined portfolio data for the size category.
    """
    folder = next(
        (f for f in size_folders if size_category in f),
        None
    )
    if folder is None:
        return pd.DataFrame()  # Return empty DataFrame if folder not found

    # Read and combine portfolio files for the size category
    bms = pd.read_csv(os.path.join(size_path, folder, "bms.csv"))
    bms["eom_ret"] = pd.to_datetime(bms["eom_ret"])

    static_ml = pd.read_pickle(os.path.join(size_path, folder, "static-ml.pkl"))["pf"]
    multiperiod_ml = pd.read_pickle(os.path.join(size_path, folder, "multiperiod-ml.pkl"))["pf"]
    portfolio_ml = pd.read_pickle(os.path.join(size_path, folder, "portfolio-ml.pkl"))["pf"]

    combined = pd.concat([bms, static_ml, multiperiod_ml, portfolio_ml], ignore_index=True)
    combined["size"] = size_category
    return combined


def download_and_process_files():
    """
    Download and process portfolio data by size categories.

    Returns:
        pd.DataFrame: Combined DataFrame of portfolio data categorized by size.
    """
    # Define size categories and paths
    size_cuts = [
        "all",
        "low80_high100",
        "low60_high80",
        "low40_high60",
        "low20_high40",
        "low00_high20"
    ]
    size_path = "Data/Generated/Portfolios/Size/"
    size_folders = os.listdir(size_path)

    # Process all size categories
    pf_by_size = pd.concat(
        [process_size_folder(size, size_folders, size_path) for size in size_cuts],
        ignore_index=True
    )

    # Map size categories to descriptive labels
    size_mapping = {
        "all": "All",
        "low80_high100": "Largest (80-100)",
        "low60_high80": "Large (60-80)",
        "low40_high60": "Mid (40-60)",
        "low20_high40": "Small (20-40)",
        "low00_high20": "Smallest (00-20)"
    }
    pf_by_size["size"] = pf_by_size["size"].map(size_mapping)
    pf_by_size["size"] = pd.Categorical(
        pf_by_size["size"],
        categories=["All", "Largest (80-100)", "Large (60-80)", "Mid (40-60)", "Small (20-40)", "Smallest (00-20)"],
        ordered=True
    )

    # Adjust the 'type' column
    pf_by_size["type"] = pf_by_size["type"].replace({"Rank-Weighted": "Rank-ML"})

    return pf_by_size

# Performance -----------------------------------------------
def calculate_performance_summary(pf_by_size, pf_order, gamma_rel):
    """
    Calculate performance summary statistics for portfolios categorized by size.

    Parameters:
        pf_by_size (pd.DataFrame): DataFrame containing portfolio data categorized by size.
        pf_order (list): List defining the order of portfolio types.
        gamma_rel (float): Gamma value for relative risk aversion.

    Returns:
        pd.DataFrame: Summary statistics for portfolios categorized by size.
    """
    # Adjust type column to follow the specified order
    pf_by_size["type"] = pd.Categorical(pf_by_size["type"], categories=pf_order, ordered=True)

    # Sort by size, type, and eom_ret
    pf_by_size = pf_by_size.sort_values(by=["size", "type", "eom_ret"])

    # Calculate adjusted variance and utility
    pf_by_size["e_var_adj"] = pf_by_size.groupby(["size", "type"])["r"].transform(
        lambda x: (x - x.mean()) ** 2
    )
    pf_by_size["utility_t"] = pf_by_size["r"] - pf_by_size["tc"] - 0.5 * pf_by_size["e_var_adj"] * gamma_rel

    # Compute summary statistics
    pf_summary_size = (
        pf_by_size.groupby(["size", "type"])
        .agg(
            n=("r", "size"),
            inv=("inv", "mean"),
            shorting=("shorting", "mean"),
            turnover_notional=("turnover", "mean"),
            r=("r", lambda x: x.mean() * 12),
            sd=("r", lambda x: x.std() * np.sqrt(12)),
            sr_gross=("r", lambda x: x.mean() / x.std() * np.sqrt(12)),
            tc=("tc", lambda x: x.mean() * 12),
            r_tc=("r", lambda x: (x - pf_by_size.loc[x.index, "tc"]).mean() * 12),
            sr=("r", lambda x: (x - pf_by_size.loc[x.index, "tc"]).mean() / x.std() * np.sqrt(12)),
            obj=("r", lambda x: (x.mean() - 0.5 * x.var() * gamma_rel - pf_by_size.loc[x.index, "tc"].mean()) * 12),
        )
        .reset_index()
    )

    # Sort by size and type
    pf_summary_size = pf_summary_size.sort_values(by=["size", "type"])

    return pf_summary_size


# Plots -----------------------------------


def plot_realized_utility_by_size(pf_summary_size, colours_theme):
    """
    Plot realized utility by portfolio type and size.

    Parameters:
        pf_summary_size (pd.DataFrame): Summary statistics for portfolios categorized by size.
        colours_theme (list): List of colors for visualizations.

    Returns:
        None
    """
    # Adjust the `obj` column to cap extremely low values at -1000
    pf_summary_size["obj"] = pf_summary_size["obj"].apply(lambda x: max(x, -1000))

    # Create the plot
    plt.figure(figsize=(12, 8))
    g = sns.catplot(
        data=pf_summary_size,
        x="type",
        y="obj",
        col="size",
        kind="bar",
        palette=[colours_theme[0]],
        facet_kws={"sharey": True, "sharex": True},
        col_wrap=3,
    )

    for ax in g.axes.flat:
        ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)

    g.set_axis_labels("", "Realized Utility")
    g.set_titles(col_template="{col_name}")
    g.set(ylim=(-0.05, None))
    for ax in g.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    g.fig.tight_layout()

    for ax in g.axes.flat:
        ax.legend_.remove()

    # Show the plot
    plt.show()
