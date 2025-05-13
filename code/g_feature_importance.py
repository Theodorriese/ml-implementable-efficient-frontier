import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as mtick


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


def calculate_feature_importance(pf_set, latest_folder, save_path=None):
    """
    Calculate and visualize feature importance based on drop in realized utility
    from permuting theme features for Portfolio-ML and Markowitz-ML.

    Parameters:
        pf_set (dict): Portfolio settings, including 'gamma_rel'.
        latest_folder (str): Path to folder with 'tpf_cf_base.pkl' and 'pfml_cf_base.pkl'.
        save_path (str): Optional path to save the plot.
    """

    # Load data
    tpf_cf_base = pd.read_pickle(os.path.join(latest_folder, "tpf_cf_base.pkl"))
    pfml_cf_base = pd.read_pickle(os.path.join(latest_folder, "pfml_cf_base.pkl"))

    # --- Compute cf_obj for Portfolio-ML ---
    pfml_cf_ss = (
        pfml_cf_base
        .groupby(["type", "cluster"], as_index=False)
        .agg(r_mean=("r", "mean"), r_var=("r", "var"), tc_mean=("tc", "mean"))
    )
    pfml_cf_ss["cf_obj"] = (
        pfml_cf_ss["r_mean"]
        - 0.5 * pf_set["gamma_rel"] * pfml_cf_ss["r_var"]
        - pfml_cf_ss["tc_mean"]
    ) * 12

    bm_cf = pfml_cf_ss.loc[pfml_cf_ss["cluster"] == "bm", "cf_obj"].values[0]
    pfml_cf_ss["fi"] = bm_cf - pfml_cf_ss["cf_obj"]
    pfml_cf_ss["wealth"] = 1e10
    pfml_cf_ss = pfml_cf_ss[["type", "cluster", "fi", "wealth"]]

    # --- Compute cf_obj for Markowitz-ML (WITH transaction cost) ---
    tpf_cf_ss = (
        tpf_cf_base
        .groupby(["type", "cluster"], as_index=False)
        .agg(r_mean=("r", "mean"), r_var=("r", "var"), tc_mean=("tc", "mean"))
    )
    tpf_cf_ss["cf_obj"] = (
        tpf_cf_ss["r_mean"]
        - 0.5 * pf_set["gamma_rel"] * tpf_cf_ss["r_var"]
        - tpf_cf_ss["tc_mean"]
    ) * 12

    bm_cf_tpf = tpf_cf_ss.loc[tpf_cf_ss["cluster"] == "bm", "cf_obj"].values[0]
    tpf_cf_ss["fi"] = bm_cf_tpf - tpf_cf_ss["cf_obj"]
    tpf_cf_ss["wealth"] = 0
    tpf_cf_ss = tpf_cf_ss[["type", "cluster", "fi", "wealth"]]

    # --- Combine and clean ---
    df = pd.concat([pfml_cf_ss, tpf_cf_ss], ignore_index=True)
    df = df[df["cluster"] != "bm"]

    df["cluster"] = (
        df["cluster"]
        .str.replace("_", " ", regex=False)
        .str.replace("short term", "short-term", regex=False)
        .str.title()
    )

    df["type"] = pd.Categorical(
        df["type"],
        categories=["Portfolio-ML", "Multiperiod-ML*", "Markowitz-ML"],
        ordered=True
    )

    # Sort clusters by PFML importance
    sort_order = (
        df[df["type"] == "Portfolio-ML"]
        .set_index("cluster")["fi"]
        .sort_values()
    )
    ordered_clusters = sort_order.index.tolist()
    df["cluster"] = pd.Categorical(df["cluster"], categories=ordered_clusters, ordered=True)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

    # Sort plots by cluster to enforce bar order
    pfml_plot = df[df["type"] == "Portfolio-ML"].sort_values("cluster")
    marko_plot = df[df["type"] == "Markowitz-ML"].sort_values("cluster")

    axes[0].barh(pfml_plot["cluster"], pfml_plot["fi"], color="#1f77b4")
    axes[0].set_title("Portfolio-ML")
    axes[0].set_xlabel("Drop in realized utility")
    axes[0].invert_yaxis()

    axes[1].barh(marko_plot["cluster"], marko_plot["fi"], color="#2ca02c")
    axes[1].set_title("Markowitz-ML")
    axes[1].set_xlabel("Drop in realized utility")

    fig.suptitle("Feature Importance by Theme", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path:
        plt.savefig(save_path.replace(".png", "_ranked_fi.png"))
    else:
        plt.show()


def calc_group_stats(group):
    """
    Helper function for calculate_ief_summary function.
    """
    gamma = group["gamma_rel"].iloc[0]
    r = group["r"]
    tc = group["tc"]
    r_mean = r.mean()
    r_var = r.var()
    tc_mean = tc.mean()
    r_net = r_mean - tc_mean
    sd_annual = r.std() * np.sqrt(12)

    return pd.Series({
        "gamma_rel": gamma,
        "obj": (r_mean - 0.5 * r_var * gamma - tc_mean) * 12,
        "r_tc_annual": r_net * 12,
        "sd_annual": sd_annual,
        "sr_net": (r_net / r.std()) * np.sqrt(12) if r.std() != 0 else np.nan
    })



# Counterfactual EF without TC --------------------------
def plot_counterfactual(output_path, colours_theme, save_path=None):
    """
    Plot counterfactual efficient frontier (EF) without trading costs.

    Parameters:
        output_path (str): Path where the tpf_cf_base.pkl is stored.
        colours_theme (list): List of colors for visualizations.
        save_path (str): Path to save the plot (optional).

    Returns:
        None
    """
    tpf_cf_base_path = os.path.join(output_path, "tpf_cf_base.pkl")
    tpf_cf_base = pd.read_pickle(tpf_cf_base_path)

    # Calculate Sharpe ratio for each cluster
    tpf_cf_ss = (
        tpf_cf_base.groupby("cluster", as_index=False)
        .agg(
            sr=("r", lambda x: x.mean() / x.std() * np.sqrt(12))
        )
    )

    tpf_cf_ss.loc[tpf_cf_ss["cluster"] == "bm", "cluster"] = "none"

    x_values = pd.DataFrame({"sd": np.arange(0, 0.36, 0.01)})

    # Cross-join Sharpe ratio data with standard deviation values
    cf_ef_markowitz = (
        tpf_cf_ss.merge(x_values, how="cross")
        .assign(
            ret=lambda df: df["sd"] * df["sr"],
            shuffled=lambda df: df["cluster"]
            .str.replace("_", " ")
            .str.replace("short term", "short-term")
            .str.title(),
        )
    )

    # Create a categorical column with desired ordering for themes
    cf_ef_markowitz["shuffled"] = pd.Categorical(
        cf_ef_markowitz["shuffled"],
        categories=["None", "Quality", "Value", "Short-Term Reversal", "Momentum"],
        ordered=True,
    )

    # Filter for the selected themes
    sub = ["None", "Quality", "Value", "Short-Term Reversal", "Momentum"]
    cf_ef_markowitz = cf_ef_markowitz[cf_ef_markowitz["shuffled"].isin(sub)]

    # Plot the data
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=cf_ef_markowitz,
        x="sd",
        y="ret",
        hue="shuffled",
        style="shuffled",
        palette=colours_theme,
        linewidth=1.5,
    )

    plt.xlim(0, 0.25)

    # Dynamically set y-axis based on the data
    y_min = cf_ef_markowitz["ret"].min()
    y_max = cf_ef_markowitz["ret"].max()
    y_margin = 0.05 * (y_max - y_min)
    plt.ylim(y_min - y_margin, y_max + y_margin)

    plt.xlabel("Volatility")
    plt.ylabel("Excess returns")
    plt.title("Counterfactual Efficient Frontier Without Trading Costs")

    plt.legend(
        title="Theme permuted:",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=3,
        frameon=True,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
