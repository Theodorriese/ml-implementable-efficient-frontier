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
    df["cluster"] = pd.Categorical(df["cluster"], categories=sort_order.index.tolist(), ordered=True)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

    pfml_plot = df[df["type"] == "Portfolio-ML"]
    axes[0].barh(pfml_plot["cluster"], pfml_plot["fi"], color="#1f77b4")
    axes[0].set_title("Portfolio-ML")
    axes[0].set_xlabel("Drop in realized utility")
    axes[0].invert_yaxis()

    marko_plot = df[df["type"] == "Markowitz-ML"]
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


def calculate_ief_summary(output_path, ef_ss, pf_set, save_path=None):
    """
    Calculates and visualizes summary statistics for the Incremental Efficiency Frontier (IEF)
    under counterfactual (shuffled) conditions across clusters.

    Parameters:
        output_path (str): Directory containing the 'pfml_cf_ief.csv' file.
        ef_ss (pd.DataFrame): Benchmark statistics for the original (non-shuffled) portfolios.
        pf_set (list): List of portfolio identifiers (not used directly in this function).
        save_path (str, optional): File path to save the plot. If None, displays the plot interactively.

    Returns:
        pd.DataFrame: Combined summary statistics of shuffled and benchmark portfolios.
    """

    file_path = os.path.join(output_path, "pfml_cf_ief.csv")
    pfml_cf_ief = pd.read_csv(file_path)

    # Compute per-cluster stats
    ef_cf_ss = (
        pfml_cf_ief
        .groupby("cluster")
        .apply(calc_group_stats)
        .reset_index()
        .rename(columns={"cluster": "shuffled"})
    )

    # Append benchmark data
    benchmark_data = ef_ss[["gamma_rel", "obj", "r_tc_annual", "sd_annual", "sr_net"]].copy()
    benchmark_data["shuffled"] = "none"

    ef_cf_ss = pd.concat([ef_cf_ss, benchmark_data], ignore_index=True)

    # Optional: sort x-axis so 'none' is last
    ef_cf_ss['shuffled'] = pd.Categorical(
        ef_cf_ss['shuffled'],
        categories=sorted(ef_cf_ss['shuffled'].unique(), key=lambda x: (x != 'none', x)),
        ordered=True
    )

    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=ef_cf_ss,
        x="shuffled",
        y="obj",
        hue="gamma_rel",
        dodge=True,
    )
    plt.xlabel("")
    plt.ylabel("Objective Function Value")
    plt.title("IEF Summary Statistics by Theme")
    plt.legend(title="Gamma Rel", loc="upper right")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved to {save_path}")
    else:
        plt.show()

    return ef_cf_ss



# With trading costs
def plot_with_trading_costs(ef_cf_ss, colours_theme, save_path=None):
    """
    Plot excess returns (net of trading cost) against volatility with trading costs.

    Parameters:
        ef_cf_ss (pd.DataFrame): Summary statistics for feature importance in IEF.
        colours_theme (list): List of colors for visualizations.
        save_path (str): Path to save the plot (optional).

    Returns:
        None
    """
    # Define relevant themes
    sub = ["Quality", "Value", "Short-Term Reversal", "None"]

    # Add zero-volatility and zero-return entries for each shuffled category
    zero_entries = pd.DataFrame({
        "shuffled": ef_cf_ss["shuffled"].unique(),
        "sd": 0,
        "r_tc": 0
    })

    # Combine the main data with the zero entries
    combined_data = pd.concat([ef_cf_ss, zero_entries], ignore_index=True)

    # Format and clean the `shuffled` column
    combined_data["shuffled"] = (
        combined_data["shuffled"]
        .str.replace("_", " ")
        .str.replace("short term", "short-term")
        .str.title()
    )

    # Create a categorical column with the desired order
    combined_data["shuffled"] = pd.Categorical(
        combined_data["shuffled"],
        categories=["None", "Quality", "Value", "Short-Term Reversal", "Momentum"],
        ordered=True
    )

    # Filter for the selected themes
    filtered_data = combined_data[combined_data["shuffled"].isin(sub)]

    # Plot the data
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=filtered_data,
        x="sd",
        y="r_tc",
        hue="shuffled",
        style="shuffled",
        size="gamma_rel",
        markers=True,
        palette=colours_theme,
        linewidth=1.5
    )
    sns.scatterplot(
        data=filtered_data,
        x="sd",
        y="r_tc",
        hue="shuffled",
        style="gamma_rel",
        s=50,
        palette=colours_theme,
        legend=False
    )

    plt.xlim(0, 0.25)
    plt.ylim(0, 0.28)
    plt.xlabel("Volatility")
    plt.ylabel("Excess returns (net of trading cost)")

    plt.legend(
        title="Theme permuted:",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=3,
        frameon=True
    )

    plt.title("Excess Returns vs Volatility with Trading Costs")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)  # Save the plot
    else:
        plt.show()  # Otherwise, show the plot


# Counterfactual EF without TC --------------------------
def plot_counterfactual_ef_without_tc(output_path, colours_theme, save_path=None):
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


# Feature importance for return predictions models --------------------------------
def plot_feature_importance_for_return_predictions(output_path, colours_theme, save_path=None):
    """
    Plot feature importance for return prediction models.

    Parameters:
        output_path (str): Path where the `ret_cf.csv` is stored.
        colours_theme (list): List of colors for visualizations.
        save_path (str): Path to save the plot (optional).

    Returns:
        None
    """
    ret_cf_path = os.path.join(output_path, "ret_cf.csv")
    ret_cf = pd.read_csv(ret_cf_path)

    # Calculate mean MSE per horizon and cluster
    ret_cf_ss = (
        ret_cf.groupby(["h", "cluster"], as_index=False)
        .agg(mse=("mse", "mean"))
    )

    # Extract benchmark MSE and compute feature importance
    bm = ret_cf_ss[ret_cf_ss["cluster"] == "bm"][["h", "mse"]].rename(columns={"mse": "bm"})
    ret_cf_ss = ret_cf_ss[ret_cf_ss["cluster"] != "bm"].merge(bm, on="h", how="left")
    ret_cf_ss["fi"] = ret_cf_ss["mse"] - ret_cf_ss["bm"]

    # Format cluster names
    ret_cf_ss["cluster"] = (
        ret_cf_ss["cluster"]
        .str.replace("_", " ")
        .str.replace("short term", "short-term")
        .str.title()
    )

    # Normalize feature importance within each horizon
    ret_cf_ss["fi"] = ret_cf_ss.groupby("h")["fi"].transform(lambda x: x / x.max())

    # Get sorting variable based on horizon == 1
    sort_map = (
        ret_cf_ss[ret_cf_ss["h"] == 1]
        .set_index("cluster")["fi"]
        .to_dict()
    )
    ret_cf_ss["sort_var"] = ret_cf_ss["cluster"].map(sort_map)

    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=ret_cf_ss,
        x="cluster",
        y="fi",
        hue="h",
        palette=colours_theme,
        dodge=True
    )
    plt.axhline(y=1, linestyle="dashed", color="black")
    plt.xlabel("")
    plt.ylabel("Drop in MSE from permuting theme features (% of max)")
    plt.title("Feature Importance for Return Prediction Models")
    plt.xticks(rotation=45, ha="right")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.legend(title="Horizon", loc="best")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()



# Seasonality effect
def analyze_seasonality_effect(chars, save_path=None):
    """
    Analyze seasonality effects by correlating predictors with returns.

    Parameters:
        chars (pd.DataFrame): DataFrame containing 'id', 'eom', predictors ('pred_ld1' to 'pred_ld12'),
                              and other return metrics ('ret_1_0', 'ret_12_1', etc.).
        save_path (str): Path to save the plot (optional).

    Returns:
        None
    """
    # Filter relevant columns and reshape to long format
    predictors = [f"pred_ld{i}" for i in range(1, 13)]
    relevant_columns = ["id", "eom"] + predictors + ["ret_1_0", "ret_12_1", "be_me", "gp_at"]
    data = chars[relevant_columns].dropna(subset=["pred_ld1"])

    melted = data.melt(id_vars=["id", "eom"] + predictors, var_name="variable", value_name="value")

    # Calculate correlations
    correlations = (
        melted.groupby(["variable", "eom"])
        .apply(lambda group: {col: group["value"].corr(group[col]) for col in predictors})
        .apply(pd.Series)
        .reset_index()
    )

    # Average correlations across periods
    correlations = (
        correlations.groupby("variable", as_index=False)
        .mean()
        .drop(columns=["eom"])
        .melt(id_vars=["variable"], var_name="h", value_name="cor")
    )

    # Extract horizon (h) as a numeric value
    correlations["h"] = correlations["h"].str.replace("pred_ld", "").astype(int)

    # Plot correlations
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=correlations,
        x="h",
        y="cor",
        hue="variable",
        marker="o"
    )
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.xlabel("Horizon (Months)")
    plt.ylabel("Correlation")
    plt.title("Correlations of Predictors with Returns Across Horizons")
    plt.legend(title="Variable", loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=3, frameon=True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)  # Save the plot
    else:
        plt.show()  # Otherwise, show the plot

