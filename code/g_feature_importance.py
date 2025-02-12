import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


# FI in base case -----------------------------------------------
def calculate_feature_importance(pf_set, colours_theme):
    """
    Calculate and visualize feature importance based on drop in realized utility.

    Parameters:
        pf_set (dict): Portfolio settings, including gamma_rel.
        colours_theme (list): Theme colors for visualization.

    Returns:
        None
    """
    fi_path = "Data/Generated/Portfolios/FI/"
    fi_folder = os.path.join(fi_path, sorted(os.listdir(fi_path))[0])  # First folder in the FI directory

    tpf_cf_base = pd.read_csv(os.path.join(fi_folder, "tpf_cf_base.csv"))
    pfml_cf_base = pd.read_csv(os.path.join(fi_folder, "pfml_cf_base.csv"))

    # Calculate feature importance for Portfolio-ML
    pfml_cf_ss = (
        pfml_cf_base
        .groupby(["type", "cluster"], as_index=False)
        .agg(
            cf_obj=lambda df: (df["r"].mean() - 0.5 * df["r"].var() * pf_set["gamma_rel"] - df["tc"].mean()) * 12
        )
    )
    pfml_cf_ss["fi"] = (
        pfml_cf_ss.loc[pfml_cf_ss["cluster"] == "bm", "cf_obj"].values[0] - pfml_cf_ss["cf_obj"]
    )
    pfml_cf_ss["wealth"] = 1e10
    pfml_cf_ss = pfml_cf_ss[["type", "cluster", "fi", "wealth"]]

    # Calculate feature importance for Tangency Portfolio
    tpf_cf_ss = (
        tpf_cf_base
        .groupby(["type", "cluster"], as_index=False)
        .agg(
            cf_obj=lambda df: (df["r"].mean() - 0.5 * df["r"].var() * pf_set["gamma_rel"] - df["tc"].mean()) * 12
        )
    )
    tpf_cf_ss["fi"] = (
        tpf_cf_ss.loc[tpf_cf_ss["cluster"] == "bm", "cf_obj"].values[0] - tpf_cf_ss["cf_obj"]
    )
    tpf_cf_ss["wealth"] = 0
    tpf_cf_ss = tpf_cf_ss[["type", "cluster", "fi", "wealth"]]

    # Combine Portfolio-ML and Tangency Portfolio data
    feature_importance = pd.concat([pfml_cf_ss, tpf_cf_ss], ignore_index=True)

    # Process feature importance
    feature_importance = (
        feature_importance
        .loc[feature_importance["cluster"] != "bm"]
        .assign(
            sort_var=lambda df: df.groupby("cluster")["fi"].transform(
                lambda x: x.sum() if df["type"].eq("Portfolio-ML").any() else 0
            ),
            cluster=lambda df: df["cluster"]
            .str.replace("_", " ")
            .str.replace("short term", "short-term")
            .str.title(),
            type=lambda df: pd.Categorical(
                df["type"],
                categories=["Portfolio-ML", "Multiperiod-ML*", "Markowitz-ML", "Expected 1m Return"],
                ordered=True
            )
        )
    )
    feature_importance = feature_importance.loc[feature_importance["type"] != "Expected 1m Return"]

    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=feature_importance,
        x="cluster",
        y="fi",
        hue="type",
        dodge=True,
        palette=[colours_theme[0], colours_theme[1], colours_theme[4]]
    )
    plt.xlabel("")
    plt.ylabel("Drop in realized utility from permuting theme features")
    plt.title("Feature Importance by Theme")
    plt.legend(title="Portfolio Type", loc="upper right")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# FI in IEF -------------------------
def calculate_ief_summary(output_path, ef_ss, pf_set):
    """
    Calculate summary statistics for feature importance in IEF (Incremental Efficiency Frontier).

    Parameters:
        output_path (str): Path to the directory containing the pfml_cf_ief.csv file.
        ef_ss (pd.DataFrame): Summary statistics dataframe.
        pf_set (dict): Portfolio settings, including gamma_rel and wealth.

    Returns:
        pd.DataFrame: Summary statistics for IEF, including benchmarks.
    """
    # Define the file path for pfml_cf_ief.csv
    file_path = os.path.join(output_path, "pfml_cf_ief.csv")

    # Read the data
    pfml_cf_ief = pd.read_csv(file_path)

    # Calculate summary statistics for IEF clusters
    ef_cf_ss = (
        pfml_cf_ief
        .groupby(["gamma_rel", "cluster"], as_index=False)
        .agg(
            obj=lambda df: (df["r"].mean() - 0.5 * df["r"].var() * df["gamma_rel"].iloc[0] - df["tc"].mean()) * 12,
            r_tc=lambda df: (df["r"] - df["tc"]).mean() * 12,
            sd=lambda df: df["r"].std() * np.sqrt(12)
        )
    )
    ef_cf_ss["sr"] = ef_cf_ss["r_tc"] / ef_cf_ss["sd"]  # Calculate Sharpe Ratio
    ef_cf_ss.rename(columns={"cluster": "shuffled"}, inplace=True)

    # Filter and add benchmark data from ef_ss
    benchmark_data = ef_ss.loc[
        ef_ss["wealth_end"] == pf_set["wealth"],
        ["gamma_rel", "obj", "r_tc", "sd", "sr"]
    ].copy()
    benchmark_data["shuffled"] = "none"

    # Combine IEF statistics with benchmark
    ef_cf_ss = pd.concat([ef_cf_ss, benchmark_data], ignore_index=True)

    return ef_cf_ss


# With trading costs
def plot_with_trading_costs(ef_cf_ss, colours_theme):
    """
    Plot excess returns (net of trading cost) against volatility with trading costs.

    Parameters:
        ef_cf_ss (pd.DataFrame): Summary statistics for feature importance in IEF.
        colours_theme (list): List of colors for visualizations.

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
    plt.show()


# Counterfactual EF without TC --------------------------
def plot_counterfactual_ef_without_tc(output_path, colours_theme):
    """
    Plot counterfactual efficient frontier (EF) without trading costs.

    Parameters:
        output_path (str): Path where the tpf_cf_base.csv is stored.
        colours_theme (list): List of colors for visualizations.

    Returns:
        None
    """
    tpf_cf_base_path = os.path.join(output_path, "tpf_cf_base.csv")
    tpf_cf_base = pd.read_csv(tpf_cf_base_path)

    # Calculate Sharpe ratio for each cluster
    tpf_cf_ss = (
        tpf_cf_base.groupby("cluster", as_index=False)
        .agg(sr=lambda df: df["r"].mean() / df["r"].std() * np.sqrt(12))
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
    plt.ylim(0, 0.55)
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
    plt.show()

# Feature importance for return predictions models --------------------------------
def plot_feature_importance_for_return_predictions(output_path, colours_theme):
    """
    Plot feature importance for return prediction models.

    Parameters:
        output_path (str): Path where the `ret_cf.csv` is stored.
        colours_theme (list): List of colors for visualizations.

    Returns:
        None
    """
    ret_cf_path = os.path.join(output_path, "ret_cf.csv")

    ret_cf = pd.read_csv(ret_cf_path)

    # Calculate MSE for each horizon and cluster
    ret_cf_ss = (
        ret_cf.groupby(["h", "cluster"], as_index=False)
        .agg(mse=("mse", "mean"))
    )

    # Extract benchmark MSE and calculate feature importance
    bm = ret_cf_ss.loc[ret_cf_ss["cluster"] == "bm", ["h", "mse"]].rename(columns={"mse": "bm"})
    ret_cf_ss = ret_cf_ss.loc[ret_cf_ss["cluster"] != "bm"]
    ret_cf_ss = ret_cf_ss.merge(bm, on="h", how="left")
    ret_cf_ss["fi"] = ret_cf_ss["mse"] - ret_cf_ss["bm"]

    # Modify cluster names
    ret_cf_ss["cluster"] = (
        ret_cf_ss["cluster"]
        .str.replace("_", " ")
        .str.replace("short term", "short-term")
        .str.title()
    )

    # Normalize feature importance by horizon and create sorting variable
    ret_cf_ss["fi"] = ret_cf_ss.groupby("h")["fi"].transform(lambda x: x / x.max())
    ret_cf_ss["sort_var"] = ret_cf_ss.groupby("cluster")["fi"].transform(lambda x: x[ret_cf_ss["h"] == 1].iloc[0])

    # Plot feature importance by cluster and horizon
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
    plt.legend(title="Horizon", loc="best")
    plt.tight_layout()
    plt.show()

    # Alternative view: Drop in MSE by cluster and horizon with facets
    ret_cf_ss["title"] = pd.Categorical(
        "Horizon: " + ret_cf_ss["h"].astype(str) + " month",
        categories=["Horizon: " + str(i) + " month" for i in range(1, 13)],
        ordered=True
    )

    plt.figure(figsize=(14, 8))
    g = sns.FacetGrid(
        ret_cf_ss,
        col="title",
        col_wrap=4,
        sharex=False,
        sharey=False,
        height=4,
        aspect=1.2
    )
    g.map_dataframe(
        sns.barplot,
        x="cluster",
        y="fi",
        color=colours_theme[0]
    )
    g.set_axis_labels("", "Drop in MSE from permuting theme features (% of max)")
    g.set_titles("{col_name}")
    g.set_xticklabels(rotation=45, horizontalalignment="right")
    plt.tight_layout()
    plt.show()


# Wierdly, ret_1_0 strongly predicts t+12? But note that it's with the opposite sign of t+1 (I know! It's the seasonality effect)
def analyze_seasonality_effect(chars):
    """
    Analyze seasonality effects by correlating predictors with returns.

    Parameters:
        chars (pd.DataFrame): DataFrame containing 'id', 'eom', predictors ('pred_ld1' to 'pred_ld12'),
                              and other return metrics ('ret_1_0', 'ret_12_1', etc.).

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
    plt.show()
