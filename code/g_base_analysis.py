import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
from scipy.stats import norm
from datetime import datetime
from tqdm import tqdm
import math


# Need to run everything from main to right before bm_pfs in 5 - base case
# Load base case portfolios -----------
def load_base_case_portfolios(base_path):
    """
    Load base case portfolio data from the specified directory.

    Parameters:
        base_path (str): Path to the base portfolio directory.

    Returns:
        dict: Dictionary containing dataframes for 'mp', 'pfml', 'static', and 'bm_pfs'.
    """
    # Locate the base folder
    base_folder = next(os.walk(base_path))[1][0]

    # Load required datasets
    mp = pd.read_pickle(os.path.join(base_path, base_folder, "multiperiod-ml.pkl"))
    pfml = pd.read_pickle(os.path.join(base_path, base_folder, "portfolio-ml.pkl"))
    static = pd.read_pickle(os.path.join(base_path, base_folder, "static-ml.pkl"))
    bm_pfs = pd.read_csv(os.path.join(base_path, base_folder, "bms.csv"))

    # Update type naming convention for bm_pfs
    bm_pfs["eom_ret"] = pd.to_datetime(bm_pfs["eom_ret"])
    bm_pfs["type"] = bm_pfs["type"].replace("Rank-Weighted", "Rank-ML")

    return {"mp": mp, "pfml": pfml, "static": static, "bm_pfs": bm_pfs}


# Final Portfolios ---------------------
def combine_portfolios(mp, pfml, static, bm_pfs, pf_order, gamma_rel):
    """
    Combine multiple portfolios into a single dataframe.

    Parameters:
        mp (pd.DataFrame): Multiperiod-ML portfolio data.
        pfml (pd.DataFrame): Portfolio-ML data.
        static (pd.DataFrame): Static-ML data.
        bm_pfs (pd.DataFrame): Benchmark portfolios data.
        pf_order (list): Order of portfolio types for factor conversion.
        gamma_rel (float): Gamma value for risk aversion.

    Returns:
        pd.DataFrame: Combined portfolio dataframe with utility adjustments.
    """
    # Combine portfolios
    pfs = pd.concat([
        mp["pf"],
        pfml["pf"],
        static["pf"],
        bm_pfs,
        mp["hps"].loc[
            (mp["hps"]["eom_ret"].isin(mp["pf"]["eom_ret"])) &
            (mp["hps"]["k"] == 1) &
            (mp["hps"]["g"] == 0) &
            (mp["hps"]["u"] == 1),
            ["eom_ret", "inv", "shorting", "turnover", "r", "tc"]
        ].assign(type="Multiperiod-ML"),
        static["hps"].loc[
            (static["hps"]["eom_ret"].isin(static["pf"]["eom_ret"])) &
            (static["hps"]["k"] == 1) &
            (static["hps"]["g"] == 0) &
            (static["hps"]["u"] == 1),
            ["eom_ret", "inv", "shorting", "turnover", "r", "tc"]
        ].assign(type="Static-ML")
    ])

    # Convert 'type' to categorical with specified order
    pfs["type"] = pd.Categorical(pfs["type"], categories=pf_order, ordered=True)

    # Sort the dataframe
    pfs.sort_values(by=["type", "eom_ret"], inplace=True)

    # Compute utility and adjusted variables
    pfs["e_var_adj"] = pfs.groupby("type")["r"].transform(lambda x: (x - x.mean()) ** 2)
    pfs["utility_t"] = pfs["r"] - pfs["tc"] - 0.5 * pfs["e_var_adj"] * gamma_rel

    return pfs


# Portfolio summary stats --------------
def compute_portfolio_summary(pfs, main_types, gamma_rel):
    """
    Compute portfolio summary statistics.

    Parameters:
        pfs (pd.DataFrame): Combined portfolio dataframe.
        main_types (list): List of essential portfolio types to include.
        gamma_rel (float): Gamma value for risk aversion.

    Returns:
        pd.DataFrame: Summary statistics for each portfolio type.
        pd.DataFrame: Filtered portfolios containing only main types.
    """
    # Calculate summary statistics for each type
    pf_summary = (
        pfs.groupby("type")
        .agg(
            n=("type", "size"),
            inv=("inv", "mean"),
            shorting=("shorting", "mean"),
            turnover_notional=("turnover", "mean"),
            r=("r", lambda x: x.mean() * 12),
            sd=("r", lambda x: x.std() * np.sqrt(12)),
            sr_gross=("r", lambda x: x.mean() / x.std() * np.sqrt(12)),
            tc=("tc", lambda x: x.mean() * 12),
            r_tc=("r", lambda x: (x - pfs.loc[x.index, "tc"]).mean() * 12),
            sr=("r", lambda x: ((x - pfs.loc[x.index, "tc"]).mean()) / x.std() * np.sqrt(12)),
            obj=("r", lambda x: (x.mean() - 0.5 * x.var() * gamma_rel - pfs.loc[x.index, "tc"].mean()) * 12),
        )
        .reset_index()
    )

    # Filter the portfolio to include only main types
    pfs = pfs[pfs["type"].isin(main_types)]
    pfs["type"] = pd.Categorical(pfs["type"], categories=main_types, ordered=True)

    # Sort the dataframe
    pfs = pfs.sort_values(by=["type", "eom_ret"])

    return pf_summary, pfs


# Performance Time-Series -----------
def compute_and_plot_performance_time_series(pfs, main_types, start_date, end_date):
    """
    Computes and plots the performance time-series for different portfolio types.

    Parameters:
        pfs (pd.DataFrame): DataFrame containing portfolio performance metrics.
        main_types (list): List of main portfolio types to include in the plots.
        start_date (str): Start date of the time-series plots (e.g., '1980-12-31').
        end_date (str): End date of the time-series plots (e.g., '2020-12-31').

    Returns:
        matplotlib.figure.Figure: Figure containing the cumulative performance plots.
    """
    # Filter and compute cumulative returns
    pfs['cumret'] = pfs.groupby('type')['r'].cumsum()
    pfs['cumret_tc'] = pfs.groupby('type').apply(lambda x: x['r'] - x['tc']).groupby(level=0).cumsum()
    pfs['cumret_tc_risk'] = pfs.groupby('type').apply(lambda x: x['utility_t']).groupby(level=0).cumsum()

    # Filter for main types and reshape data
    ts_data = (
        pfs[pfs['type'].isin(main_types)]
        .melt(id_vars=['type', 'eom_ret'], value_vars=['cumret', 'cumret_tc', 'cumret_tc_risk'], var_name='metric')
    )

    # Add initial zero values for each type
    initial_values = pd.DataFrame({
        'eom_ret': pd.Timestamp(start_date) - pd.offsets.MonthBegin(),
        'type': main_types * 3,
        'value': 0,
        'metric': ['cumret'] * len(main_types) + ['cumret_tc'] * len(main_types) + ['cumret_tc_risk'] * len(main_types)
    })
    ts_data = pd.concat([ts_data, initial_values], ignore_index=True)

    # Create pretty labels for the metrics
    metric_labels = {
        'cumret': 'Gross return',
        'cumret_tc': 'Return net of TC',
        'cumret_tc_risk': 'Return net of TC and Risk'
    }
    ts_data['metric_pretty'] = ts_data['metric'].map(metric_labels)
    ts_data['metric_pretty'] = pd.Categorical(ts_data['metric_pretty'], categories=metric_labels.values())

    # Plot settings
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False, gridspec_kw={'width_ratios': [1.07, 1, 1]})

    for ax, metric in zip(axes, ts_data['metric_pretty'].cat.categories):
        plot_data = ts_data[ts_data['metric_pretty'] == metric]
        for portfolio_type in main_types:
            portfolio_data = plot_data[plot_data['type'] == portfolio_type]
            ax.plot(portfolio_data['eom_ret'], portfolio_data['value'], label=portfolio_type)
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_ylabel("Cumulative Performance")
        ax.xaxis.set_major_formatter(DateFormatter('%Y'))
        ax.set_xlim(pd.Timestamp(start_date), pd.Timestamp(end_date))
        if metric == 'Gross return':
            max_val = plot_data[plot_data['type'] != "Markowitz-ML"]['value'].max()
            min_val = plot_data[plot_data['type'] != "Markowitz-ML"]['value'].min()
            ax.set_ylim(min_val, max_val)
        ax.legend(loc="upper left", fontsize='small', title="Method:")

    # Adjust layout
    fig.tight_layout()
    return fig


# Test probability of outperformance ---------------------------------------------------
def compute_probability_of_outperformance(pfs, main_types):
    """
    Compute the probability of outperformance for each main portfolio type.

    Parameters:
        pfs (pd.DataFrame): Dataframe containing portfolio data, including 'type', 'eom_ret', and 'utility_t'.
        main_types (list): List of main portfolio types to analyze.

    Returns:
        pd.DataFrame: Dataframe with probabilities of outperformance for each portfolio type.
    """
    # Reshape the portfolio data into wide format
    pfs_wide = pfs.pivot(index="eom_ret", columns="type", values="utility_t")

    # Calculate probability of outperformance for each main type
    prob_outperformance_results = []

    for main_type in main_types:
        # Melt data for easier comparison
        melted = (
            pfs_wide.reset_index()
            .melt(id_vars=["eom_ret", main_type], var_name="alt", value_name="utility_t_alt")
        )

        melted["diff"] = melted[main_type] - melted["utility_t_alt"]

        # Calculate probabilities of outperformance
        prob_outperformance = (
            melted.groupby("alt")["diff"]
            .apply(lambda diff: 1 - norm.cdf(0, loc=diff.mean(), scale=diff.std() / (len(diff) ** 0.5)))
            .reset_index(name="prob_main_op")
        )
        prob_outperformance["main"] = main_type

        prob_outperformance_results.append(prob_outperformance)

    # Combine results for all main types
    prob_outperformance_df = pd.concat(prob_outperformance_results, ignore_index=True)

    return prob_outperformance_df


# Portfolio statistics over time --------------------
# Compute expected portfolio risk
def compute_expected_risk(dates, cov_list, weights):
    """
    Compute expected portfolio risks based on weights and covariance matrices.

    Parameters:
        dates (list): Out-of-sample dates.
        cov_list (dict): Covariance matrices.
        weights (pd.DataFrame): Portfolio weights.

    Returns:
        pd.DataFrame: DataFrame containing expected risks for each portfolio and date.
    """
    risk_records = []
    for date in dates:
        if date in cov_list:
            cov_matrix = cov_list[date]
            weights_at_date = weights[weights["eom"] == date]
            for portfolio_type in weights_at_date["type"].unique():
                w = weights_at_date[weights_at_date["type"] == portfolio_type]["weights"].values
                risk = np.dot(w.T, np.dot(cov_matrix, w))  # Portfolio variance
                risk_records.append({"type": portfolio_type, "eom": date, "pf_var": risk})

    return pd.DataFrame(risk_records)


# Compute portfolios statistics
def compute_and_plot_portfolio_statistics_over_time(pfml, tpf, mp, static, factor_ml, pfs, barra_cov, dates_oos,
                                                    pf_order, main_types):
    """
    Compute and plot portfolio statistics over time, including ex-ante volatility, turnover, and leverage.

    Parameters:
        pfml (pd.DataFrame): Portfolio-ML weights.
        tpf (pd.DataFrame): Markowitz-ML weights.
        mp (pd.DataFrame): Multiperiod-ML* weights.
        static (pd.DataFrame): Static-ML* weights.
        factor_ml (pd.DataFrame): Factor-ML weights.
        pfs (pd.DataFrame): Portfolio statistics and metadata.
        barra_cov (dict): Covariance matrices for expected risk calculation.
        dates_oos (list): Out-of-sample dates.
        pf_order (list): Ordered list of portfolio types.
        main_types (list): List of main portfolio types to include in the analysis.

    Returns:
        None
    """
    # Combine portfolio weights
    pfml_weights = pfml["w"].copy()
    pfml_weights["type"] = "Portfolio-ML"

    tpf_weights = tpf["w"].copy()
    tpf_weights["type"] = "Markowitz-ML"

    mp_weights = mp["w"].copy()
    mp_weights["type"] = "Multiperiod-ML*"

    static_weights = static["w"].copy()
    static_weights["type"] = "Static-ML*"
    static_weights.drop(columns=["pred_ld1"], inplace=True)

    factor_ml_weights = factor_ml["w"].copy()
    factor_ml_weights["type"] = "Factor-ML"

    # Combine all weights
    combined_weights = pd.concat(
        [pfml_weights, tpf_weights, mp_weights, static_weights, factor_ml_weights],
        ignore_index=True
    )
    combined_weights["type"] = pd.Categorical(combined_weights["type"], categories=pf_order, ordered=True)

    # Compute expected portfolio risks
    pf_vars = compute_expected_risk(dates_oos, barra_cov, combined_weights)

    # Merge risk and portfolio statistics
    pfs = pfs.copy()
    pfs["eom"] = pfs["eom_ret"] + pd.DateOffset(months=-1) - pd.DateOffset(days=1)
    merged_stats = pf_vars.merge(
        pfs[["type", "eom", "inv", "turnover"]],
        on=["type", "eom"],
        how="inner"
    )
    merged_stats = merged_stats[merged_stats["type"].isin(main_types)]

    # Compute additional statistics
    merged_stats["e_sd"] = np.sqrt(merged_stats["pf_var"] * 252)
    merged_stats_long = merged_stats.drop(columns=["pf_var"]).melt(
        id_vars=["type", "eom"], var_name="stat", value_name="value"
    )

    # Map statistic names to display-friendly labels
    stat_name_map = {
        "e_sd": "Ex-ante Volatility",
        "turnover": "Turnover",
        "inv": "Leverage"
    }
    merged_stats_long["stat"] = merged_stats_long["stat"].map(stat_name_map)

    # Plot portfolio statistics over time
    sns.set_theme(style="whitegrid")
    g = sns.FacetGrid(
        merged_stats_long, col="stat", sharey=False, height=4, aspect=1.5,
        col_wrap=1, margin_titles=True
    )
    g.map_dataframe(
        sns.lineplot, x="eom", y="value", hue="type", style="type", palette="deep"
    )
    g.set_axis_labels("Date", "Value")
    g.set_titles(col_template="{col_name}")
    g.set(yscale="log")  # Log scale for better visualization
    g.add_legend(title="Portfolio Type", bbox_to_anchor=(0.5, -0.1), loc="lower center", ncol=2)
    g.tight_layout()

    # Show the plot
    plt.show()


# Correlation ----------------------------------
def compute_and_plot_correlation_matrix(pfs, main_types):
    """
    Compute and visualize the correlation matrix of portfolio returns.

    Parameters:
        pfs (pd.DataFrame): DataFrame containing portfolio returns (`r`) and metadata.
        main_types (list): List of main portfolio types to include in the correlation analysis.

    Returns:
        pd.DataFrame: Correlation matrix of portfolio returns.
    """
    # Pivot portfolio returns to a wide format with portfolio types as columns
    wide_returns = pfs.pivot(index="eom_ret", columns="type", values="r")

    wide_returns = wide_returns[main_types]
    correlation_matrix = wide_returns.corr(method="pearson")

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        annot=True,  # Display the correlation values
        fmt=".2f",  # Format the numbers with 2 decimals
        cmap="coolwarm",  # Color map
        cbar=True,  # Show the color bar
        square=True,  # Keep cells square-shaped
        linewidths=0.5,  # Line width between cells
        annot_kws={"size": 10},  # Annotation font size
        vmin=-1, vmax=1  # Set color scale limits
    )
    plt.title("Portfolio Return Correlation Matrix", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.show()

    return correlation_matrix


# Apple vs. Xerox ----------------------------------------------------
def plot_apple_vs_xerox(mp, pfml, static, tpf, factor_ml, mkt, pfs, liquid_id, illiquid_id, start_year=2015):
    """
    Generates a plot comparing portfolio weights for Apple and Xerox stocks across portfolios.

    Parameters:
        mp (pd.DataFrame): Multiperiod-ML portfolio dataframe.
        pfml (pd.DataFrame): Portfolio-ML portfolio dataframe.
        static (pd.DataFrame): Static-ML portfolio dataframe.
        tpf (pd.DataFrame): Tangency portfolio dataframe.
        factor_ml (pd.DataFrame): Factor-ML portfolio dataframe.
        mkt (pd.DataFrame): Market portfolio dataframe.
        pfs (pd.DataFrame): Combined portfolio dataframe for additional data.
        liquid_id (int): ID of the liquid stock (e.g., Johnson and Johnson).
        illiquid_id (int): ID of the illiquid stock (e.g., Xerox).
        start_year (int): Start year for filtering data.
    """
    # Filter positions for liquid and illiquid stocks from each portfolio
    position_frames = [
        mp[(mp["id"] == liquid_id) | (mp["id"] == illiquid_id)].assign(type="Multiperiod-ML*"),
        pfml[(pfml["id"] == liquid_id) | (pfml["id"] == illiquid_id)].assign(type="Portfolio-ML"),
        static[(static["id"] == liquid_id) | (static["id"] == illiquid_id)].assign(type="Static-ML*"),
        tpf[(tpf["id"] == liquid_id) | (tpf["id"] == illiquid_id)].assign(type="Markowitz-ML"),
        factor_ml[(factor_ml["id"] == liquid_id) | (factor_ml["id"] == illiquid_id)].assign(type="Factor-ML"),
        mkt[(mkt["id"] == liquid_id) | (mkt["id"] == illiquid_id)].assign(type="Market"),
    ]

    # Combine all positions into a single DataFrame
    positions = pd.concat(position_frames, ignore_index=True)

    # Map stock IDs to readable names
    stock_type_map = {
        14593: "Apple (liquid)",
        27983: "Xerox (illiquid)",
        93436: "Tesla",
        91103: "Visa",
        19561: "Boeing",
        10107: "Microsoft",
        22111: "Johnson and Johnson (liquid)",
        55976: "Walmart (liquid)",
    }

    # Add stock type labels
    positions["stock_type"] = positions["id"].map(stock_type_map).fillna(positions["id"].astype(str))

    # Filter data by year
    positions["eom"] = pd.to_datetime(positions["eom"])
    positions = positions[positions["eom"].dt.year >= start_year]

    # Add weights normalization
    pfs["eom"] = pd.to_datetime(pfs["eom_ret"]).dt.to_period("M").dt.to_timestamp() - pd.DateOffset(days=1)
    positions = pd.merge(
        positions,
        pfs[["type", "eom", "inv"]],
        on=["type", "eom"],
        how="left",
    )
    positions["w_z"] = positions.groupby(["type", "id"])["w"].transform(lambda x: (x - x.mean()) / x.std())

    # Plot portfolio weights over time
    plt.figure(figsize=(10, 6))
    for portfolio_type in positions["type"].unique():
        subset = positions[positions["type"] == portfolio_type]
        for stock_type in subset["stock_type"].unique():
            stock_data = subset[subset["stock_type"] == stock_type]
            plt.plot(
                stock_data["eom"],
                stock_data["w"],
                label=f"{portfolio_type} - {stock_type}",
                alpha=0.8,
            )
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.title("Portfolio Weights Over Time: Liquid vs. Illiquid Stocks")
    plt.xlabel("End of Month")
    plt.ylabel("Portfolio Weight")
    plt.legend(loc="upper right", bbox_to_anchor=(1.4, 1), fontsize=8)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


# Optimal Hyper-parameters ----------------------------
def load_model_hyperparameters(model_path, horizon_label):
    """
    Load and process hyperparameters for a given model.

    Parameters:
        model_path (str): Path to the model file.
        horizon_label (str): Label indicating the prediction horizon.

    Returns:
        pd.DataFrame: Processed hyperparameter data.
    """
    model = pd.read_pickle(model_path)
    processed_data = []

    for entry in model:
        pred = entry["pred"]
        opt_hps = entry["opt_hps"]
        data = (
            pd.concat(
                [pred.assign(eom_ret=pred["eom"] + pd.DateOffset(months=1) - pd.DateOffset(days=1)),
                 opt_hps[["lambda", "p", "g"]]],
                axis=1
            )
            .drop_duplicates()
            .dropna(subset=["eom_ret"])
        )
        data["horizon"] = horizon_label
        processed_data.append(data)

    return pd.concat(processed_data, ignore_index=True)


def process_portfolio_tuning_data(mp, static, pfml, start_year):
    """
    Process portfolio tuning data for Multiperiod-ML, Static-ML, and Portfolio-ML.

    Parameters:
        mp (pd.DataFrame): DataFrame containing tuning results for Multiperiod-ML.
        static (pd.DataFrame): DataFrame containing tuning results for Static-ML.
        pfml (pd.DataFrame): DataFrame containing tuning results for Portfolio-ML.
        start_year (int): Start year for filtering tuning results.

    Returns:
        pd.DataFrame: Processed tuning data.
    """
    # Filter Multiperiod-ML data
    mp_hps = mp[
        (mp["rank"] == 1) &
        (mp["eom_ret"].dt.year >= start_year) &
        (mp["eom_ret"].dt.month == 12)
    ][["eom_ret", "k", "g", "u"]].copy()
    mp_hps["type"] = "Multiperiod-ML*"

    # Filter Static-ML data
    static_hps = static[
        (static["rank"] == 1) &
        (static["eom_ret"].dt.year >= start_year) &
        (static["eom_ret"].dt.month == 12)
    ][["eom_ret", "k", "g", "u"]].copy()
    static_hps["type"] = "Static-ML*"

    # Filter Portfolio-ML data and process lambda
    pfml_hps = pfml[
        pfml["eom_ret"].dt.year >= start_year
    ][["eom_ret", "l", "p", "g"]].copy()
    pfml_hps["type"] = "Portfolio-ML"
    pfml_hps["log(lambda)"] = np.log(pfml_hps["l"])
    pfml_hps = pfml_hps.drop(columns="l").rename(columns={"g": "eta"})

    # Combine all hyperparameter data
    combined_hps = pd.concat(
        [
            mp_hps.melt(id_vars=["type", "eom_ret"]),
            static_hps.melt(id_vars=["type", "eom_ret"]),
            pfml_hps.melt(id_vars=["type", "eom_ret"]),
        ],
        ignore_index=True
    )

    # Create a descriptive column for combined hyperparameter names
    combined_hps["comb_name"] = combined_hps["type"] + ": " + combined_hps["name"]

    return combined_hps


def plot_hyperparameter_trends(data, colours_theme):
    """
    Plot the trends for hyperparameters over time.

    Parameters:
        data (pd.DataFrame): Data containing hyperparameter trends.
        colours_theme (list): List of colors for the plot.

    Returns:
        None
    """
    plt.figure(figsize=(10, 8))
    for comb_name, group in data.groupby("comb_name"):
        plt.scatter(group["eom_ret"], group["value"], alpha=0.75, color=colours_theme[0], label=comb_name)
    plt.xlabel("End of Month")
    plt.ylabel("Optimal Hyper-Parameter")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Hyper-Parameter")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.title("Optimal Hyper-Parameters Over Time")
    plt.tight_layout()
    plt.show()


def plot_portfolio_tuning_results(data):
    """
    Plot portfolio tuning results over time.

    Parameters:
        data (pd.DataFrame): Data containing portfolio tuning results.

    Returns:
        None
    """
    plt.figure(figsize=(10, 8))
    for comb_name, group in data.groupby("comb_name"):
        plt.scatter(group["eom_ret"], group["value"], alpha=0.75, label=comb_name)
    plt.xlabel("End of Month")
    plt.ylabel("Optimal Hyper-Parameter")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Tuning Type")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.title("Portfolio Tuning Results Over Time")
    plt.tight_layout()
    plt.show()


def plot_optimal_hyperparameters(get_from_path_model, mp, static, pfml, colours_theme, start_year=1981):
    """
    Main function to plot optimal hyperparameters and portfolio tuning results.

    Parameters:
        get_from_path_model (str): Path to the model files.
        mp, static, pfml (pd.DataFrame): DataFrames containing tuning results for Multiperiod-ML, Static-ML, and Portfolio-ML.
        colours_theme (list): List of colors for the plot.
        start_year (int): Start year for filtering tuning results.

    Returns:
        None
    """
    # Load and process hyperparameters for each horizon
    data_1 = load_model_hyperparameters(f"{get_from_path_model}/model_1.pkl", "Return t+1")
    data_6 = load_model_hyperparameters(f"{get_from_path_model}/model_6.pkl", "Return t+6")
    data_12 = load_model_hyperparameters(f"{get_from_path_model}/model_12.pkl", "Return t+12")

    # Combine hyperparameter data
    combined_data = pd.concat([data_1, data_6, data_12], ignore_index=True)
    combined_data["log(lambda)"] = np.log(combined_data["lambda"])
    combined_data = combined_data.rename(columns={"g": "eta"})
    melted_data = combined_data.melt(
        id_vars=["horizon", "eom_ret"],
        value_vars=["log(lambda)", "p", "eta"],
        var_name="name",
        value_name="value"
    )
    melted_data["comb_name"] = melted_data["horizon"] + ": " + melted_data["name"]

    # Plot hyperparameter trends
    plot_hyperparameter_trends(melted_data, colours_theme)

    # Process and plot portfolio tuning results
    tuning_data = process_portfolio_tuning_data(mp, static, pfml, start_year)
    plot_portfolio_tuning_results(tuning_data)


# AR1 plot -------------------------------------------
def compute_ar1_plot(chars, features, cluster_labels):
    """
    Compute and plot AR1 (average monthly autocorrelation) for characteristics grouped by clusters.

    Parameters:
        chars (pd.DataFrame): DataFrame containing characteristics data with columns ["id", "eom", "valid"] and features.
        features (list): List of characteristic feature names.
        cluster_labels (pd.DataFrame): DataFrame with columns ["characteristic", "cluster"].

    Returns:
        matplotlib.figure.Figure: AR1 plot figure.
    """
    # Sort by ID and EOM (end of month)
    chars = chars.sort_values(by=["id", "eom"]).copy()
    chars["lag_ok"] = chars.groupby("id")["eom"].diff().dt.days.between(28, 31)
    ar1_results = []

    # Iterate over features to calculate AR1
    for feature in tqdm(features, desc="Processing features"):
        chars["var"] = chars[feature]
        chars["var_l1"] = chars.groupby("id")["var"].shift(1)

        # Subset data for valid conditions
        valid_subset = chars[
            chars["valid"] &
            chars["lag_ok"] &
            ~chars["var"].isin([0.5, np.nan]) &
            ~chars["var_l1"].isin([0.5, np.nan])
        ].copy()

        # Group by ID and ensure at least 60 observations
        valid_subset["n"] = valid_subset.groupby("id")["id"].transform("size")
        valid_subset = valid_subset[valid_subset["n"] >= 12 * 5]

        # Calculate AR1 for each ID
        id_ar1 = valid_subset.groupby("id").apply(
            lambda group: np.corrcoef(group["var"], group["var_l1"])[0, 1] if len(group) > 1 else 1
        ).fillna(1).reset_index(name="ar1")

        # Compute mean AR1 for the feature
        ar1_mean = id_ar1["ar1"].mean()
        ar1_results.append({"char": feature, "ar1": ar1_mean})

    ar1_df = pd.DataFrame(ar1_results)
    ar1_df = ar1_df.merge(cluster_labels, left_on="char", right_on="characteristic")

    # Compute cluster-level averages for sorting
    cluster_means = (
        ar1_df.groupby("cluster")["ar1"]
        .mean()
        .sort_values()
        .index
    )

    # Update cluster names for presentation
    ar1_df["pretty_name"] = ar1_df["cluster"].str.replace("_", " ").str.replace("short term", "short-term").str.title()
    ar1_df["pretty_name"] = pd.Categorical(ar1_df["pretty_name"], categories=cluster_means, ordered=True)

    # Sort features within clusters
    ar1_df["sort_var"] = ar1_df.groupby("cluster")["ar1"].transform("mean") + ar1_df["ar1"] / 100000
    ar1_df = ar1_df.sort_values(by=["pretty_name", "sort_var"], ascending=[True, True])

    # Plot AR1
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=ar1_df,
        y="char",
        x="ar1",
        hue="pretty_name",
        dodge=False
    )
    ax.set_title("Average Monthly Autocorrelation (AR1) by Characteristic")
    ax.set_xlabel("Average Monthly Autocorrelation")
    ax.set_ylabel("")
    ax.legend(title="Theme", loc="upper right", frameon=True)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    return plt.gcf()


# Features with sufficient coverage ------------------
def process_features_with_sufficient_coverage(features_m, feat_excl, settings):
    """
    Processes the dataset by filtering and computing features with sufficient coverage.

    Parameters:
        features_m (list): List of primary features.
        feat_excl (list): List of excluded features.
        settings (dict): Dictionary containing screen settings.

    Returns:
        pd.DataFrame: Filtered dataset.
    """
    # Combine features_m and feat_excl
    features_all = list(set(features_m + feat_excl))

    # Define the necessary IDs and columns to load
    ids = [
        "source_crsp", "common", "obs_main", "primary_sec", "exch_main", "id",
        "eom", "sic", "ff49", "size_grp", "me", "rvol_21d", "dolvol_126d"
    ]

    # Load the data
    data = pd.read_csv(
        "../Data/usa.csv",
        usecols=list(set(ids + features_all)),  # Load required columns
        dtype={"eom": str, "sic": str}  # Define column data types
    )

    # The rest of the code remains the same
    data["eom"] = pd.to_datetime(data["eom"], format="%Y%m%d")
    data["dolvol"] = data["dolvol_126d"]
    data["rvol_m"] = data["rvol_252d"] * np.sqrt(21)

    # Filter dataset based on settings
    data = data[
        (data["source_crsp"] == 1) &
        (data["common"] == 1) &
        (data["obs_main"] == 1) &
        (data["primary_sec"] == 1) &
        (data["exch_main"] == 1)
        ]

    # Screen data: Start date
    # Create a numeric column explicitly: 1 for True, 0 for False
    start_screen = np.zeros(len(data), dtype=int)  # Initialize with 0
    start_screen[data["eom"] < settings["screens"]["start"]] = 1  # Set to 1 where condition is True
    excluded_start = round(np.mean(start_screen) * 100, 2)
    print(f"Start date screen excludes {excluded_start}% of the observations")

    # Apply start date filter
    data = data[data["eom"] >= settings["screens"]["start"]]

    # Monitor screen impact
    n_start = len(data)
    me_start = data["me"].sum(skipna=True)

    # Require 'me'
    me_missing = np.zeros(len(data), dtype=int)  # Initialize with 0
    me_missing[data["me"].isna()] = 1  # Set to 1 where 'me' is missing
    excluded_me = round(np.mean(me_missing) * 100, 2)
    print(f"Non-missing 'me' excludes {excluded_me}% of the observations")
    data = data[~data["me"].isna()]

    # Require 'rvol_m'
    if settings["screens"]["require_rvol"]:
        rvol_missing = np.zeros(len(data), dtype=int)  # Initialize with 0
        rvol_missing[data["rvol_m"].isna()] = 1  # Set to 1 where 'rvol_m' is missing
        excluded_rvol = round(np.mean(rvol_missing) * 100, 2)
        print(f"Non-missing 'rvol_252d' excludes {excluded_rvol}% of the observations")
        data = data[~data["rvol_m"].isna()]

    # Require 'dolvol'
    if settings["screens"]["require_dolvol"]:
        dolvol_missing = np.zeros(len(data), dtype=int)  # Initialize with 0
        dolvol_missing[data["dolvol"].isna()] = 1  # Set to 1 where 'dolvol' is missing
        excluded_dolvol = round(np.mean(dolvol_missing) * 100, 2)
        print(f"Non-missing 'dolvol_126d' excludes {excluded_dolvol}% of the observations")
        data = data[~data["dolvol"].isna()]

    # Size screen
    size_screen = np.zeros(len(data), dtype=int)  # Initialize with 0
    size_screen[
        ~data["size_grp"].isin(settings["screens"]["size_grps"])] = 1  # Set to 1 where size group is not in the list
    excluded_size = round(np.mean(size_screen) * 100, 2)
    print(f"Size screen excludes {excluded_size}% of the observations")
    data = data[data["size_grp"].isin(settings["screens"]["size_grps"])]

    # Feature screen
    feat_available = data[features_all].notna().sum(axis=1)
    min_feat = math.floor(len(features_all) * settings["screens"]["feat_pct"])
    excluded_feat_coverage = np.zeros(len(data), dtype=int)  # Initialize with 0
    excluded_feat_coverage[feat_available < min_feat] = 1  # Set to 1 where feature coverage is insufficient
    excluded_feat_coverage_pct = round(np.mean(excluded_feat_coverage) * 100, 2)
    print(
        f"At least {settings['screens']['feat_pct'] * 100}% of feature excludes {excluded_feat_coverage_pct}% of the observations")
    data = data[feat_available >= min_feat]

    # Summary
    final_obs_pct = round((len(data) / n_start) * 100, 2)
    final_market_cap_pct = round((data["me"].sum() / me_start) * 100, 2)
    print(
        f"In total, the final dataset has {final_obs_pct}% of the observations and {final_market_cap_pct}% of the market cap in the post {settings['screens']['start']} data")

    # Check coverage by 'eom'
    coverage = data[features_all].notna().groupby(data["eom"]).mean()
    return data, coverage




