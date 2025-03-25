import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
from scipy.stats import norm
from tqdm import tqdm
import math
from a_general_functions import create_cov


# Load base case portfolios -----------
def load_base_case_portfolios(base_path):
    """
    Load base case portfolio data from the specified directory.

    Parameters:
        base_path (str): Path to the base portfolio directory.

    Returns:
        dict: Dictionary containing dataframes for 'mp', 'pfml', 'static', 'bm_pfs', 'tpf', 'factor_ml', and 'mkt'.
    """
    # Locate the base folder
    base_folder = next(os.walk(base_path))[1][0]

    # Load required datasets
    mp = pd.read_pickle(os.path.join(base_path, base_folder, "multiperiod-ml.pkl"))
    pfml = pd.read_pickle(os.path.join(base_path, base_folder, "portfolio-ml.pkl"))
    static = pd.read_pickle(os.path.join(base_path, base_folder, "static-ml.pkl"))
    bm_pfs = pd.read_csv(os.path.join(base_path, base_folder, "bms.csv"))
    tpf = pd.read_pickle(os.path.join(base_path, base_folder, "tpf.pkl"))
    factor_ml = pd.read_pickle(os.path.join(base_path, base_folder, "factor_ml.pkl"))
    mkt = pd.read_pickle(os.path.join(base_path, base_folder, "mkt.pkl"))

    # Update type naming convention for bm_pfs
    bm_pfs["eom_ret"] = pd.to_datetime(bm_pfs["eom_ret"])
    bm_pfs["type"] = bm_pfs["type"].replace("Rank-Weighted", "Rank-ML")

    return {
        "mp": mp,
        "pfml": pfml,
        "static": static,
        "bm_pfs": bm_pfs,
        "tpf": tpf,
        "factor_ml": factor_ml,
        "mkt": mkt  # Include market portfolio
    }


# Final Portfolios ---------------------
def combine_portfolios(mp, pfml, static, bm_pfs, pf_order_new, gamma_rel):
    """
    Combine multiple portfolios into a single dataframe.

    Parameters:
        mp (dict): Multiperiod-ML portfolio data.
        pfml (dict): Portfolio-ML data.
        static (dict): Static-ML data (now handled as a dictionary).
        bm_pfs (pd.DataFrame): Benchmark portfolios data.
        pf_order_new (list): Order of portfolio types for factor conversion.
        gamma_rel (float): Gamma value for risk aversion.

    Returns:
        pd.DataFrame: Combined portfolio dataframe with utility adjustments.
    """
    # Ensure all relevant dates are converted to Timestamps
    mp['pf']['eom_ret'] = pd.to_datetime(mp['pf']['eom_ret'])
    pfml['pf']['eom_ret'] = pd.to_datetime(pfml['pf']['eom_ret'])
    static['pf']['eom_ret'] = pd.to_datetime(static['pf']['eom_ret'])
    bm_pfs['eom_ret'] = pd.to_datetime(bm_pfs['eom_ret'])

    # Process each component separately
    mp_pf = mp['pf'].copy()
    pfml_pf = pfml['pf'].copy()
    static_pf = static['pf'].copy()  # Use 'pf' from the 'static' dictionary
    bm_pfs_pf = bm_pfs.copy()

    # Extract and process hps data for Multiperiod-ML and Static-ML
    mp_hps = mp['hps'].loc[
        (mp['hps']['eom_ret'].isin(mp['pf']['eom_ret'])) &
        (mp['hps']['k'] == 1) & (mp['hps']['g'] == 0) & (mp['hps']['u'] == 1),
        ['eom_ret', 'inv', 'shorting', 'turnover', 'r', 'tc']
    ].assign(type='Multiperiod-ML')

    static_hps = static['hps'].loc[
        (static['hps']['eom_ret'].isin(static['pf']['eom_ret'])) &  # Match only those with relevant 'eom_ret'
        (static['hps']['k'] == 1) & (static['hps']['g'] == 0) & (static['hps']['u'] == 1),
        ['eom_ret', 'inv', 'shorting', 'turnover', 'r', 'tc']
    ].assign(type='Static-ML')

    # Combine all parts together
    pfs = pd.concat([mp_pf, pfml_pf, static_pf, bm_pfs_pf, mp_hps, static_hps], ignore_index=True)

    # Remove the 'eom' column if it exists in the combined DataFrame
    if 'eom' in pfs.columns:
        pfs.drop(columns=['eom'], inplace=True)

    # Convert 'type' to categorical with specified order
    pfs['type'] = pd.Categorical(pfs['type'], categories=pf_order_new, ordered=True)

    # Sort the dataframe
    pfs.sort_values(by=['type', 'eom_ret'], inplace=True)

    # Compute utility and adjusted variables
    pfs['e_var_adj'] = pfs.groupby('type')['r'].transform(lambda x: (x - x.mean()) ** 2)
    pfs['utility_t'] = pfs['r'] - pfs['tc'] - 0.5 * pfs['e_var_adj'] * gamma_rel

    return pfs


# Portfolio summary stats --------------
def compute_portfolio_summary(pfs, main_types, pf_order, gamma_rel):
    """
    Compute portfolio summary statistics.

    Parameters:
        pfs (pd.DataFrame): Combined portfolio dataframe.
        main_types (list): List of essential portfolio types to include.
        pf_order (list): Full list of portfolio types for categorization.
        gamma_rel (float): Gamma value for risk aversion.

    Returns:
        pd.DataFrame: Summary statistics for each portfolio type.
        pd.DataFrame: Filtered portfolios containing only main types.
    """
    # Make a copy to avoid modifying the original DataFrame
    pfs_copy = pfs.copy()

    # Convert 'type' to categorical with full order
    pfs_copy["type"] = pd.Categorical(pfs_copy["type"], categories=pf_order, ordered=True)

    # Calculate summary statistics for each type
    pf_summary = (
        pfs_copy.groupby("type")
        .agg(
            n=("type", "size"),
            inv=("inv", "mean"),
            shorting=("shorting", "mean"),
            turnover_notional=("turnover", "mean"),
            r=("r", lambda x: x.mean() * 12),
            sd=("r", lambda x: x.std() * np.sqrt(12)),
            sr_gross=("r", lambda x: x.mean() / x.std() * np.sqrt(12)),
            tc=("tc", lambda x: x.mean() * 12),
            r_tc=("r", lambda x: (x - pfs_copy.loc[x.index, "tc"]).mean() * 12),
            sr=("r", lambda x: ((x - pfs_copy.loc[x.index, "tc"]).mean()) / x.std() * np.sqrt(12)),
            obj=("r", lambda x: (x.mean() - 0.5 * x.var() * gamma_rel - pfs_copy.loc[x.index, "tc"].mean()) * 12),
        )
        .reset_index()
    )

    # Filter the portfolio to include only main types
    pfs_filtered = pfs_copy[pfs_copy["type"].isin(main_types)].copy()
    pfs_filtered["type"] = pd.Categorical(pfs_filtered["type"], categories=main_types, ordered=True)

    # Sort the filtered dataframe
    pfs_filtered = pfs_filtered.sort_values(by=["type", "eom_ret"])

    return pf_summary, pfs_filtered


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
    pfs['cumret'] = pfs.groupby('type', observed=True)['r'].cumsum()
    pfs['cumret_tc'] = pfs['r'] - pfs['tc']
    pfs['cumret_tc'] = pfs.groupby('type', observed=True)['cumret_tc'].cumsum()
    pfs['cumret_tc_risk'] = pfs.groupby('type', observed=True)['utility_t'].cumsum()

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
        'cumret': 'Gross Return',
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
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True)

        # Set axis limits
        ax.set_xlim(pd.Timestamp(start_date), pd.Timestamp(end_date))

        if metric == 'Gross Return':
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
        cov_list (dict): Covariance matrices (as dictionaries with keys: 'fct_load', 'fct_cov', 'ivol_vec').
        weights (pd.DataFrame): Portfolio weights.

    Returns:
        pd.DataFrame: DataFrame containing expected risks for each portfolio and date.
    """
    risk_records = []

    # Sort weights to ensure consistency
    weights = weights.sort_values(by=['type', 'eom', 'id'])
    weights_list = {date: weights[weights["eom"] == date] for date in dates}

    for date in dates:
        if date not in cov_list:
            continue

        # Generate the covariance matrix for the date in the loop
        w_sub = weights_list.get(date)

        if w_sub is None or w_sub.empty:
            continue

        sigma = create_cov(cov_list[date])

        if sigma is None or sigma.shape[0] == 0:
            continue

        # Loop through all portfolio types for this date
        for portfolio_type in w_sub["type"].unique():
            w = w_sub.loc[w_sub["type"] == portfolio_type, "w"].values.flatten()

            if len(w) == sigma.shape[0]:  # Check if the sizes match
                pf_var = np.dot(w.T, np.dot(sigma, w))  # Calculate variance
                risk_records.append({"type": portfolio_type, "pf_var": pf_var, "eom": date})
            else:
                print(f"Warning: Size mismatch for {portfolio_type} on {date}")

    # Return results as a DataFrame
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
    pfs["eom"] = (pfs["eom_ret"] - pd.offsets.MonthEnd(1))
    merged_stats = pf_vars.merge(
        pfs[["type", "eom", "inv", "turnover"]],
        on=["type", "eom"],
        how="inner"
    )
    merged_stats = merged_stats[merged_stats["type"].isin(main_types)]

    # Compute additional statistics
    # Pre-compute e_sd as a separate series to avoid modifying the original DataFrame
    e_sd_series = np.sqrt(merged_stats["pf_var"].values * 252)

    # Create sub-dataframes directly instead of melting
    dfs = [
        merged_stats[["type", "eom"]].assign(stat="e_sd", value=e_sd_series),
        merged_stats[["type", "eom"]].assign(stat="inv", value=merged_stats["inv"]),
        merged_stats[["type", "eom"]].assign(stat="turnover", value=merged_stats["turnover"])
    ]

    # Concatenate all at once
    merged_stats_long = pd.concat(dfs, ignore_index=True)

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
def plot_apple_vs_xerox(mp, pfml, static, tpf, factor_ml, mkt,
                        pfs, liquid_id, illiquid_id, start_year):
    """
    Compare portfolio weights for a liquid and illiquid stock across different portfolios over time.

    Parameters:
    - mp, pfml, static, tpf, factor_ml, mkt: Dictionaries containing DataFrames of portfolio weights.
    - pfs: DataFrame of additional portfolio statistics.
    - liquid_id, illiquid_id: Stock IDs for liquid (e.g., Apple) and illiquid (e.g., Xerox) stocks.
    - start_year: Starting year for the plot.

    Returns:
    - None. Displays the weight comparison plot.
    """

    liquid_id = str(liquid_id)
    illiquid_id = str(illiquid_id)

    # Generate position_frames directly, checking if "w" exists before accessing
    position_frames = [
        mp["w"][mp["w"]["id"].isin([liquid_id, illiquid_id])].assign(type="Multiperiod-ML*") if "w" in mp else None,
        pfml["w"][pfml["w"]["id"].isin([liquid_id, illiquid_id])].assign(type="Portfolio-ML") if "w" in pfml else None,
        static["w"][static["w"]["id"].isin([liquid_id, illiquid_id])].assign(
            type="Static-ML*") if "w" in static else None,
        tpf["w"][tpf["w"]["id"].isin([liquid_id, illiquid_id])].assign(type="Markowitz-ML") if "w" in tpf else None,
        factor_ml["w"][factor_ml["w"]["id"].isin([liquid_id, illiquid_id])].assign(
            type="Factor-ML") if "w" in factor_ml else None,
        mkt["w"][mkt["w"]["id"].isin([liquid_id, illiquid_id])].assign(type="Market") if "w" in mkt else None,
    ]

    # Remove None entries from the list and concatenate all DataFrames
    positions = pd.concat([frame for frame in position_frames if frame is not None])

    # Define stock type names
    stock_type_map = {
        "14593": "Apple (liquid)",
        "27983": "Xerox (illiquid)",
        "93436": "Tesla",
        "91103": "Visa",
        "19561": "Boeing",
        "10107": "Microsoft",
        "22111": "Johnson and Johnson (liquid)",
        "55976": "Walmart (liquid)"
    }
    positions["stock_type"] = positions["id"].map(stock_type_map).fillna(positions["id"])

    # Filter data by year
    positions["eom"] = pd.to_datetime(positions["eom"])
    positions = positions[positions["eom"].dt.year >= start_year]

    # Make sure the eom in pfs is end-of-month
    pfs["eom"] = pd.to_datetime(pfs["eom_ret"]) + pd.offsets.MonthEnd(0)

    # Merge with portfolio stats
    positions = pd.merge(
        positions,
        pfs[["type", "eom", "inv"]],
        on=["type", "eom"],
        how="left",
    )

    # Standardize weights per stock type
    positions["w_z"] = positions.groupby(["type", "id"])["w"].transform(lambda x: (x - x.mean()) / x.std())

    # Ensure correct ordering for plot categories
    positions["type"] = pd.Categorical(
        positions["type"],
        categories=["Multiperiod-ML*", "Portfolio-ML", "Static-ML*", "Markowitz-ML", "Factor-ML", "Market"],
        ordered=True
    )

    # Plotting
    plt.figure(figsize=(12, 8))
    unique_types = positions["type"].cat.categories

    for i, portfolio_type in enumerate(unique_types):
        subset = positions[positions["type"] == portfolio_type]

        if subset.empty:
            continue  # Skip if no data for this type

        plt.subplot(2, 3, i + 1)

        for stock_type in subset["stock_type"].unique():
            stock_data = subset[subset["stock_type"] == stock_type]
            plt.plot(
                stock_data["eom"],
                stock_data["w"],
                label=f"{stock_type}",
                alpha=0.8,
            )

        plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
        plt.title(portfolio_type)
        plt.xlabel("End of Month")
        plt.ylabel("Portfolio Weight")
        plt.legend(loc="best", fontsize=8)
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

    for entry in model.values():
        # Ensure the entry is valid
        if not isinstance(entry, dict) or "pred" not in entry or "opt_hps" not in entry:
            continue

        # Extract prediction data and hyperparameters
        pred = entry["pred"].copy()
        opt_hps = entry["opt_hps"]

        # Create eom_ret as the last day of the following month
        pred["eom_ret"] = pd.to_datetime(pred["eom"]) + pd.DateOffset(months=1)
        pred["eom_ret"] = pred["eom_ret"] + pd.offsets.MonthEnd(0)

        # Extract unique `eom_ret` values
        unique_eom_ret = pred[["eom_ret"]].drop_duplicates().reset_index(drop=True)

        # Add hyperparameters as columns to `unique_eom_ret`
        unique_eom_ret["lambda"] = opt_hps["lambda"]
        unique_eom_ret["p"] = opt_hps["p"]
        unique_eom_ret["g"] = opt_hps["g"]

        # Add horizon label
        unique_eom_ret["horizon"] = horizon_label

        # Append to the processed data list
        processed_data.append(unique_eom_ret)

    # Combine all processed entries into one DataFrame
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
        pd.DataFrame: Processed and combined tuning data.
    """
    # Ensure columns are present in DataFrames before processing
    required_columns_mp_static = {"rank", "eom_ret", "k", "g", "u"}
    required_columns_pfml = {"eom_ret", "l", "p", "g"}

    if not required_columns_mp_static.issubset(mp.columns) or not required_columns_mp_static.issubset(static.columns):
        raise ValueError("mp or static DataFrames are missing required columns.")

    if not required_columns_pfml.issubset(pfml.columns):
        raise ValueError("pfml DataFrame is missing required columns.")

    # ---- Process Multiperiod-ML Data ----
    mp_hps = mp[
        (mp["rank"] == 1) &
        (mp["eom_ret"].dt.year >= start_year) &
        (mp["eom_ret"].dt.month == 12)
    ][["eom_ret", "k", "g", "u"]].copy()
    mp_hps["type"] = "Multiperiod-ML*"
    mp_hps["horizon"] = "Multiperiod-ML"  # Adding horizon label
    mp_hps.rename(columns={"g": "eta"}, inplace=True)

    # ---- Process Static-ML Data ----
    static_hps = static[
        (static["rank"] == 1) &
        (static["eom_ret"].dt.year >= start_year) &
        (static["eom_ret"].dt.month == 12)
    ][["eom_ret", "k", "g", "u"]].copy()
    static_hps["type"] = "Static-ML*"
    static_hps["horizon"] = "Static-ML"
    static_hps.rename(columns={"g": "eta"}, inplace=True)

    # ---- Process Portfolio-ML Data ----
    pfml_hps = pfml[
        pfml["eom_ret"].dt.year >= start_year
    ][["eom_ret", "l", "p", "g"]].copy()
    pfml_hps["type"] = "Portfolio-ML"
    pfml_hps["log(lambda)"] = np.log(pfml_hps["l"])
    pfml_hps = pfml_hps.drop(columns="l")
    pfml_hps.rename(columns={"g": "eta"}, inplace=True)

    # ---- Reshape Data for Plotting ----
    mp_long = mp_hps.melt(id_vars=["type", "eom_ret", "horizon"])
    static_long = static_hps.melt(id_vars=["type", "eom_ret", "horizon"])
    pfml_long = pfml_hps.melt(id_vars=["type", "eom_ret"])

    # ---- Combine All Results ----
    combined_hps = pd.concat([mp_long, static_long, pfml_long], ignore_index=True)

    # ---- Add Combination Names for Plotting ----
    combined_hps["comb_name"] = combined_hps["type"] + ": " + combined_hps["variable"]

    return combined_hps


def plot_hyperparameter_trends(data, colours_theme):
    """
    Plot the trends for hyperparameters over time in a 3x3 grid.

    Parameters:
        data (pd.DataFrame): Data containing hyperparameter trends.
        colours_theme (list): List of colors for the plot.

    Returns:
        None
    """

    # Define the plot grid (3 rows x 3 columns)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True)
    fig.suptitle("Optimal Hyper-Parameters Over Time", fontsize=16, y=0.93)

    # Create a dictionary to map each combination to a specific subplot
    combination_mapping = {
        "Return t+1: log(lambda)": (0, 0),
        "Return t+1: p": (0, 1),
        "Return t+1: eta": (0, 2),
        "Return t+6: log(lambda)": (1, 0),
        "Return t+6: p": (1, 1),
        "Return t+6: eta": (1, 2),
        "Return t+12: log(lambda)": (2, 0),
        "Return t+12: p": (2, 1),
        "Return t+12: eta": (2, 2)
    }

    # Plot each combination in its respective subplot
    for comb_name, group in data.groupby("comb_name"):
        if comb_name not in combination_mapping:
            continue

        row, col = combination_mapping[comb_name]
        ax = axes[row, col]

        # Plot the scatter plot for this particular combination
        ax.scatter(group["eom_ret"], group["value"], alpha=0.75, color=colours_theme[0])

        # Set titles and labels
        ax.set_title(comb_name, fontsize=12)
        ax.grid(True, linestyle="--", linewidth=0.5)

        if col == 0:
            ax.set_ylabel("Optimal Hyper-Parameter")

        if row == 2:
            ax.set_xlabel("End of Month")

    # Adjust the layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()


def plot_portfolio_tuning_results(data):
    """
    Plot portfolio tuning results over time in a 3x3 grid.

    Parameters:
        data (pd.DataFrame): Data containing portfolio tuning results.

    Returns:
        None
    """
    # Define the plot grid (3 rows x 3 columns)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True)
    fig.suptitle("Portfolio Tuning Results Over Time", fontsize=16, y=0.93)

    # Create a dictionary to map each combination to a specific subplot
    combination_mapping = {
        "Multiperiod-ML*: k": (0, 0),
        "Multiperiod-ML*: eta": (0, 1),
        "Multiperiod-ML*: u": (0, 2),
        "Static-ML*: k": (1, 0),
        "Static-ML*: eta": (1, 1),
        "Static-ML*: u": (1, 2),
        "Portfolio-ML: log(lambda)": (2, 0),
        "Portfolio-ML: p": (2, 1),
        "Portfolio-ML: eta": (2, 2)
    }

    # Plot each combination in its respective subplot
    for comb_name, group in data.groupby("comb_name"):
        if comb_name not in combination_mapping:
            continue

        row, col = combination_mapping[comb_name]
        ax = axes[row, col]

        # Plot the scatter plot for this particular combination
        ax.scatter(group["eom_ret"], group["value"], alpha=0.75, color="steelblue")

        # Set titles and labels
        ax.set_title(comb_name, fontsize=12)
        ax.grid(True, linestyle="--", linewidth=0.5)

        if col == 0:
            ax.set_ylabel("Optimal Hyper-Parameter")

        if row == 2:
            ax.set_xlabel("End of Month")

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()


def plot_optimal_hyperparameters(model_folder, mp, static, pfml, colours_theme, start_year):
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
    data_1 = load_model_hyperparameters(f"{model_folder}/model_1.pkl", "Return t+1")
    data_6 = load_model_hyperparameters(f"{model_folder}/model_6.pkl", "Return t+6")
    data_12 = load_model_hyperparameters(f"{model_folder}/model_12.pkl", "Return t+12")

    # Combine hyperparameter data
    combined_data = pd.concat([data_1, data_6, data_12], ignore_index=True)
    combined_data["log(lambda)"] = np.log(combined_data["lambda"])
    combined_data = combined_data.rename(columns={"g": "eta"})

    # Reshape the DataFrame to a long format
    melted_data = combined_data.melt(
        id_vars=["horizon", "eom_ret"],
        value_vars=["log(lambda)", "p", "eta"],
        var_name="name",
        value_name="value"
    )

    # Create 'comb_name' column
    melted_data["comb_name"] = melted_data["horizon"] + ": " + melted_data["name"]

    # Reorder 'comb_name' levels as per R's factor(levels = c(...))
    desired_order = [
        "Return t+1: log(lambda)", "Return t+1: p", "Return t+1: eta",
        "Return t+6: log(lambda)", "Return t+6: p", "Return t+6: eta",
        "Return t+12: log(lambda)", "Return t+12: p", "Return t+12: eta"
    ]
    melted_data["comb_name"] = pd.Categorical(melted_data["comb_name"], categories=desired_order, ordered=True)

    # Plot hyperparameter trends
    plot_hyperparameter_trends(melted_data, colours_theme)

    # Process and plot portfolio tuning results
    tuning_data = process_portfolio_tuning_data(mp, static, pfml, start_year)
    plot_portfolio_tuning_results(tuning_data)


# Plot AR ----------------------------
def compute_ar1_plot(chars, features, cluster_labels, output_path):
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
    chars["lag_ok"] = (
                chars["eom"] == (chars["prev_eom"] + pd.DateOffset(months=1)).replace(day=1) + pd.offsets.MonthEnd(0))

    ar1_results = []

    # Iterate over features to calculate AR1
    for feature in tqdm(features, desc="Processing features"):
        chars["var"] = chars[feature]
        chars["var_l1"] = chars.groupby("id")["var"].shift(1)

        # Subset data for valid conditions
        valid_subset = chars[
            chars["valid"] &
            chars["lag_ok"] &
            ~chars["var"].isin([0.5]) &
            chars["var"].notna() &
            chars["var_l1"].notna()
        ].copy()

        # Group by ID and ensure at least 60 observations
        valid_subset["n"] = valid_subset.groupby("id")["id"].transform("size")
        valid_subset = valid_subset[valid_subset["n"] >= 12 * 5]

        # Calculate AR1 for each ID
        id_ar1 = valid_subset.groupby("id").apply(
            lambda group: np.corrcoef(group["var"], group["var_l1"])[0, 1] if len(group) > 1 else np.nan
        ).dropna().reset_index(name="ar1")

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
    plt.savefig(f"{output_path}/ar1_plot.png", dpi=300)  # Save as PNG with high resolution
    plt.savefig(f"{output_path}/ar1_plot.pdf")  # Save as PDF (optional)


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




