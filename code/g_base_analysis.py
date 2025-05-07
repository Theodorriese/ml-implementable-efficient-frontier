import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from matplotlib.dates import YearLocator, DateFormatter
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm
from tqdm import tqdm
import math
from a_general_functions import create_cov


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
        # "mp": mp,
        "pfml": pfml,
        "static": static,
        "bm_pfs": bm_pfs,
        "tpf": tpf,
        "factor_ml": factor_ml,
        "mkt": mkt
    }


# Final Portfolios ---------------------
# MP has been removed
def combine_portfolios(pfml, static, bm_pfs, pf_order_new, gamma_rel):
    """
    Combine multiple portfolios into a single dataframe.
    """
    # Ensure all relevant dates are converted to Timestamps
    pfml['pf']['eom_ret'] = pd.to_datetime(pfml['pf']['eom_ret'])
    static['pf']['eom_ret'] = pd.to_datetime(static['pf']['eom_ret'])
    bm_pfs['eom_ret'] = pd.to_datetime(bm_pfs['eom_ret'])

    # Process each component separately
    pfml_pf = pfml['pf'].copy()
    static_pf = static['pf'].copy()
    bm_pfs_pf = bm_pfs.copy()

    static_hps = static['hps'].loc[
        (static['hps']['eom_ret'].isin(static['pf']['eom_ret'])) &
        (static['hps']['k'] == 1) & (static['hps']['g'] == 0) & (static['hps']['u'] == 1),
        ['eom_ret', 'inv', 'shorting', 'turnover', 'r', 'tc']
    ].assign(type='Static-ML')

    # Combine all parts together
    pfs = pd.concat([pfml_pf, bm_pfs_pf, static_pf, static_hps], ignore_index=True)

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


# Portfolio time series --------------
def ts_plot(pfs, main_types, start_date, end_date, save_path=None):
    """
    Computes and creates three separate performance plots (gross, net of TC, net of TC & risk)
    with optional saving to disk.

    Parameters:
        pfs (pd.DataFrame): Performance DataFrame with 'r', 'tc', 'utility_t', 'eom_ret', 'type'.
        main_types (list): Portfolio types to include.
        start_date (str): Start date of the plot.
        end_date (str): End date of the plot.
        save_path (str or Path, optional): If provided, saves figures as PNGs in this directory.

    Returns:
        dict: Dictionary with matplotlib figures.
    """

    pfs = pfs.copy()
    pfs['cumret'] = pfs.groupby('type', observed=True)['r'].cumsum()
    pfs['cumret_tc'] = pfs['r'] - pfs['tc']
    pfs['cumret_tc'] = pfs.groupby('type', observed=True)['cumret_tc'].cumsum()
    pfs['cumret_tc_risk'] = pfs.groupby('type', observed=True)['utility_t'].cumsum()

    ts_data = (
        pfs[pfs['type'].isin(main_types)]
        .melt(
            id_vars=['type', 'eom_ret'],
            value_vars=['cumret', 'cumret_tc', 'cumret_tc_risk'],
            var_name='metric'
        )
    )

    initial_values = pd.DataFrame({
        'eom_ret': pd.Timestamp(start_date) - pd.offsets.MonthBegin(),
        'type': main_types * 3,
        'value': 0,
        'metric': (
                ['cumret'] * len(main_types)
                + ['cumret_tc'] * len(main_types)
                + ['cumret_tc_risk'] * len(main_types)
        )
    })
    ts_data = pd.concat([ts_data, initial_values], ignore_index=True)
    ts_data['eom_ret'] = pd.to_datetime(ts_data['eom_ret'])
    ts_data = ts_data.sort_values(by=['type', 'eom_ret', 'metric'])

    metric_labels = {
        'cumret': 'Gross Return',
        'cumret_tc': 'Return net of TC',
        'cumret_tc_risk': 'Return net of TC and Risk'
    }
    ts_data['metric_pretty'] = ts_data['metric'].map(metric_labels)
    ts_data['metric_pretty'] = pd.Categorical(
        ts_data['metric_pretty'],
        categories=list(metric_labels.values())
    )

    figures = {}
    for metric in ts_data['metric_pretty'].cat.categories:
        plot_data = ts_data[ts_data['metric_pretty'] == metric]
        fig, ax = plt.subplots(figsize=(10, 6))

        for portfolio_type in main_types:
            subset = plot_data[plot_data['type'] == portfolio_type]
            ax.plot(subset['eom_ret'], subset['value'], label=portfolio_type, linewidth=2)

        ax.set_title(metric)
        ax.set_xlabel("End of Month")
        ax.set_ylabel("Cumulative Performance")
        ax.xaxis.set_major_formatter(DateFormatter('%Y'))
        ax.tick_params(axis='x', rotation=45)
        ax.grid(False)
        pad_months = 12
        ax.set_xlim(pd.Timestamp(start_date) - pd.DateOffset(months=pad_months), pd.Timestamp(end_date))

        ax.legend(loc='best', frameon=False)
        fig.tight_layout()

        figures[metric] = fig

        # Save if path provided
        if save_path:
            filename = metric.lower().replace(" ", "_").replace("tc", "tc").replace("&", "and") + ".png"
            fig.savefig(os.path.join(save_path, filename), bbox_inches="tight")

    return figures



# Portfolio time series with manual y-axis ----------------
def ts_plot_manual_ylim(pfs, main_types, start_date, end_date, save_path=None):
    """
    Computes and creates three separate performance plots (gross, net of TC, net of TC & risk)
    with manual y-axis limits and optional saving to disk.

    Parameters:
        pfs (pd.DataFrame): Performance DataFrame with 'r', 'tc', 'utility_t', 'eom_ret', 'type'.
        main_types (list): Portfolio types to include.
        start_date (str): Start date of the plot.
        end_date (str): End date of the plot.
        save_path (str or Path, optional): If provided, saves figures as PNGs in this directory.

    Returns:
        dict: Dictionary with matplotlib figures.
    """

    pfs = pfs.copy()
    pfs['cumret'] = pfs.groupby('type', observed=True)['r'].cumsum()
    pfs['cumret_tc'] = pfs['r'] - pfs['tc']
    pfs['cumret_tc'] = pfs.groupby('type', observed=True)['cumret_tc'].cumsum()
    pfs['cumret_tc_risk'] = pfs.groupby('type', observed=True)['utility_t'].cumsum()

    ts_data = (
        pfs[pfs['type'].isin(main_types)]
        .melt(
            id_vars=['type', 'eom_ret'],
            value_vars=['cumret', 'cumret_tc', 'cumret_tc_risk'],
            var_name='metric'
        )
    )

    initial_values = pd.DataFrame({
        'eom_ret': pd.Timestamp(start_date) - pd.offsets.MonthBegin(),
        'type': main_types * 3,
        'value': 0,
        'metric': (
            ['cumret'] * len(main_types)
            + ['cumret_tc'] * len(main_types)
            + ['cumret_tc_risk'] * len(main_types)
        )
    })
    ts_data = pd.concat([ts_data, initial_values], ignore_index=True)
    ts_data['eom_ret'] = pd.to_datetime(ts_data['eom_ret'])
    ts_data = ts_data.sort_values(by=['type', 'eom_ret', 'metric'])

    metric_labels = {
        'cumret': 'Gross Return',
        'cumret_tc': 'Return net of TC',
        'cumret_tc_risk': 'Return net of TC and Risk'
    }
    ts_data['metric_pretty'] = ts_data['metric'].map(metric_labels)
    ts_data['metric_pretty'] = pd.Categorical(
        ts_data['metric_pretty'],
        categories=list(metric_labels.values())
    )

    # Define manual y-axis limits
    y_axis_limits = {
        'Gross Return': (-0.5, 3.5),
        'Return net of TC': (-0.5, 1.5),
        'Return net of TC and Risk': (-0.5, 1.12)
    }

    figures = {}
    for metric in ts_data['metric_pretty'].cat.categories:
        plot_data = ts_data[ts_data['metric_pretty'] == metric]
        fig, ax = plt.subplots(figsize=(10, 6))

        for portfolio_type in main_types:
            subset = plot_data[plot_data['type'] == portfolio_type]
            ax.plot(subset['eom_ret'], subset['value'], label=portfolio_type, linewidth=2)

        ax.set_title(metric)
        ax.set_xlabel("End of Month")
        ax.set_ylabel("Cumulative Performance")
        ax.xaxis.set_major_formatter(DateFormatter('%Y'))
        ax.tick_params(axis='x', rotation=45)
        ax.grid(False)
        pad_months = 12
        ax.set_xlim(pd.Timestamp(start_date) - pd.DateOffset(months=pad_months), pd.Timestamp(end_date))

        # Set y-axis limit manually
        if metric in y_axis_limits:
            ax.set_ylim(y_axis_limits[metric])

        ax.legend(loc='best', frameon=False)
        fig.tight_layout()

        figures[metric] = fig
        if save_path:
            name_map = {
                'Gross Return': "performance_gross_return",
                'Return net of TC': "performance_net_tc",
                'Return net of TC and Risk': "performance_net_tc_risk"
            }
            filename = f"{name_map.get(metric, metric.lower().replace(' ', '_'))}_man_y_limit.png"
            fig.savefig(os.path.join(save_path, filename), bbox_inches="tight")

    return figures



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
def compute_and_plot_portfolio_statistics_over_time(pfml, tpf, static, factor_ml, pfs, barra_cov, dates_oos,
                                                    pf_order, main_types):
    """
    Computes and plots portfolio-level statistics (ex-ante volatility, leverage, turnover) over time.

    Parameters:
        pfml (dict): Portfolio-ML output with weights in 'w'.
        tpf (dict): Markowitz (Tangency) portfolio output with weights in 'w'.
        static (dict): Static-ML* portfolio output with weights in 'w'.
        factor_ml (dict): Factor-ML portfolio output with weights in 'w'.
        pfs (pd.DataFrame): Portfolio-level summary statistics including 'inv' and 'turnover'.
        barra_cov (dict): Dictionary with stock-level factor loadings and covariance information.
        dates_oos (list): List of out-of-sample evaluation dates.
        pf_order (list): Desired plotting order for portfolio types.
        main_types (list): Portfolio types to include in the final plot.

    Returns:
        matplotlib.figure.Figure: Figure object with portfolio statistics plotted over time.
    """


    # Combine portfolio weights
    pfml_weights = pfml["w"].copy();
    pfml_weights["type"] = "Portfolio-ML"
    tpf_weights = tpf["w"].copy();
    tpf_weights["type"] = "Markowitz-ML"
    # mp_weights = mp["w"].copy(); mp_weights["type"] = "Multiperiod-ML*"
    static_weights = static["w"].copy();
    static_weights["type"] = "Static-ML*";
    static_weights.drop(columns=["pred_ld1"], inplace=True)
    factor_ml_weights = factor_ml["w"].copy();
    factor_ml_weights["type"] = "Factor-ML"

    combined_weights = pd.concat(
        [pfml_weights, tpf_weights, static_weights, factor_ml_weights],
        ignore_index=True
    )
    combined_weights["type"] = pd.Categorical(combined_weights["type"], categories=pf_order, ordered=True)

    # Compute expected risks
    pf_vars = compute_expected_risk(dates_oos, barra_cov, combined_weights)

    # Merge with pfs
    pfs = pfs.copy()
    pfs["eom"] = pfs["eom_ret"] - pd.offsets.MonthEnd(1)
    merged_stats = pf_vars.merge(pfs[["type", "eom", "inv", "turnover"]], on=["type", "eom"], how="inner")
    merged_stats = merged_stats[merged_stats["type"].isin(main_types)]

    # Create long format for plotting
    e_sd_series = np.sqrt(merged_stats["pf_var"].values * 252)
    dfs = [
        merged_stats[["type", "eom"]].assign(stat="e_sd", value=e_sd_series),
        merged_stats[["type", "eom"]].assign(stat="inv", value=merged_stats["inv"]),
        merged_stats[["type", "eom"]].assign(stat="turnover", value=merged_stats["turnover"])
    ]
    merged_stats_long = pd.concat(dfs, ignore_index=True)

    stat_name_map = {
        "e_sd": "Ex-ante Volatility",
        "inv": "Leverage",
        "turnover": "Turnover"
    }
    merged_stats_long["stat"] = merged_stats_long["stat"].map(stat_name_map)

    # Plotting
    sns.set_theme(style="white")  # Removes background grids
    g = sns.FacetGrid(
        merged_stats_long, col="stat", sharey=False, height=2.5, aspect=3.0,
        col_wrap=1, margin_titles=True
    )
    g.map_dataframe(
        sns.lineplot, x="eom", y="value", hue="type", style="type", palette="deep"
    )

    # Set labels and titles
    g.set_axis_labels("Date", "Value")
    g.set_titles(col_template="{col_name}")
    g.set(yscale="log")

    # Turn off gridlines and disable scientific y-axis labels
    for ax in g.axes.flat:
        ax.grid(False)  # Remove gridlines
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(style='plain', axis='y')

    # Add one legend centered above all plots
    g.add_legend(bbox_to_anchor=(0.5, 1.05), loc="upper center", ncol=len(main_types), frameon=False)

    # Adjust layout for legend spacing
    g.tight_layout(rect=[0, 0, 1, 0.97])

    return g.figure


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


# liquid vs. illiquid ----------------------------------------------------
def plot_liquid_vs_illiquid(pfml, static, tpf, factor_ml, mkt,
                        pfs, liquid_id, illiquid_id, start_year):
    """
    Compare portfolio weights for liquid and illiquid across different portfolios over time.

    Returns:
        matplotlib.figure.Figure
    """
    liquid_id = str(liquid_id)
    illiquid_id = str(illiquid_id)

    position_frames = [
        pfml["w"][pfml["w"]["id"].isin([liquid_id, illiquid_id])].assign(type="Portfolio-ML") if "w" in pfml else None,
        static["w"][static["w"]["id"].isin([liquid_id, illiquid_id])].assign(type="Static-ML*") if "w" in static else None,
        tpf["w"][tpf["w"]["id"].isin([liquid_id, illiquid_id])].assign(type="Markowitz-ML") if "w" in tpf else None,
        factor_ml["w"][factor_ml["w"]["id"].isin([liquid_id, illiquid_id])].assign(type="Factor-ML") if "w" in factor_ml else None,
        mkt["w"][mkt["w"]["id"].isin([liquid_id, illiquid_id])].assign(type="Market") if "w" in mkt else None,
    ]

    positions = pd.concat([frame for frame in position_frames if frame is not None])

    stock_type_map = {
        "10107": "Microsoft (liquid)",
        "27983": "Xerox (illiquid)"
    }
    positions["stock_type"] = positions["id"].map(stock_type_map).fillna(positions["id"])

    positions["eom"] = pd.to_datetime(positions["eom"])
    positions = positions[positions["eom"].dt.year >= start_year]

    pfs["eom"] = pd.to_datetime(pfs["eom_ret"]) + pd.offsets.MonthEnd(0)
    positions = pd.merge(
        positions,
        pfs[["type", "eom", "inv"]],
        on=["type", "eom"],
        how="left",
    )

    positions["type"] = pd.Categorical(
        positions["type"],
        categories=["Portfolio-ML", "Static-ML*", "Markowitz-ML", "Factor-ML", "Market"],
        ordered=True
    )

    # Plotting with dynamic Y-axis and no grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharex=False, sharey=False)
    axes = axes.flatten()

    for i, portfolio_type in enumerate(positions["type"].cat.categories):
        subset = positions[positions["type"] == portfolio_type]
        if subset.empty:
            continue

        ax = axes[i]
        for stock_type in ["Microsoft (liquid)", "Xerox (illiquid)"]:
            stock_data = subset[subset["stock_type"] == stock_type]
            if not stock_data.empty:
                ax.plot(stock_data["eom"], stock_data["w"], label=stock_type, alpha=0.85, linewidth=2)

        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(portfolio_type)
        ax.set_xlabel("End of Month")
        ax.set_ylabel("Weight")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(False)  # Disable grid
        ax.legend(fontsize=10)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Portfolio Weights: Microsoft vs. Xerox", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    return fig




# Optimal Hyper-parameters ----------------------------
def load_model_hyperparameters(model_path, horizon_label):
    """
    Loads and processes optimal hyperparameters from a saved model for a given prediction horizon.

    Parameters:
        model_path (str): Path to the pickled model file.
        horizon_label (str): Label indicating the prediction horizon (e.g., "Return t+1").

    Returns:
        pd.DataFrame: DataFrame with columns ['eom_ret', 'lambda', 'p', 'g', 'horizon']
                      for each time period with available hyperparameters.
    """
    model = pd.read_pickle(model_path)
    processed_data = []

    for entry in model.values():
        if not isinstance(entry, dict) or "pred" not in entry or "opt_hps" not in entry:
            continue

        pred = entry["pred"].copy()
        opt_hps = entry["opt_hps"]
        pred["eom_ret"] = pd.to_datetime(pred["eom"]) + pd.DateOffset(months=1)
        pred["eom_ret"] = pred["eom_ret"] + pd.offsets.MonthEnd(0)
        unique_eom_ret = pred[["eom_ret"]].drop_duplicates().reset_index(drop=True)
        unique_eom_ret["lambda"] = opt_hps["lambda"]
        unique_eom_ret["p"] = opt_hps["p"]
        unique_eom_ret["g"] = opt_hps["g"]
        unique_eom_ret["horizon"] = horizon_label
        processed_data.append(unique_eom_ret)

    return pd.concat(processed_data, ignore_index=True)


def process_portfolio_tuning_data(static, pfml, start_year):
    """
    Processes and reshapes hyperparameter tuning results for Static-ML* and Portfolio-ML
    into a long format suitable for plotting.

    Parameters:
        static (pd.DataFrame): Static-ML* tuning results with optimal parameters per year-end.
        pfml (pd.DataFrame): Portfolio-ML tuning results with hyperparameters per month.
        start_year (int): Year threshold for filtering data (inclusive).

    Returns:
        pd.DataFrame: Long-form DataFrame with ['type', 'eom_ret', 'name', 'value', 'comb_name'] columns,
                      ready for faceted plotting.
    """

    static_hps = static[
        (static["rank"] == 1) &
        (static["eom_ret"].dt.year >= start_year) &
        (static["eom_ret"].dt.month == 12)
    ][["eom_ret", "k", "g", "u"]].copy()
    static_hps["type"] = "Static-ML*"
    static_hps = static_hps.rename(columns={"g": "v"})
    static_long = static_hps.melt(id_vars=["type", "eom_ret"], var_name="name", value_name="value")


    pfml_hps = pfml[pfml["eom_ret"].dt.year >= start_year].copy()
    pfml_hps["type"] = "Portfolio-ML"
    pfml_hps["log(lambda)"] = pfml_hps["l"].apply(lambda x: "zero" if x == 0 else np.log(x))
    pfml_hps = pfml_hps.rename(columns={"g": "eta"})
    pfml_hps = pfml_hps[["type", "eom_ret", "log(lambda)", "p", "eta"]]
    pfml_long = pfml_hps.melt(id_vars=["type", "eom_ret"], var_name="name", value_name="value")

    df = pd.concat([pfml_long, static_long], ignore_index=True)
    df["comb_name"] = df["type"] + ": " + df["name"]

    facet_levels = (
        [f"Portfolio-ML: {x}" for x in ["log(lambda)", "p", "eta"]] +
        [f"Static-ML*: {x}" for x in ["k", "u", "v"]]
    )

    df["comb_name"] = pd.Categorical(df["comb_name"], categories=facet_levels, ordered=True)
    df = df.sort_values(by=["type", "eom_ret"])
    return df


def plot_hyperparameter_trends(data, colours_theme):
    """
    Plots the time trends of optimal hyperparameters for different return horizons
    in a 3x3 grid of subplots.

    Parameters:
        data (pd.DataFrame): Long-form DataFrame with 'eom_ret', 'name', 'value', and 'comb_name' columns.
        colours_theme (list): List of color codes for consistent plotting aesthetics.

    Returns:
        matplotlib.figure.Figure: The generated matplotlib figure showing hyperparameter trends over time.
    """

    data["eom_ret"] = pd.to_datetime(data["eom_ret"])
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True)
    fig.suptitle("Optimal Hyper-Parameters Over Time", fontsize=16, y=0.93)

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

    for comb_name, group in data.groupby("comb_name"):
        if comb_name not in combination_mapping:
            continue
        row, col = combination_mapping[comb_name]
        ax = axes[row, col]

        if group["name"].iloc[0] == "log(lambda)":
            zero_mask = group["value"] == "zero"
            non_zero = group[~zero_mask].copy()
            non_zero["value"] = non_zero["value"].astype(float)
            ax.scatter(non_zero["eom_ret"], non_zero["value"], alpha=0.75, color=colours_theme[0])
            if zero_mask.any():
                ax.scatter(group.loc[zero_mask, "eom_ret"], [0] * zero_mask.sum(), marker='x', color='red')
        else:
            ax.scatter(group["eom_ret"], group["value"], alpha=0.75, color=colours_theme[0])

        ax.set_title(comb_name, fontsize=12)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True, prune='both'))
        ax.xaxis.set_major_formatter(DateFormatter('%Y'))

    fig.supxlabel("End of Month", fontsize=14)
    fig.supylabel("Optimal Hyper-Parameter", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_hyper = plt.gcf()
    plt.show()
    return fig_hyper


def plot_portfolio_tuning_results(data):
    """
    Plots optimal hyperparameter tuning results over time for both Portfolio-ML and Static-ML* models.

    Parameters:
        data (pd.DataFrame): Long-form DataFrame with 'eom_ret', 'name', 'value', and 'comb_name' columns,
                             including processed tuning data.

    Returns:
        matplotlib.figure.Figure: The generated matplotlib figure visualizing hyperparameter evolution.
    """

    data["eom_ret"] = pd.to_datetime(data["eom_ret"])
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    fig.suptitle("Portfolio Tuning Results Over Time", fontsize=16, y=0.93)

    combination_mapping = {
        "Portfolio-ML: log(lambda)": (0, 0),
        "Portfolio-ML: p": (0, 1),
        "Portfolio-ML: eta": (0, 2),
        "Static-ML*: k": (1, 0),
        "Static-ML*: u": (1, 1),
        "Static-ML*: v": (1, 2)
    }

    for comb_name, group in data.groupby("comb_name"):
        if comb_name not in combination_mapping:
            continue
        row, col = combination_mapping[comb_name]
        ax = axes[row, col]

        if comb_name == "Portfolio-ML: log(lambda)":
            zero_mask = group["value"] == "zero"
            non_zero = group[~zero_mask].copy()
            non_zero["value"] = non_zero["value"].astype(float)
            ax.scatter(non_zero["eom_ret"], non_zero["value"], alpha=0.75, color="steelblue")
            if zero_mask.any():
                ax.scatter(group.loc[zero_mask, "eom_ret"], [0] * zero_mask.sum(), marker='x', color='red')
        else:
            ax.scatter(group["eom_ret"], group["value"], alpha=0.75, color="steelblue")

        ax.set_title(comb_name, fontsize=12)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True, prune='both'))
        ax.xaxis.set_major_formatter(DateFormatter('%Y'))

    fig.supxlabel("End of Month", fontsize=14)
    fig.supylabel("Optimal Hyper-Parameter", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_tuning = plt.gcf()
    plt.show()
    return fig_tuning



def plot_optimal_hyperparameters(model_folder, static, pfml, colours_theme, start_year):
    """
    Loads, processes, and visualizes optimal hyperparameters over time for both Portfolio-ML and Static-ML* models.

    Parameters:
        model_folder (str): Path to folder containing pickled hyperparameter model files.
        static (pd.DataFrame): Static-ML* tuning results.
        pfml (pd.DataFrame): Portfolio-ML tuning results.
        colours_theme (list): List of colors for plotting.
        start_year (int): Minimum year for inclusion in the portfolio tuning plot.

    Returns:
        tuple: A tuple containing:
            - fig_hyper (matplotlib.figure.Figure): Figure of hyperparameter trends.
            - fig_tuning (matplotlib.figure.Figure): Figure of portfolio tuning results.
    """

    data_1 = load_model_hyperparameters(f"{model_folder}/model_1.pkl", "Return t+1")
    data_6 = load_model_hyperparameters(f"{model_folder}/model_6.pkl", "Return t+6")
    data_12 = load_model_hyperparameters(f"{model_folder}/model_12.pkl", "Return t+12")

    combined_data = pd.concat([data_1, data_6, data_12], ignore_index=True)
    combined_data = combined_data.rename(columns={"g": "eta"})
    combined_data["log_lambda"] = combined_data["lambda"].apply(lambda x: "zero" if x == 0 else np.log(x))

    melted_data = pd.melt(
        combined_data,
        id_vars=["horizon", "eom_ret"],
        value_vars=["log_lambda", "p", "eta"],
        var_name="name",
        value_name="value"
    )

    melted_data["name"] = melted_data["name"].replace({"log_lambda": "log(lambda)"})
    melted_data["comb_name"] = melted_data["horizon"] + ": " + melted_data["name"]

    desired_order = [
        "Return t+1: log(lambda)", "Return t+1: p", "Return t+1: eta",
        "Return t+6: log(lambda)", "Return t+6: p", "Return t+6: eta",
        "Return t+12: log(lambda)", "Return t+12: p", "Return t+12: eta"
    ]
    melted_data["comb_name"] = pd.Categorical(melted_data["comb_name"], categories=desired_order, ordered=True)

    fig_hyper = plot_hyperparameter_trends(melted_data, colours_theme)
    tuning_data = process_portfolio_tuning_data(static, pfml, start_year)
    fig_tuning = plot_portfolio_tuning_results(tuning_data)

    return fig_hyper, fig_tuning


# Plot AR ----------------------------
def compute_ar1_plot(chars, features, cluster_labels, output_path=None):
    """
    Compute and plot AR1 (average monthly autocorrelation) for characteristics grouped by clusters.

    Parameters:
        chars (pd.DataFrame): DataFrame containing characteristics data with columns ["id", "eom", "valid"] and features.
        features (list): List of characteristic feature names.
        cluster_labels (pd.DataFrame): DataFrame with columns ["characteristic", "cluster"].
        output_path (str, optional): If provided, saves the figure to this path.

    Returns:
        matplotlib.figure.Figure: AR1 plot figure.
    """

    # # Sort by ID and EOM (end of month)
    # chars = chars.sort_values(by=["id", "eom"]).copy()
    # chars["prev_eom"] = chars.groupby("id")["eom"].shift(1)
    # chars["lag_ok"] = (chars["eom"] == (chars["prev_eom"] + pd.DateOffset(months=1)).apply(
    #     lambda x: x + pd.offsets.MonthEnd(0)))

    # ar1_results = []

    # for feature in tqdm(features, desc="Processing features"):
    #     chars["var"] = chars[feature]
    #     chars["var_l1"] = chars.groupby("id")["var"].shift(1)

    #     valid_subset = chars[
    #         chars["valid"] &
    #         chars["lag_ok"] &
    #         ~chars["var"].isin([0.5]) &
    #         chars["var"].notna() &
    #         chars["var_l1"].notna()
    #     ].copy()

    #     valid_subset["n"] = valid_subset.groupby("id")["id"].transform("size")
    #     valid_subset = valid_subset[valid_subset["n"] >= 12 * 5]

    #     id_ar1 = valid_subset.groupby("id").apply(
    #         lambda group: np.corrcoef(group["var"], group["var_l1"])[0, 1] if len(group) > 1 else np.nan
    #     ).dropna().reset_index(name="ar1")

    #     ar1_mean = id_ar1["ar1"].mean()
    #     ar1_results.append({"char": feature, "ar1": ar1_mean})

    # ar1_df = pd.DataFrame(ar1_results)
    # ar1_df = ar1_df.merge(cluster_labels, left_on="char", right_on="characteristic")

    # cluster_means = (
    #     ar1_df.groupby("cluster")["ar1"]
    #     .mean()
    #     .sort_values()
    #     .index
    # )

    # ar1_df["pretty_name"] = ar1_df["cluster"]
    # ar1_df["pretty_name"] = pd.Categorical(ar1_df["pretty_name"], categories=cluster_means, ordered=True)

    # ar1_df["sort_var"] = ar1_df.groupby("cluster")["ar1"].transform("mean") + ar1_df["ar1"] / 100000
    # ar1_df = ar1_df.sort_values(by=["pretty_name", "sort_var"], ascending=[True, True])

    ar1_df = pd.read_pickle("ar1_data.pkl")

    # Plot setup
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 16))

    # Get unique clusters
    unique_themes = ar1_df["pretty_name"].cat.categories

    # Create a custom palette excluding color at index 13 (pink)
    base_palette = sns.color_palette("tab20", 20)
    cleaned_palette = [c for i, c in enumerate(base_palette) if i not in [8, 11, 12, 13]]

    # Map cleaned colors to themes
    palette = cleaned_palette[:len(unique_themes)]
    theme_colors = dict(zip(unique_themes, palette))

    ax = sns.barplot(
        data=ar1_df,
        y="char",
        x="ar1",
        hue="pretty_name",
        dodge=False,
        palette=theme_colors
    )
    ax.set_title("Average Monthly Autocorrelation (AR1) by Characteristic", fontsize=16)
    ax.set_xlabel("Average Monthly Autocorrelation", fontsize=12)
    ax.set_ylabel("")
    ax.tick_params(axis='y', labelsize=9)
    ax.tick_params(axis='x', labelsize=10)

    # Legend outside the plot
    ax.legend(title="Theme", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.gca().invert_yaxis()
    plt.tight_layout(rect=[0, 0, 0.8, 1])

    fig = plt.gcf()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")

    plt.show()

    return fig



# Features with sufficient coverage ------------------
def process_features_with_sufficient_coverage(features, feat_excl, settings, data_path):
    """
    Processes the dataset by filtering and computing features with sufficient coverage.

    Parameters:
        features (list): List of primary features.
        feat_excl (list): List of excluded features.
        settings (dict): Dictionary containing screen settings.

    Returns:
        pd.DataFrame: Filtered dataset.
    """
    # Combine features and feat_excl
    features_all = list(set(features + feat_excl))

    # Define the necessary IDs and columns to load
    ids = [
        "source_crsp", "common", "obs_main", "primary_sec", "exch_main", "id",
        "eom", "sic", "ff49", "size_grp", "me", "rvol_21d", "dolvol_126d"
    ]

    # Load the data
    data = pd.read_csv(
        os.path.join(data_path, "usa.csv"),
        usecols=list(set(ids + features_all)),
        dtype={"eom": str, "sic": str}
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