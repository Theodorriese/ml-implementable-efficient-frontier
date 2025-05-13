import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from a_portfolio_choice_functions import (tpf_implement, factor_ml_implement, mv_risky_fun)
from a_general_functions import create_cov, pf_ts_fun
from b_prepare_data import wealth_func
from i1_Main import settings, pf_set


def summarize_factor_ef(df, gamma_rel):
    """
    Summarize factor efficient frontier portfolios.

    Parameters:
        df (pd.DataFrame): Concatenated DataFrame from different vol targets.
        gamma_rel (float): Risk aversion parameter.

    Returns:
        pd.DataFrame: Aggregated summary statistics by volatility level.
    """
    summaries = []

    for vol_target, group in df.groupby('vol_target'):
        r = group['r']
        tc = group['tc']
        inv = group['inv'].mean()
        to = group['turnover'].mean()

        r_mean = r.mean()
        r_var = r.var()
        r_std = r.std()
        tc_mean = tc.mean()
        r_net = r - tc
        r_net_mean = r_net.mean()

        # Avoid divide-by-zero
        sr_gross = (r_mean / r_std * np.sqrt(12)) if r_std != 0 else np.nan
        sr_net = (r_net_mean / r_std * np.sqrt(12)) if r_std != 0 else np.nan

        obj = (r_mean - 0.5 * r_var * gamma_rel - tc_mean) * 12

        summaries.append({
            'vol_target': vol_target,
            'n': len(group),
            'inv': inv,
            'to': to,
            'r': r_mean * 12,
            'sd': r_std * np.sqrt(12),
            'sr_gross': sr_gross,
            'tc': tc_mean * 12,
            'r_tc': r_net_mean * 12,
            'sr': sr_net,
            'obj': obj
        })

    return pd.DataFrame(summaries)


# 1) Generate Factor-ML with different volatility levels
def generate_factor_ml_volatility(chars, dates_oos, pf_set, wealth):
    """
    Generate Factor-ML portfolios with different volatility levels.

    Returns:
        pd.DataFrame: Summary statistics for Factor-ML portfolios with different volatility levels.
    """
    vol_range = np.arange(0, 0.51, 0.01)

    # Base portfolio
    factor_base = factor_ml_implement(chars, wealth, dates_oos, settings['factor_ml']['n_pfs'])
    factor_base_vol = factor_base['pf']['r'].std() * np.sqrt(12)

    # Portfolios across volatility levels
    factor_ef = []
    for vol_target in vol_range:
        scale = vol_target / factor_base_vol
        scaled_factors = factor_base['w'].copy()
        scaled_factors['w'] *= scale
        scaled_factors['w_start'] *= scale

        pf = pf_ts_fun(scaled_factors, chars, wealth)
        pf['vol_target'] = vol_target
        factor_ef.append(pf)

    factor_ef_df = pd.concat(factor_ef, ignore_index=True)
    factor_ef_df['type'] = 'Factor-ML'

    factor_ss = summarize_factor_ef(factor_ef_df, pf_set['gamma_rel'])

    return factor_ss


# 2) Generate Markowitz-ML with different volatility levels
def generate_markowitz_ml_volatility(chars, barra_cov, wealth, dates_oos, pf_set):
    """
    Generate Markowitz-ML portfolios with different volatility levels.

    Returns:
        pd.DataFrame: Summary statistics for Markowitz-ML portfolios with different volatility levels.
    """
    vol_range = np.arange(0, 0.51, 0.01)

    # Base portfolio
    tpf_base = tpf_implement(chars, barra_cov, wealth, dates_oos, pf_set['gamma_rel'])
    tpf_base_vol = tpf_base['pf']['r'].std() * np.sqrt(12)

    # Portfolios across volatility levels
    tpf_ef = []
    for vol_target in vol_range:
        scale = vol_target / tpf_base_vol
        scaled_tpf = tpf_base['w'].copy()
        scaled_tpf['w'] *= scale
        scaled_tpf['w_start'] *= scale

        pf = pf_ts_fun(scaled_tpf, chars, wealth)
        pf['vol_target'] = vol_target
        tpf_ef.append(pf)

    tpf_ef_df = pd.concat(tpf_ef, ignore_index=True)
    tpf_ef_df['type'] = 'Markowitz-ML'

    tpf_ss = summarize_factor_ef(tpf_ef_df, pf_set['gamma_rel'])

    return tpf_ss


# 3) Generate Mean-Variance Efficient Frontier
def generate_mv_risky_ef(chars, barra_cov, wealth_0, dates_oos, pf_set, u_vec):
    """
    Generate mean-variance efficient frontier of risky assets.

    Returns:
        pd.DataFrame: Summary statistics for each utility level.
    """
    mv_risky_ef_df = mv_risky_fun(
        data=chars,
        cov_list=barra_cov,
        wealth=wealth_0,
        dates=dates_oos,
        gam=pf_set['gamma_rel'],
        u_vec=u_vec
    )

    grouped = []

    for u, group in mv_risky_ef_df.groupby('u'):
        r = group['r']
        tc = group['tc']
        r_net = r - tc
        sd_r = r.std()
        obj = (r.mean() - 0.5 * r.var() * pf_set['gamma_rel'] - tc.mean()) * 12

        row = {
            'u': u,
            'n': len(group),
            'inv': group['inv'].mean(),
            'to': group['turnover'].mean(),
            'r': r.mean() * 12,
            'sd': sd_r * np.sqrt(12),
            'sr_gross': r.mean() / sd_r * np.sqrt(12) if sd_r != 0 else np.nan,
            'tc': tc.mean() * 12,
            'r_tc': r_net.mean() * 12,
            'sr': r_net.mean() / sd_r * np.sqrt(12) if sd_r != 0 else np.nan,
            'obj': obj
        }

        grouped.append(row)

    return pd.DataFrame(grouped)


# 3) Get portfolios
def load_ief_portfolio_folder(folder_path, gamma_val):
    """
    Loads portfolio performance data from a specified IEF (Incremental Efficiency Frontier) folder,
    and filters the Static-ML hyperparameter results for a specific configuration.

    Parameters:
        folder_path (str): Path to the folder containing IEF result files.
        gamma_val (float): The risk aversion parameter to annotate the portfolios with.

    Returns:
        list: A list of DataFrames containing:
              - Benchmark returns (bms),
              - Static-ML portfolio results,
              - Portfolio-ML results,
              - Filtered Static-ML hyperparameter statistics.
    """

    bms = pd.read_csv(os.path.join(folder_path, "bms.csv"))
    bms["eom_ret"] = pd.to_datetime(bms["eom_ret"])

    static_ml = pd.read_pickle(os.path.join(folder_path, "static-ml.pkl"))
    portfolio_ml = pd.read_pickle(os.path.join(folder_path, "portfolio-ml.pkl"))

    # Extract data
    static_pf = static_ml["pf"]
    pf_dates = static_pf["eom_ret"].unique()

    hps_filtered = static_ml["hps"][
        static_ml["hps"]["eom_ret"].isin(pf_dates) &
        (static_ml["hps"]["k"] == 1) &
        (static_ml["hps"]["g"] == 0) &
        (static_ml["hps"]["u"] == 1)
        ][["eom_ret", "inv", "shorting", "turnover", "r", "tc"]].copy()
    hps_filtered["type"] = "Static-ML"

    # Always add gamma_rel
    for df in [bms, static_pf, portfolio_ml["pf"], hps_filtered]:
        df["gamma_rel"] = gamma_val

    return [bms, static_pf, portfolio_ml["pf"], hps_filtered]


def get_ief_portfolios(output_path, settings, pf_set):
    """
    Loads and aggregates Incremental Efficiency Frontier (IEF) portfolio results from disk,
    supporting both single and multiple gamma configurations.

    Parameters:
        output_path (str): Directory path containing IEF results (can include gamma-specific subfolders).
        settings (dict): Dictionary of settings, including the 'ief_multi_gamma' flag.
        pf_set (dict): Portfolio metadata including 'gamma_rel' (used in single-gamma mode).

    Returns:
        tuple:
            - pd.DataFrame: Combined portfolio-level data from all sources.
            - pd.DataFrame: Aggregated summary statistics across portfolio types and gamma values.
    """

    all_portfolios = []

    # Default is multi-gamma if key is missing
    if settings.get("ief_multi_gamma", True):
        for folder in os.listdir(output_path):
            folder_path = os.path.join(output_path, folder)

            if not os.path.isdir(folder_path):
                continue

            if folder.startswith("gamma_"):
                try:
                    gamma_val = float(folder.split("_")[1])
                    all_portfolios.extend(load_ief_portfolio_folder(folder_path, gamma_val))
                except (IndexError, ValueError):
                    print(f"Skipping malformed folder name: {folder}")
                    continue
    else:
        gamma_val = pf_set["gamma_rel"]
        all_portfolios.extend(load_ief_portfolio_folder(output_path, gamma_val))

    # Combine all
    ief_ss = pd.concat(all_portfolios, ignore_index=True)

    # Duplicate check
    duplicate_check = ief_ss.groupby(['type', 'gamma_rel', 'eom_ret']).size()
    if (duplicate_check > 1).any():
        raise ValueError("Found duplicates in the IEF portfolios!")

    # Grouping
    group_cols = ["type", "gamma_rel"]

    # Aggregation
    summary_base = (
        ief_ss.groupby(group_cols)
        .agg(
            inv_mean=("inv", "mean"),
            to_mean=("turnover", "mean"),
            r_mean=("r", "mean"),
            r_std=("r", "std"),
            tc_mean=("tc", "mean"),
        )
        .reset_index()
    )

    # Derived metrics
    summary_base["r_annual"] = summary_base["r_mean"] * 12
    summary_base["sd_annual"] = summary_base["r_std"] * (12 ** 0.5)
    summary_base["sr_gross"] = summary_base["r_mean"] / summary_base["r_std"] * (12 ** 0.5)
    summary_base["tc_annual"] = summary_base["tc_mean"] * 12
    summary_base["r_tc_annual"] = (summary_base["r_mean"] - summary_base["tc_mean"]) * 12
    summary_base["sr_net"] = (summary_base["r_mean"] - summary_base["tc_mean"]) / summary_base["r_std"] * (12 ** 0.5)

    # Objective
    summary_base["obj"] = (
            (summary_base["r_mean"]
             - 0.5 * (summary_base["r_std"] ** 2) * summary_base["gamma_rel"]
             - summary_base["tc_mean"]) * 12
    )

    return ief_ss, summary_base


def build_ief_input(factor_summary, tpf_summary, ief_result, pf_set):
    """
    Combines IEF summary statistics with those from Factor-ML and Markowitz-ML
    for unified analysis or plotting.

    Parameters:
        factor_summary (pd.DataFrame): Summary stats for the Factor-ML strategy.
        tpf_summary (pd.DataFrame): Summary stats for the Markowitz-ML strategy.
        ief_result (tuple): Tuple containing (full_ief_data, ief_summary).
        pf_set (dict): Dictionary with configuration info (not directly used here, kept for extensibility).

    Returns:
        pd.DataFrame: Combined summary statistics for all strategies.
    """

    _, ief_summary = ief_result

    # Filter only relevant IEF strategies
    ief_filtered = ief_summary[
        ief_summary['type'].isin(['Portfolio-ML', 'Static-ML*', 'Static-ML'])
    ].copy()

    # Add metadata to match structure
    factor_summary = factor_summary.copy()
    factor_summary['type'] = 'Factor-ML'
    factor_summary['gamma_rel'] = np.nan

    tpf_summary = tpf_summary.copy()
    tpf_summary['type'] = 'Markowitz-ML'
    tpf_summary['gamma_rel'] = np.nan

    # Combine all into long-form summary DataFrame
    ef_all_ss = pd.concat([ief_filtered, factor_summary, tpf_summary], ignore_index=True, sort=False)

    return ef_all_ss


def get_indifference_points(ef_all_ss):
    """
    Extracts only the IEF strategies (Portfolio-ML, Static-ML*, Markowitz-ML)
    across all gamma values.
    """
    return ef_all_ss[
        ef_all_ss['type'].isin(['Portfolio-ML', 'Static-ML*', 'Markowitz-ML'])
    ].copy()



def create_indifference_curves(points):
    """
    Generates indifference curves for a given set of utility levels and gamma values per row.

    Parameters:
        points (pd.DataFrame): DataFrame containing 'obj' (target utility) and 'gamma_rel' per row.

    Returns:
        pd.DataFrame: Combined DataFrame of indifference curves with columns ['sd', 'r_tc', 'u', 'gamma_rel'].
    """
    curves = []
    for _, row in points.iterrows():
        u_target = row['obj']
        gamma = row['gamma_rel']
        vol_space = np.linspace(0, 0.4, 41)
        r_tc = u_target + 0.5 * gamma * vol_space ** 2
        curve = pd.DataFrame({
            'sd': vol_space,
            'r_tc': r_tc,
            'u': u_target,
            'gamma_rel': gamma
        })
        curves.append(curve)
    return pd.concat(curves, ignore_index=True)



# 7) Plot indifference curves
def plot_indifference_curves(ax, indifference_curves, ef_all_ss, tpf_ss, factor_ss, output_path):
    """
    Plots indifference curves on a given axis and saves the figure to disk.

    Parameters:
        ax (matplotlib.axes.Axes): Matplotlib axis to plot on.
        indifference_curves (pd.DataFrame): DataFrame with indifference curves (columns: 'sd', 'r_tc', 'u').
        ef_all_ss (pd.DataFrame): Combined summary stats (not used in this plot, included for extensibility).
        tpf_ss (pd.DataFrame): Markowitz-ML summary stats (not used in this plot, included for extensibility).
        factor_ss (pd.DataFrame): Factor-ML summary stats (not used in this plot, included for extensibility).
        output_path (str): Directory path to save the plot.

    Returns:
        None
    """

    for u_val, curve in indifference_curves.groupby('u'):
        ax.plot(curve['sd'], curve['r_tc'], linestyle='dashed', alpha=0.40)

    ax.set_xlabel("Volatility")
    ax.set_ylabel("Excess returns (net of trading cost)")
    ax.set_title("Indifference Curves")

    # Save figure
    fig = ax.get_figure()
    save_path = os.path.join(output_path, "indifference_curves.png")
    fig.savefig(save_path, bbox_inches='tight')
    print(f"Indifference curve plot saved to: {save_path}")


def plot_figure_1A(ef_all_ss, mv_ss, indifference_curves, points, output_path):
    """
    Plots Figure 1A: Implementable Efficient Frontier (IEF) with indifference curves and strategy points.

    Parameters:
        ef_all_ss (pd.DataFrame): Combined summary statistics for all strategies.
        mv_ss (pd.DataFrame): Markowitz summary statistics (not used directly, included for extensibility).
        indifference_curves (pd.DataFrame): Indifference curves generated for specific utility levels.
        points (pd.DataFrame): Key strategies and their summary points (e.g., Static-ML, Portfolio-ML).
        output_path (str): Directory where the plot will be saved.

    Returns:
        None: Saves the figure to disk as 'figure_1A.png'.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    # 1. Gross tangency line (Markowitz-ML gross)
    tpf_df = ef_all_ss[ef_all_ss['type'] == 'Markowitz-ML'].dropna(subset=['sr_gross'])
    tpf_slope = tpf_df['sr_gross'].iloc[0]  # You can also use .median() if you prefer
    x_vals = np.linspace(0, 0.4, 100)
    ax.plot(x_vals, tpf_slope * x_vals, label="Markowitz-ML (gross)", linestyle='-', color='gray')

    # 2. Plot full frontier curves
    for strategy in ['Factor-ML', 'Markowitz-ML']:
        df = ef_all_ss[ef_all_ss['type'] == strategy].dropna(subset=['sd', 'r_tc'])
        ax.plot(df['sd'], df['r_tc'], label=strategy)

    # 3. Plot points for fixed strategies
    fixed_strategies = ['Portfolio-ML', 'Static-ML*', 'Static-ML']
    markers = {'Portfolio-ML': 'o', 'Static-ML*': '^', 'Static-ML': 's'}
    colors = {'Portfolio-ML': 'tab:blue', 'Static-ML*': 'tab:green', 'Static-ML': 'black'}

    for strategy in fixed_strategies:
        row = points[points['type'] == strategy]
        if not row.empty:
            ax.scatter(row['sd_annual'], row['r_tc_annual'],
                       marker=markers[strategy],
                       color=colors[strategy],
                       label=strategy)

    # 4. Annotate Static-ML specifically
    static_row = points[points['type'] == 'Static-ML']
    if not static_row.empty:
        sr = static_row.iloc[0]
        ax.annotate("Static-ML (one layer)", xy=(sr['sd'], sr['r_tc']),
                    xytext=(sr['sd'], 0.05), ha='center',
                    arrowprops=dict(arrowstyle='->', lw=0.5))

    # 4b. Draw lines through Static-ML and Portfolio-ML across gammas
    try:
        strat_points = points.set_index(['type', 'gamma_rel'])

        for strat in ['Static-ML*', 'Portfolio-ML']:
            gammas = sorted(points[points['type'] == strat]['gamma_rel'].unique())
            if len(gammas) >= 2:
                g1, g2 = gammas[1], gammas[0]
                p1 = strat_points.loc[(strat, g1), ['sd_annual', 'r_tc_annual']].values.ravel()
                p2 = strat_points.loc[(strat, g2), ['sd_annual', 'r_tc_annual']].values.ravel()

                # Line from origin to first point
                ax.plot([0, p1[0]], [0, p1[1]], color='black', linestyle='-', alpha=0.7)

                # Line from first point to second point
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black', linestyle='-', alpha=0.7)

    except Exception as e:
        print(f"Could not draw strategy lines across gammas: {e}")

    # 5. Plot indifference curves
    for u_val in points['obj']:
        curve = indifference_curves[indifference_curves['u'] == u_val]
        ax.plot(curve['sd'], curve['r_tc'], linestyle='dashed', color='gray', alpha=0.4)

    # Labels and formatting
    ax.set_xlim(0, 0.4)
    ax.set_ylim(-0.5, 0.8)
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Excess returns (net of trading cost)")
    ax.set_title("Implementable Efficient Frontier")
    ax.legend()

    save_path = os.path.join(output_path, "figure_1A.png")
    fig.savefig(save_path, bbox_inches='tight')
    print(f"Figure 1A saved to: {save_path}")


# Main execution function
def run_ief(chars, dates_oos, pf_set, wealth, barra_cov, market_data, risk_free, settings, latest_folder, output_path):
    # Generate Factor-ML portfolio
    factor_ss = generate_factor_ml_volatility(chars, dates_oos, pf_set, wealth)

    # Generate Markowitz-ML portfolio
    tpf_ss = generate_markowitz_ml_volatility(chars, barra_cov, wealth, dates_oos, pf_set)

    # Generate Mean-Variance Risky Portfolio
    u_vec = np.concatenate([np.arange(-0.5, 0.5, 0.05), [0.6, 0.75, 1, 2]]) / 12
    wealth_0 = wealth_func(wealth_end=0, end=settings['split']['test_end'], market=market_data, risk_free=risk_free)
    mv_ss = generate_mv_risky_ef(chars, barra_cov, wealth_0, dates_oos, pf_set, u_vec)

    # Save summary statistics to disk
    factor_ss.to_pickle(os.path.join(latest_folder, "factor_ss.pkl"))
    tpf_ss.to_pickle(os.path.join(latest_folder, "tpf_ss.pkl"))
    mv_ss.to_pickle(os.path.join(latest_folder, "mv_ss.pkl"))

    # factor_ss = pd.read_pickle(os.path.join(latest_folder, "factor_ss.pkl"))
    # tpf_ss = pd.read_pickle(os.path.join(latest_folder, "tpf_ss.pkl"))
    # mv_ss = pd.read_pickle(os.path.join(latest_folder, "mv_ss.pkl"))

    ief_result = get_ief_portfolios(latest_folder, settings, pf_set)
    ef_all_ss = build_ief_input(factor_ss, tpf_ss, ief_result, pf_set)

    points = get_indifference_points(ef_all_ss)
    indifference_curves = create_indifference_curves(points)

    # Plot 1A
    plot_figure_1A(ef_all_ss, mv_ss, indifference_curves, points, output_path)

    # Extract subset for IEF "Portfolio-ML" only, for later use
    ief_summary = ief_result[1]
    ef_ss = ief_summary[ief_summary['type'] == 'Portfolio-ML'].copy()

    return ef_all_ss, ef_ss