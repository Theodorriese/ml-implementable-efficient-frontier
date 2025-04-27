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


# 1) Generate Factor-ML with different volatility levels
def generate_factor_ml_volatility(chars, dates_oos, pf_set, wealth):
    """
    Generate Factor-ML portfolios with different volatility levels.

    Parameters:
        chars (pd.DataFrame): The dataset containing characteristics of assets.
        dates_oos (list): List of out-of-sample dates.
        pf_set (dict): Portfolio settings, including factor_ml configurations.
        wealth (pd.DataFrame): Wealth data for the portfolio.

    Returns:
        pd.DataFrame: Summary statistics for Factor-ML portfolios with different volatility levels.
    """
    vol_range = np.arange(0, 0.51, 0.01)

    # Generate Factor-ML base portfolio
    factor_base = factor_ml_implement(chars, wealth, dates_oos, pf_set['factor_ml']['n_pfs'])

    # Calculate base portfolio volatility
    factor_base_vol = factor_base['pf']['r'].std() * np.sqrt(12)

    # Generate Factor-ML portfolios with different volatility levels
    factor_ef = []
    for vol_target in vol_range:
        scale = vol_target / factor_base_vol
        scaled_factors = factor_base['w'] * scale
        factor_ef.append(pf_ts_fun(scaled_factors, chars, wealth))

    factor_ef_df = pd.concat(factor_ef, ignore_index=True)
    factor_ef_df['type'] = 'Factor-ML'

    # Summary statistics for Factor-ML portfolios
    factor_ss = factor_ef_df.groupby('vol_target').agg(
        n='size',
        inv='mean',
        to='mean',
        r='mean',
        sd='std',
        sr_gross='mean',
        tc='mean',
        r_tc='mean',
        sr='mean',
        obj='mean'
    )

    return factor_ss


# 2) Generate Markowitz-ML with different volatility levels
def generate_markowitz_ml_volatility(chars, barra_cov, wealth, dates_oos, pf_set):
    vol_range = np.arange(0, 0.51, 0.01)

    tpf_base = tpf_implement(chars, barra_cov, wealth, dates_oos, pf_set['gamma_rel'])
    tpf_base_vol = tpf_base['pf']['r'].std() * np.sqrt(12)

    tpf_ef = []
    for vol_target in vol_range:
        scale = vol_target / tpf_base_vol
        scaled_tpf = tpf_base['w'] * scale
        tpf_ef.append(pf_ts_fun(scaled_tpf, chars, wealth))

    tpf_ef_df = pd.concat(tpf_ef, ignore_index=True)
    tpf_ef_df['type'] = 'Markowitz-ML'

    # Summary statistics for Markowitz-ML
    tpf_ss = tpf_ef_df.groupby('vol_target').agg(
        n='size',
        inv='mean',
        to='mean',
        r='mean',
        sd='std',
        sr_gross='mean',
        tc='mean',
        r_tc='mean',
        sr='mean',
        obj='mean'
    )

    return tpf_ss


# 3) Generate Mean-Variance Efficient Frontier
def generate_mv_risky_ef(barra_cov, wealth, dates_oos, pf_set, u_vec):
    mv_risky_ef = mv_risky_fun(barra_cov, wealth, dates_oos, pf_set['gamma_rel'], u_vec)

    mv_risky_ef_df = mv_risky_ef['pf']

    mv_ss = mv_risky_ef_df.groupby('u').agg(
        n='size',
        inv='mean',
        to='mean',
        r='mean',
        sd='std',
        sr_gross='mean',
        tc='mean',
        r_tc='mean',
        sr='mean',
        obj='mean'
    )

    return mv_ss


def get_ief_portfolios(output_path):
    """
    Reads the IEF portfolio data from the given output_path and returns it as a DataFrame.

    Parameters:
        output_path (str): Path to the directory containing the portfolios.

    Returns:
        pd.DataFrame: The combined IEF portfolio data.
    """
    # Path to the folder containing the IEF portfolios
    ief_path = os.path.join(output_path, "Data/Generated/Portfolios/IEF/")

    ief_pfs = []

    # Read pfml_cf_ief.csv directly into DataFrame
    pfml_cf_ief_path = os.path.join(ief_path, "pfml_cf_ief.csv")
    pfml_cf_ief_df = pd.read_csv(pfml_cf_ief_path)

    # Iterate through each folder in the directory (we only have one gamma and wealth config)
    for folder in os.listdir(ief_path):
        folder_path = os.path.join(ief_path, folder)

        if os.path.isdir(folder_path):
            # Read files for each portfolio
            bms_path = os.path.join(folder_path, "bms.csv")
            static_ml_rds_path = os.path.join(folder_path, "static-ml.RDS")
            portfolio_ml_rds_path = os.path.join(folder_path, "portfolio-ml.RDS")

            # Read 'bms.csv' into DataFrame
            bms_df = pd.read_csv(bms_path)
            bms_df['eom_ret'] = pd.to_datetime(bms_df['eom_ret'])

            # Read .RDS files - We will assume that the static-ml and portfolio-ml RDS files have been converted to DataFrames.
            # For this example, assuming the data is available as pandas DataFrames:
            static_ml_df = pd.read_pickle(
                static_ml_rds_path)  # Placeholder, if static-ml.RDS has been converted to pickle
            portfolio_ml_df = pd.read_pickle(portfolio_ml_rds_path)  # Placeholder, same for portfolio-ml.RDS

            # Combine data from CSV and the two RDS converted DataFrames
            combined_df = pd.concat([bms_df, static_ml_df, portfolio_ml_df], ignore_index=True)

            # Assuming the type of the portfolio is Static-ML for the static ML data
            combined_df['type'] = "Static-ML"

            # Append the combined DataFrame to the list
            ief_pfs.append(combined_df)

    # Combine all DataFrames into a single DataFrame
    ief_pfs_df = pd.concat(ief_pfs, ignore_index=True)

    # Combine pfml_cf_ief.csv data with the rest of the portfolios (from the other CSVs and RDS files)
    ief_ss = pd.concat([pfml_cf_ief_df, ief_pfs_df], ignore_index=True)

    # Check for duplicates (in terms of wealth_end, gamma_rel, type, and eom_ret)
    duplicate_check = ief_ss.groupby(['wealth_end', 'gamma_rel', 'type', 'eom_ret']).size()
    if (duplicate_check != 1).any():
        raise ValueError("Found duplicates in the IEF portfolios data!")

    return ief_ss


# 5) Build inputs for IEF
def build_ief_input(factor_ss, tpf_ss, ief_ss, pf_set):
    ef_all_ss = pd.concat([
        ief_ss[ief_ss['type'].isin(['Portfolio-ML', 'Static-ML*', 'Static-ML'])],
        factor_ss.assign(gamma_rel=np.nan, type='Factor-ML', wealth_end=pf_set['wealth']),
        tpf_ss.assign(gamma_rel=np.nan, type='Markowitz-ML', wealth_end=pf_set['wealth'])
    ])

    return ef_all_ss


# 6) Indifference curve generation
def create_indifference_curve(u_target, sd_target):
    # Placeholder function to generate indifference curves
    vol_space = np.linspace(0, 0.4, 40)
    r_tc = u_target + (pf_set['gamma_rel'] / 2) * vol_space**2
    return pd.DataFrame({'sd': vol_space, 'r_tc': r_tc, 'u': u_target})


# 7) Plot indifference curves
def plot_indifference_curves(ax, indifference_curves, ef_all_ss, tpf_ss, factor_ss):
    for curve in indifference_curves:
        ax.plot(curve['sd'], curve['r_tc'], linestyle='dashed', alpha=0.40)
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Excess returns (net of trading cost)")
    ax.set_title("Indifference Curves")


# 8) Plot Figure 1B
def plot_figure_1B(comb_data):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(comb_data['sd'], comb_data['r_tc'])
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Excess returns")
    ax.set_title("Figure 1B: Portfolio Performance")


# Main execution function
def run_ief(chars, dates_oos, pf_set, wealth, barra_cov, market, risk_free, settings, output_path):
    # Generate Factor-ML portfolio
    factor_ss = generate_factor_ml_volatility(chars, dates_oos, pf_set, wealth)

    # Generate Markowitz-ML portfolio
    tpf_ss = generate_markowitz_ml_volatility(chars, barra_cov, wealth, dates_oos, pf_set)

    # Generate Mean-Variance Risky Portfolio
    u_vec = np.concatenate([np.arange(-0.5, 0.5, 0.05), [0.6, 0.75, 1, 2]]) / 12
    wealth_0 = wealth_func(wealth_end=0, end=settings['split']['test_end'], market=market, risk_free=risk_free)
    mv_ss = generate_mv_risky_ef(barra_cov, wealth_0, dates_oos, pf_set, u_vec)

    # Get IEF portfolios
    ief_ss = get_ief_portfolios(output_path)

    # Combine all summary stats
    ef_all_ss = build_ief_input(factor_ss, tpf_ss, ief_ss, pf_set)

    # Create Indifference Curves
    indifference_curves = []
    for i in range(len(ef_all_ss)):
        u_target = ef_all_ss.iloc[i]['obj']
        sd_target = ef_all_ss.iloc[i]['sd']
        indifference_curves.append(create_indifference_curve(u_target, sd_target))

    # Plot Figure 1A
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_indifference_curves(ax, indifference_curves, ef_all_ss, tpf_ss, factor_ss)
    plt.show()

    # Plot Figure 1B
    comb_data = pd.concat([
        ef_all_ss[ef_all_ss['type'] == 'tpf'],
        mv_ss.assign(wealth_end=1, type='mv')
    ])
    plot_figure_1B(comb_data)
    plt.show()