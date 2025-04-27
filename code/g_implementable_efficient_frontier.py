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
    grouped = []

    for vol_target, group in df.groupby('vol_target'):
        r = group['r']
        tc = group['tc']
        r_net = r - tc
        sd_r = r.std()
        obj = (r.mean() - 0.5 * r.var() * gamma_rel - tc.mean()) * 12

        row = {
            'vol_target': vol_target,
            'n': len(group),
            'inv': group['inv'].mean(),
            'to': group['turnover'].mean(),
            'r': r.mean() * 12,
            'sd': sd_r * np.sqrt(12),
            'sr_gross': r.mean() / sd_r * np.sqrt(12),
            'tc': tc.mean() * 12,
            'r_tc': r_net.mean() * 12,
            'sr': r_net.mean() / sd_r * np.sqrt(12),
            'obj': obj
        }

        grouped.append(row)

    return pd.DataFrame(grouped)


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
def get_ief_portfolios(output_path, settings):
    """
    Reads IEF portfolio data from disk, supporting both single and multi-gamma folder structures.

    Parameters:
        output_path (str): Root path to IEF folders.
        settings (dict): Must contain 'ief_multi_gamma' (bool).

    Returns:
        pd.DataFrame: Combined portfolio performance data.
    """

    all_portfolios = []

    if settings.get("ief_multi_gamma", False):
        # MULTI-GAMMA mode
        for folder in os.listdir(output_path):
            folder_path = os.path.join(output_path, folder)
            if not os.path.isdir(folder_path) or not folder.startswith("gamma_"):
                continue

            try:
                gamma_val = float(folder.split("_")[1])
            except (IndexError, ValueError):
                continue

            # Read core files
            pfml_cf = pd.read_csv(os.path.join(folder_path, "pfml_cf_ief.csv"))
            bms = pd.read_csv(os.path.join(folder_path, "bms.csv"))
            static_ml = pd.read_pickle(os.path.join(folder_path, "static-ml.pkl"))
            portfolio_ml = pd.read_pickle(os.path.join(folder_path, "portfolio-ml.pkl"))

            # Clean/label
            bms['eom_ret'] = pd.to_datetime(bms['eom_ret'])
            for df in [pfml_cf, bms, static_ml, portfolio_ml]:
                df['gamma_rel'] = gamma_val

            # Merge and tag
            all_portfolios.extend([pfml_cf, bms, static_ml, portfolio_ml])

    else:
        # SINGLE-GAMMA mode
        demo_path = os.path.join(output_path, "demo")
        pfml_cf = pd.read_csv(os.path.join(demo_path, "pfml_cf_ief.csv"))
        bms = pd.read_csv(os.path.join(demo_path, "bms.csv"))
        static_ml = pd.read_pickle(os.path.join(demo_path, "static-ml.pkl"))
        portfolio_ml = pd.read_pickle(os.path.join(demo_path, "portfolio-ml.pkl"))

        bms['eom_ret'] = pd.to_datetime(bms['eom_ret'])
        all_portfolios.extend([pfml_cf, bms, static_ml, portfolio_ml])

    # Combine everything
    ief_ss = pd.concat(all_portfolios, ignore_index=True)

    # Check duplicates
    duplicate_check = ief_ss.groupby(['wealth_end', 'gamma_rel', 'type', 'eom_ret']).size()
    if (duplicate_check > 1).any():
        raise ValueError("Found duplicates in the IEF portfolios!")

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
def run_ief(chars, dates_oos, pf_set, wealth, barra_cov, market_data, risk_free, settings, output_path):
    # Generate Factor-ML portfolio
    # factor_ss = generate_factor_ml_volatility(chars, dates_oos, pf_set, wealth)

    # Generate Markowitz-ML portfolio
    # tpf_ss = generate_markowitz_ml_volatility(chars, barra_cov, wealth, dates_oos, pf_set)

    # Generate Mean-Variance Risky Portfolio
    u_vec = np.concatenate([np.arange(-0.5, 0.5, 0.05), [0.6, 0.75, 1, 2]]) / 12
    wealth_0 = wealth_func(wealth_end=0, end=settings['split']['test_end'], market=market_data, risk_free=risk_free)
    mv_ss = generate_mv_risky_ef(chars, barra_cov, wealth_0, dates_oos, pf_set, u_vec)

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