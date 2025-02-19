import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from collections import defaultdict
import seaborn as sns
from matplotlib.cm import get_cmap


# -------------------- Output Dictionary --------------------
output = {}

# -------------------- Layout Settings --------------------
# Define the color theme, combining manual colors and a predefined Matplotlib colormap
colormap = get_cmap("Dark2")  # Retrieve the colormap
colours_theme = [
    "#0C6291", "#A63446",  # Custom colors
    *[colormap(i) for i in range(colormap.N)],  # Add colors from "Dark2" colormap
    "darkslategrey", "blue", "red", "purple",
    "yellow", "aquamarine", "grey", "salmon",
    "antiquewhite", "chartreuse"
]

pf_order = [
    "Portfolio-ML", "Multiperiod-ML", "Multiperiod-ML*",
    "Static-ML", "Static-ML*", "Market", "1/N",
    "Minimum Variance", "Factor-ML", "Rank-ML", "Markowitz-ML"
]

pf_order_new = [
    "Portfolio-ML", "Multiperiod-ML*", "Static-ML*",
    "Multiperiod-ML", "Static-ML", "Markowitz-ML",
    "Rank-ML", "Factor-ML", "Minimum Variance", "1/N", "Market"
]

main_types = [
    "Portfolio-ML", "Multiperiod-ML*", "Static-ML*",
    "Factor-ML", "Markowitz-ML"
]

cluster_order = [
    "Accruals", "Debt Issuance", "Investment",
    "Short-Term Reversal", "Value", "Low Risk",
    "Quality", "Momentum", "Profitability", "Profit Growth",
    "Seasonality", "Size", "Low Leverage"
]

txt_size = 10

# -------------------- Figure Paths --------------------
output_path_fig = "Figures"
format = "pdf"
fig_h = 3.2
fig_w = 6.5

# -------------------- Setup --------------------
# Data settings dictionary
settings = {
    "data_path": "C:\Master",
    "parallel": True,
    "seed_no": 1,
    "months": False,
    "split": {
        "train_end": pd.Timestamp("2015-12-31"), # obs
        "test_end": pd.Timestamp("2020-12-31"),
        "val_years": 2,
        "model_update_freq": "yearly",  # Options: "once", "yearly", "decade"
        "train_lookback": 1,
        "retrain_lookback": 1
    },
    "feat_prank": True,
    "ret_impute": "zero",  # Impute missing returns
    "feat_impute": True,
    "addition_n": 12,  # Need to be valid for the past n months to be included in investment universe
    "deletion_n": 12,  # Exclude from universe after N periods where the stock is non-valid
    "screens": {
        "start": pd.Timestamp("2010-01-01"),
        "end": pd.Timestamp("2020-12-31"),
        "feat_pct": 0.5,
        "nyse_stocks": True
    },
    "pi": 0.1,
    "rff": {
        "p_vec": [2 ** i for i in range(1, 3)], #fix
        "g_vec": np.exp(np.arange(-3, -1)),
        "l_vec": np.concatenate(([0], np.exp(np.linspace(-10, 10, 100))))
    },
    "pf": {
        "dates": {
            "start_year": 2015,
            "end_yr": 2020,
            "split_years": 4
        },
        "hps": {
            "cov_type": "cov_add",
            "m1": {
                "k": [1, 2, 3],
                "u": [0.25, 0.5, 1],
                "g": [0, 1, 2],
                "K": 12
            },
            "static": {
                "k": [1 / 1, 1 / 3, 1 / 5],
                "u": [0.25, 0.5, 1],
                "g": [0, 1, 2]
            }
        }
    },
    "pf_ml": {
        "g_vec": np.exp(np.arange(-3, -1)),
        "p_vec": [2**i for i in range(6, 10)],
        "l_vec": np.concatenate(([0], np.exp(np.linspace(-10, 10, 100)))),
        "orig_feat": False,
        "scale": True
    },
    "ef": {
        "wealth": [1, 1e9, 1e10, 1e11],
        "gamma_rel": [1, 5, 10, 20, 100]
    },
    "cov_set": {
        "industries": True,
        "obs": 252 * 2, # obs
        "hl_cor": 252 * 3 / 2,
        "hl_var": 252 / 2,
        "hl_stock_var": 252 / 2,
        "min_stock_obs": 252,
        "initial_var_obs": 21 * 3
    },
    "factor_ml": {
        "n_pfs": 10
    }
}


# Set random seed
np.random.seed(settings["seed_no"])

# -------------------- Portfolio Settings --------------------
pf_set = {
    "wealth": 1e10,
    "gamma_rel": 10,
    "mu": 0.007,  # Expected monthly portfolio growth
    "lb_hor": 11
}

# -------------------- Features --------------------
# Full feature list
features = [
    "age", "aliq_at", "aliq_mat", "ami_126d", "at_be", "at_gr1", "at_me", "at_turnover",
    "be_gr1a", "be_me", "beta_60m", "beta_dimson_21d", "betabab_1260d", "betadown_252d",
    "bev_mev", "bidaskhl_21d", "capex_abn", "capx_gr1", "capx_gr2", "capx_gr3",
    "cash_at", "chcsho_12m", "coa_gr1a", "col_gr1a", "cop_at", "cop_atl1", "corr_1260d",
    "coskew_21d", "cowc_gr1a", "dbnetis_at", "debt_gr3", "debt_me", "dgp_dsale",
    "div12m_me", "dolvol_126d", "dolvol_var_126d", "dsale_dinv", "dsale_drec", "dsale_dsga",
    "earnings_variability", "ebit_bev", "ebit_sale", "ebitda_mev", "emp_gr1", "eq_dur",
    "eqnetis_at", "eqnpo_12m", "eqnpo_me", "eqpo_me", "f_score", "fcf_me", "fnl_gr1a",
    "gp_at", "gp_atl1", "ival_me", "inv_gr1", "inv_gr1a", "iskew_capm_21d", "iskew_ff3_21d",
    "iskew_hxz4_21d", "ivol_capm_21d", "ivol_capm_252d", "ivol_ff3_21d", "ivol_hxz4_21d",
    "kz_index", "lnoa_gr1a", "lti_gr1a", "market_equity", "mispricing_mgmt", "mispricing_perf",
    "ncoa_gr1a", "ncol_gr1a", "netdebt_me", "netis_at", "nfna_gr1a", "ni_ar1", "ni_be",
    "ni_inc8q", "ni_ivol", "ni_me", "niq_at", "niq_at_chg1", "niq_be", "niq_be_chg1", "niq_su",
    "nncoa_gr1a", "noa_at", "noa_gr1a", "o_score", "oaccruals_at", "oaccruals_ni", "ocf_at",
    "ocf_at_chg1", "ocf_me", "ocfq_saleq_std", "op_at", "op_atl1", "ope_be", "ope_bel1",
    "opex_at", "pi_nix", "ppeinv_gr1a", "prc", "prc_highprc_252d", "qmj", "qmj_growth",
    "qmj_prof", "qmj_safety", "rd_me", "rd_sale", "rd5_at", "resff3_12_1", "resff3_6_1",
    "ret_1_0", "ret_12_1", "ret_12_7", "ret_3_1", "ret_6_1", "ret_60_12", "ret_9_1",
    "rmax1_21d", "rmax5_21d", "rmax5_rvol_21d", "rskew_21d", "rvol_21d", "sale_bev",
    "sale_emp_gr1", "sale_gr1", "sale_gr3", "sale_me", "saleq_gr1", "saleq_su",
    "seas_1_1an", "seas_1_1na", "seas_11_15an", "seas_11_15na", "seas_16_20an",
    "seas_16_20na", "seas_2_5an", "seas_2_5na", "seas_6_10an", "seas_6_10na", "sti_gr1a",
    "taccruals_at", "taccruals_ni", "tangibility", "tax_gr1a", "turnover_126d",
    "turnover_var_126d", "z_score", "zero_trades_126d", "zero_trades_21d",
    "zero_trades_252d", "rvol_252d"
]

# Excluded features
feat_excl = [
    "capex_abn", "capx_gr2", "capx_gr3", "debt_gr3", "dgp_dsale", "dsale_dinv",
    "dsale_drec", "dsale_dsga", "earnings_variability", "eqnetis_at", "eqnpo_me",
    "eqpo_me", "f_score", "iskew_hxz4_21d", "ivol_hxz4_21d", "netis_at", "ni_ar1",
    "ni_inc8q", "ni_ivol", "niq_at", "niq_at_chg1", "niq_be", "niq_be_chg1", "niq_su",
    "ocfq_saleq_std", "qmj", "qmj_growth", "rd_me", "rd_sale", "rd5_at", "resff3_12_1",
    "resff3_6_1", "sale_gr3", "saleq_gr1", "saleq_su", "seas_16_20an", "seas_16_20na",
    "sti_gr1a", "z_score"
]

# Exclude features with insufficient coverage
features = [feat for feat in features if feat not in feat_excl]

print("Setup complete!")