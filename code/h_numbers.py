# #%%
#
# import pandas as pd
#
# # Inspect columns in the CSV file
# data_path_JW = r"C:\Users\Wiingaard\OneDrive\CBS\0_Speciale FIN\Data"
# csv_file_path = f"{data_path_JW}\\usa.csv"
# df = pd.read_csv(csv_file_path, nrows=5)  # Load only the first few rows for inspection
# print(df.columns.tolist())  # Print the list of column names
#
#

#%%
import pandas as pd
import numpy as np
from b_prepare_data import preprocess_chars
#%%

# Mock inputs (replace with actual data)
data_path_JW = r"C:\Users\Wiingaard\OneDrive\CBS\0_Speciale FIN\Data"
features = ['obs_main', 'exch_main', 'common', 'primary_sec', 'permno', 'date', 'permco', 'gvkey', 'iid', 'id', 'excntry', 'eom', 'size_grp', 'me', 'me_company', 'ret_exc_lead1m', 'ret_exc', 'ret', 'ret_local', 'ret_lag_dif', 'prc_local', 'prc_high', 'prc_low', 'bidask', 'curcd', 'fx', 'gics', 'naics', 'sic', 'ff49', 'dolvol', 'shares', 'tvol', 'adjfct', 'comp_tpci', 'crsp_shrcd', 'comp_exchg', 'crsp_exchcd', 'source_crsp', 'prc', 'market_equity', 'div12m_me', 'chcsho_12m', 'eqnpo_12m', 'ret_1_0', 'ret_3_1', 'ret_6_1', 'ret_9_1', 'ret_12_1', 'ret_12_7', 'ret_60_12', 'seas_1_1an', 'seas_1_1na', 'seas_2_5an', 'seas_2_5na', 'seas_6_10an', 'seas_6_10na', 'seas_11_15an', 'seas_11_15na', 'seas_16_20an', 'seas_16_20na', 'at_gr1', 'sale_gr1', 'capx_gr1', 'inv_gr1', 'debt_gr3', 'sale_gr3', 'capx_gr3', 'inv_gr1a', 'lti_gr1a', 'sti_gr1a', 'coa_gr1a', 'col_gr1a', 'cowc_gr1a', 'ncoa_gr1a', 'ncol_gr1a', 'nncoa_gr1a', 'fnl_gr1a', 'nfna_gr1a', 'tax_gr1a', 'be_gr1a', 'ebit_sale', 'gp_at', 'cop_at', 'ope_be', 'ni_be', 'ebit_bev', 'netis_at', 'eqnetis_at', 'dbnetis_at', 'oaccruals_at', 'oaccruals_ni', 'taccruals_at', 'taccruals_ni', 'noa_at', 'opex_at', 'at_turnover', 'sale_bev', 'rd_sale', 'cash_at', 'sale_emp_gr1', 'emp_gr1', 'ni_inc8q', 'noa_gr1a', 'ppeinv_gr1a', 'lnoa_gr1a', 'capx_gr2', 'saleq_gr1', 'niq_be', 'niq_at', 'niq_be_chg1', 'niq_at_chg1', 'rd5_at', 'dsale_dinv', 'dsale_drec', 'dgp_dsale', 'dsale_dsga', 'saleq_su', 'niq_su', 'capex_abn', 'op_atl1', 'gp_atl1', 'ope_bel1', 'cop_atl1', 'pi_nix', 'ocf_at', 'op_at', 'ocf_at_chg1', 'at_be', 'ocfq_saleq_std', 'tangibility', 'earnings_variability', 'aliq_at', 'f_score', 'o_score', 'z_score', 'kz_index', 'ni_ar1', 'ni_ivol', 'at_me', 'be_me', 'debt_me', 'netdebt_me', 'sale_me', 'ni_me', 'ocf_me', 'fcf_me', 'eqpo_me', 'eqnpo_me', 'rd_me', 'ival_me', 'bev_mev', 'ebitda_mev', 'aliq_mat', 'eq_dur', 'beta_60m', 'resff3_12_1', 'resff3_6_1', 'mispricing_mgmt', 'mispricing_perf', 'ivol_capm_21d', 'iskew_capm_21d', 'coskew_21d', 'beta_dimson_21d', 'ivol_ff3_21d', 'iskew_ff3_21d', 'ivol_hxz4_21d', 'iskew_hxz4_21d', 'rmax5_21d', 'rmax1_21d', 'rvol_21d', 'rskew_21d', 'zero_trades_21d', 'dolvol_126d', 'dolvol_var_126d', 'turnover_126d', 'turnover_var_126d', 'zero_trades_126d', 'zero_trades_252d', 'ami_126d', 'ivol_capm_252d', 'prc_highprc_252d', 'betadown_252d', 'bidaskhl_21d', 'corr_1260d', 'betabab_1260d', 'rmax5_rvol_21d', 'age', 'qmj', 'qmj_prof', 'qmj_growth', 'qmj_safety', 'enterprise_value', 'book_equity', 'assets', 'sales', 'net_income', 'div1m_me', 'div3m_me', 'div6m_me', 'divspc1m_me', 'divspc12m_me', 'chcsho_1m', 'chcsho_3m', 'chcsho_6m', 'eqnpo_1m', 'eqnpo_3m', 'eqnpo_6m', 'ret_2_0', 'ret_3_0', 'ret_6_0', 'ret_9_0', 'ret_12_0', 'ret_18_1', 'ret_24_1', 'ret_24_12', 'ret_36_1', 'ret_36_12', 'ret_48_12', 'ret_48_1', 'ret_60_1', 'ret_60_36', 'ca_gr1', 'nca_gr1', 'lt_gr1', 'cl_gr1', 'ncl_gr1', 'be_gr1', 'pstk_gr1', 'debt_gr1', 'cogs_gr1', 'sga_gr1', 'opex_gr1', 'at_gr3', 'ca_gr3', 'nca_gr3', 'lt_gr3', 'cl_gr3', 'ncl_gr3', 'be_gr3', 'pstk_gr3', 'cogs_gr3', 'sga_gr3', 'opex_gr3', 'cash_gr1a', 'rec_gr1a', 'ppeg_gr1a', 'intan_gr1a', 'debtst_gr1a', 'ap_gr1a', 'txp_gr1a', 'debtlt_gr1a', 'txditc_gr1a', 'oa_gr1a', 'ol_gr1a', 'fna_gr1a', 'gp_gr1a', 'ebitda_gr1a', 'ebit_gr1a', 'ope_gr1a', 'ni_gr1a', 'nix_gr1a', 'dp_gr1a', 'fincf_gr1a', 'ocf_gr1a', 'fcf_gr1a', 'nwc_gr1a', 'eqnetis_gr1a', 'dltnetis_gr1a', 'dstnetis_gr1a', 'dbnetis_gr1a', 'netis_gr1a', 'eqnpo_gr1a', 'eqbb_gr1a', 'eqis_gr1a', 'div_gr1a', 'eqpo_gr1a', 'capx_gr1a', 'cash_gr3a', 'inv_gr3a', 'rec_gr3a', 'ppeg_gr3a', 'lti_gr3a', 'intan_gr3a', 'debtst_gr3a', 'ap_gr3a', 'txp_gr3a', 'debtlt_gr3a', 'txditc_gr3a', 'coa_gr3a', 'col_gr3a', 'cowc_gr3a', 'ncoa_gr3a', 'ncol_gr3a', 'nncoa_gr3a', 'oa_gr3a', 'ol_gr3a', 'fna_gr3a', 'fnl_gr3a', 'nfna_gr3a', 'gp_gr3a', 'ebitda_gr3a', 'ebit_gr3a', 'ope_gr3a', 'ni_gr3a', 'nix_gr3a', 'dp_gr3a', 'fincf_gr3a', 'ocf_gr3a', 'fcf_gr3a', 'nwc_gr3a', 'eqnetis_gr3a', 'dltnetis_gr3a', 'dstnetis_gr3a', 'dbnetis_gr3a', 'netis_gr3a', 'eqnpo_gr3a', 'tax_gr3a', 'eqbb_gr3a', 'eqis_gr3a', 'div_gr3a', 'eqpo_gr3a', 'capx_gr3a', 'capx_at', 'rd_at', 'spi_at', 'xido_at', 'nri_at', 'gp_sale', 'ebitda_sale', 'pi_sale', 'ni_sale', 'nix_sale', 'ocf_sale', 'fcf_sale', 'ebitda_at', 'ebit_at', 'fi_at', 'ni_at', 'nix_be', 'ocf_be', 'fcf_be', 'gp_bev', 'ebitda_bev', 'fi_bev', 'cop_bev', 'gp_ppen', 'ebitda_ppen', 'fcf_ppen', 'fincf_at', 'eqis_at', 'dltnetis_at', 'dstnetis_at', 'eqnpo_at', 'eqbb_at', 'div_at', 'be_bev', 'debt_bev', 'cash_bev', 'pstk_bev', 'debtlt_bev', 'debtst_bev', 'int_debt', 'int_debtlt', 'ebitda_debt', 'profit_cl', 'ocf_cl', 'ocf_debt', 'cash_lt', 'inv_act', 'rec_act', 'debtst_debt', 'cl_lt', 'debtlt_debt', 'lt_ppen', 'debtlt_be', 'nwc_at', 'fcf_ocf', 'debt_at', 'debt_be', 'ebit_int', 'inv_days', 'rec_days', 'ap_days', 'cash_conversion', 'cash_cl', 'caliq_cl', 'ca_cl', 'inv_turnover', 'rec_turnover', 'ap_turnover', 'adv_sale', 'staff_sale', 'sale_be', 'div_ni', 'sale_nwc', 'tax_pi', 'ni_emp', 'sale_emp', 'niq_saleq_std', 'roeq_be_std', 'roe_be_std', 'intrinsic_value', 'gpoa_ch5', 'roe_ch5', 'roa_ch5', 'cfoa_ch5', 'gmar_ch5', 'cash_me', 'gp_me', 'ebitda_me', 'ebit_me', 'ope_me', 'nix_me', 'cop_me', 'div_me', 'eqbb_me', 'eqis_me', 'eqnetis_me', 'at_mev', 'ppen_mev', 'be_mev', 'cash_mev', 'sale_mev', 'gp_mev', 'ebit_mev', 'cop_mev', 'ocf_mev', 'fcf_mev', 'debt_mev', 'pstk_mev', 'debtlt_mev', 'debtst_mev', 'dltnetis_mev', 'dstnetis_mev', 'dbnetis_mev', 'netis_mev', 'fincf_mev', 'ivol_capm_60m', 'beta_21d', 'beta_252d', 'rvol_252d', 'rvolhl_21d']
settings = {"pi": 3.1415, "screens": {"nyse_stocks": True}}
data_ret_ld1 = pd.DataFrame()  # Replace with actual data_ret_ld1 DataFrame
wealth = pd.DataFrame()  # Replace with actual wealth DataFrame

# Call the preprocess_chars function
chars = preprocess_chars(data_path_JW, features, settings, data_ret_ld1, wealth)

#%%

# Number of stocks by the end of 2020
num_stocks_2020 = chars[chars["eom"] == pd.to_datetime("2020-12-31")].shape[0]
print(f"Number of stocks by the end of 2020: {num_stocks_2020}")


# Dollar volume of Walmart and Xerox
walmart_xerox_dolvol = chars[chars["id"].isin([22111, 27983])][["id", "eom", "dolvol"]]
walmart_xerox_dolvol["dolvol"] /= 1e6
walmart_xerox_dolvol = walmart_xerox_dolvol[walmart_xerox_dolvol["eom"] == pd.to_datetime("2020-12-31")]

# Auto correlation of ret_1_0 and be_me
auto_corr = ar1_ss[ar1_ss["char"].isin(["be_me", "ret_1_0"])]

# Median auto correlation for "quality" and "momentum" clusters
quality_median_ar1 = cluster_labels.merge(ar1_ss, left_on="characteristic", right_on="char")
quality_median_ar1 = quality_median_ar1[quality_median_ar1["cluster"] == "quality"]["ar1"].median()

momentum_median_ar1 = cluster_labels.merge(ar1_ss, left_on="characteristic", right_on="char")
momentum_median_ar1 = momentum_median_ar1[momentum_median_ar1["cluster"] == "momentum"]["ar1"].median()

# Correlation of be_me and ret_12_1
valid_data_tc = data_tc[
    (data_tc["valid"]) &
    (~data_tc["be_me"].isna()) &
    (~data_tc["ret_12_1"].isna()) &
    (data_tc["be_me"] != 0.5) &
    (data_tc["ret_12_1"] != 0.5)
]

correlation_by_eom = valid_data_tc.groupby("eom")[["be_me", "ret_12_1"]].corr().iloc[0::2, -1].reset_index(drop=True)
correlation_summary = correlation_by_eom.describe()
overall_correlation = valid_data_tc[["be_me", "ret_12_1"]].corr().iloc[0, 1]

# Realized utility without second layer of portfolio tuning
validation_m1_filtered = validation_m1[(validation_m1["k"] == 1) & (validation_m1["g"] == 0) & (validation_m1["u"] == 1)]
validation_m1_result = validation_m1_filtered[["r", "tc"]].agg(lambda x: (x.mean() - 0.5 * x.var() * pf_set["gamma_rel"] - x.mean()) * 12)
validation_m1_sr = (validation_m1_filtered["r"] - validation_m1_filtered["tc"]).mean() / validation_m1_filtered["r"].std() * np.sqrt(12)

# Similar calculations for validation_static
validation_static_1 = validation_static[(validation_static["k"] == 1) & (validation_static["g"] == 0) & (validation_static["u"] == 1)]
validation_static_result_1 = validation_static_1[["r", "tc"]].agg(lambda x: (x.mean() - 0.5 * x.var() * pf_set["gamma_rel"] - x.mean()) * 12)
validation_static_sr_1 = (validation_static_1["r"] - validation_static_1["tc"]).mean() / validation_static_1["r"].std() * np.sqrt(12)

validation_static_2 = validation_static[(validation_static["k"] == 0.2) & (validation_static["g"] == 0) & (validation_static["u"] == 1)]
validation_static_result_2 = validation_static_2[["r", "tc"]].agg(lambda x: (x.mean() - 0.5 * x.var() * pf_set["gamma_rel"] - x.mean()) * 12)
validation_static_sr_2 = (validation_static_2["r"] - validation_static_2["tc"]).mean() / validation_static_2["r"].std() * np.sqrt(12)

# Move investors from 10b to 1b at a relative risk aversion of 10
large = ef_ss[(ef_ss["wealth_end"] == 1e10) & (ef_ss["gamma_rel"] == 10)]
large_r_tc = large["r_tc"].values

# Small investors for a volatility of 14%
small_er = ef_ss[ef_ss["wealth_end"] == 1e9].copy()
small_er["estimated_r_tc"] = (
    large_r_tc +
    (large["sd"].values - small_er["sd"][small_er["gamma_rel"] == 100].values) *
    (small_er["r_tc"][small_er["gamma_rel"] == 20].values - small_er["r_tc"][small_er["gamma_rel"] == 100].values) /
    (small_er["sd"][small_er["gamma_rel"] == 20].values - small_er["sd"][small_er["gamma_rel"] == 100].values)
)

# Median dollar volume by end of sample (for simulations)
median_dolvol = chars[chars["valid"]].groupby("eom")["dolvol"].median() / 1e6
median_dolvol_end_sample = median_dolvol.loc[median_dolvol.index.max()]

# Shorting fees
short_fees_latest = short_fees[short_fees["date"] == short_fees["date"].max()]
short_fees_summary = short_fees_latest.groupby("dcbs").agg(n=("dcbs", "size"), fee=("indicative_fee", "mean"))
short_fees_summary["pct"] = short_fees_summary["n"] / short_fees_summary["n"].sum()

short_sub = short_fees[short_fees["date"] == pd.to_datetime("2020-12-31")].merge(
    chars[(chars["valid"]) & (chars["eom"] == chars["eom"].max())][["id", "eom", "size_grp"]],
    left_on=["permno", "date"],
    right_on=["id", "eom"],
    how="inner"
)

short_fee_availability = short_sub["indicative_fee"].notna().mean()
short_fee_quantiles = short_sub["indicative_fee"].quantile(np.arange(0, 1.01, 0.01))
short_fees_summary_final = short_sub.groupby("dcbs").agg(n=("dcbs", "size"), fee=("indicative_fee", "mean"))
short_fees_summary_final["pct"] = short_fees_summary_final["n"] / short_fees_summary_final["n"].sum()
