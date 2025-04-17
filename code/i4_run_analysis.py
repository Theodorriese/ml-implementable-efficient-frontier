#%%
import os
import pandas as pd
from b_prepare_data import load_cluster_labels
from g_base_analysis import (
    load_base_case_portfolios, combine_portfolios, compute_portfolio_summary,
    compute_and_plot_performance_time_series, compute_probability_of_outperformance,
    compute_and_plot_portfolio_statistics_over_time, compute_and_plot_correlation_matrix,
    plot_apple_vs_xerox, plot_optimal_hyperparameters, compute_ar1_plot,
    process_features_with_sufficient_coverage
)
from i1_Main import (settings, pf_set, features, pf_order, pf_order_new, main_types,
                     cluster_order, feat_excl)


# # -------------------- CONFIGURATION (local) --------------------
# data_path = r"C:\Master"
# base_path = r"C:\Master\Data\Generated\Portfolios"
# output_path = r"C:\Master\Data\Generated\Analysis"
# latest_folder = r"C:\Master\Data\Generated\Portfolios\demo"
# model_folder = r"C:\Master\Outputs"

# -------------------- CONFIGURATION --------------------
# data_path = "/work/frontier_ml/data"
# base_path = "/work/frontier_ml/data/Portfolios"
# output_path = "/work/frontier_ml/data/Analysis_EU"
# latest_folder = "/work/frontier_ml/data/Portfolios/demo"
# model_folder = "/work/frontier_ml/Outputs"

# # -------------------- CONFIGURATION (23Y outputs) --------------------
# data_path = r"C:\Master"
# base_path = r"C:\Master\Results_storage\23Y_US_Results_2000_2023_140425\Data\Portfolios"
# output_path = r"C:\Master\Results_storage\23Y_US_Results_2000_2023_140425\Data\Analysis"
# latest_folder = r"C:\Master\Results_storage\23Y_US_Results_2000_2023_140425\Data\Portfolios\demo"
# model_folder = r"C:\Master\Results_storage\23Y_US_Results_2000_2023_140425\Data\Outputs"

# -------------------- CONFIGURATION (36Y outputs) --------------------
data_path = r"C:\Master"
base_path = r"C:\Master\Results_storage\36Y_US_Results_150425\Data\Portfolios"
output_path = r"C:\Master\Results_storage\36Y_US_Results_150425\Data\Analysis"
latest_folder = r"C:\Master\Results_storage\36Y_US_Results_150425\Data\Portfolios\demo"
model_folder = r"C:\Master\Results_storage\36Y_US_Results_150425\Data\Outputs"

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Load Settings and Portfolio Set
settings_path = os.path.join(latest_folder, "settings.pkl")
pf_set_path = os.path.join(latest_folder, "pf_set.pkl")
settings = pd.read_pickle(settings_path)
pf_set = pd.read_pickle(pf_set_path)

# Load Covariance Results
cov_results_path = os.path.join(latest_folder, "cov_results.pkl")
cov_results = pd.read_pickle(cov_results_path)
barra_cov = cov_results["barra_cov"]

# Load Portfolio Data
chars_path = os.path.join(latest_folder, "chars_with_predictions.pkl")
lambda_list_path = os.path.join(latest_folder, "lambda_list.pkl")
dates_path = os.path.join(latest_folder, "dates.pkl")

chars = pd.read_pickle(chars_path)
lambda_list = pd.read_pickle(lambda_list_path)
dates_data = pd.read_pickle(dates_path)

# Load cluster labels
cluster_labels = load_cluster_labels(data_path)
print("Loaded cluster labels")

# Extract dates
dates_oos = dates_data["dates_oos"]
dates_hp = dates_data["dates_hp"]

# Configuration parameters
config_params = {
    "size_screen": "perc_low50_high100_min40",
    "wealth": pf_set["wealth"],
    "gamma_rel": pf_set["gamma_rel"],
    "industry_cov": settings["cov_set"]["industries"],
    "update_mp": True,
    "update_base": True,
    "update_fi_base": True,
    "update_fi_ief": True,
    "update_fi_ret": True,
}

gamma_rel = config_params["gamma_rel"]


colours_theme = ["blue", "green", "red", "orange", "purple", "brown"] # For parameter and ACR plots later

# -------------------- LOAD DATA --------------------
print("Loading base case portfolios...")

base_case = load_base_case_portfolios(base_path)

# mp = base_case["mp"]
pfml = base_case["pfml"]
static = base_case["static"]
bm_pfs = base_case["bm_pfs"]
tpf = base_case["tpf"]
factor_ml = base_case["factor_ml"]
mkt = base_case["mkt"]

# Combine portfolios
# pfs = combine_portfolios(mp, pfml, static, bm_pfs, pf_order, gamma_rel)
pfs = combine_portfolios(pfml, static, bm_pfs, pf_order, gamma_rel)

# 2) Compute portfolio summary
pf_summary, filtered_pfs = compute_portfolio_summary(pfs, main_types, pf_order_new, gamma_rel)
pf_summary.to_csv(os.path.join(output_path, "portfolio_summary.csv"), index=False)
print('Portfolio summary created')

#%%
# -------------------- 1) PERFORMANCE TIME SERIES --------------------
print("Computing and plotting performance time series...")

# Dynamically determine start_date and end_date based on pfs['eom_ret']
start_date = pd.to_datetime(pfs['eom_ret'].min()).strftime('%Y-%m-%d')
end_date = pd.to_datetime(pfs['eom_ret'].max()).strftime('%Y-%m-%d')

# Now, plot the performance time series
performance_fig = compute_and_plot_performance_time_series(
        filtered_pfs,
        main_types,
        start_date,
        end_date,
        # plot_colors=plot_colors,
        # plot_linestyles=plot_linestyles
        )
performance_fig.savefig(os.path.join(output_path, "performance_time_series.png"), bbox_inches="tight", dpi=300)


#%%
# -------------------- 2) PROBABILITY OF OUTPERFORMANCE --------------------
print("Computing probability of outperformance...")

prob_outperformance_df = compute_probability_of_outperformance(filtered_pfs, main_types)
prob_outperformance_df.to_csv(os.path.join(output_path, "probability_of_outperformance.csv"), index=False)
print("Probability of outperformance saved as csv")

#%%
# -------------------- 3) PORTFOLIO STATISTICS OVER TIME --------------------
print("Computing portfolio statistics over time...")

portfolio_stats_timeseries = compute_and_plot_portfolio_statistics_over_time(
    pfml=pfml,
    tpf=tpf,
    # mp=mp,
    static=static,
    factor_ml=factor_ml,
    pfs=pfs,
    barra_cov=barra_cov,
    dates_oos=dates_oos,
    pf_order=pf_order,
    main_types=main_types,
)
portfolio_stats_timeseries.savefig(os.path.join(output_path, "portfolio_stats_time_series.png"), bbox_inches="tight", dpi=300)

#%%
# -------------------- 4) CORRELATION MATRIX --------------------
print("Computing and plotting correlation matrix...")

correlation_matrix = compute_and_plot_correlation_matrix(filtered_pfs, main_types)
correlation_matrix.to_csv(os.path.join(output_path, "correlation_matrix.csv"))

#%%
# -------------------- 5) APPLE VS. XEROX PLOT --------------------
print("Generating plot...")

liquid_id = 22111 # 22111 is Johnson and Johnson
illiquid_id = 27983 # 27983 is Xerox
start_year = 2019

apple_vs_xerox = plot_apple_vs_xerox(
    # mp=mp,
    pfml=pfml,
    static=static,
    tpf=tpf,
    factor_ml=factor_ml,
    mkt=mkt,
    pfs=pfs,
    liquid_id=liquid_id,
    illiquid_id=illiquid_id,
    start_year=start_year
)
apple_vs_xerox.savefig(os.path.join(output_path, "apple_vs_xerox.png"), bbox_inches="tight", dpi=300)

#%%
# -------------------- 6) Optimal Hyper-parameters Plot --------------------
fig_hyper, fig_tuning = plot_optimal_hyperparameters(
    model_folder=model_folder,
    # mp=mp["hps"],
    static=static["hps"],
    pfml=pfml["best_hps"],
    colours_theme=colours_theme,
    start_year=start_year
)
fig_hyper.savefig(os.path.join(output_path, "optimal_hyperparameters.png"), bbox_inches="tight", dpi=300)
fig_tuning.savefig(os.path.join(output_path, "portfolio_tuning_results.png"), bbox_inches="tight", dpi=300)



#%%
# -------------------- 7) Autocorrelation plot --------------------
ar1_plot = compute_ar1_plot(
    chars=chars,
    features=features,
    cluster_labels=cluster_labels,
    output_path=output_path
)
ar1_plot.savefig(os.path.join(output_path, "ar1_plot.png"), bbox_inches="tight", dpi=300)

    #%%
# -------------------- 8) Coverage plot (NOT USED) --------------------
# data, coverage = process_features_with_sufficient_coverage(features, feat_excl, settings)

print("Analysis completed successfully!")

#%%
