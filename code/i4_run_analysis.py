import os
import pandas as pd
import numpy as np
from b_prepare_data import (load_cluster_labels,
    load_market_data,
    load_risk_free
)
from g_base_analysis import (
    load_base_case_portfolios, combine_portfolios, compute_portfolio_summary,
    compute_probability_of_outperformance,
    compute_and_plot_portfolio_statistics_over_time, compute_and_plot_correlation_matrix,
    plot_liquid_vs_illiquid, plot_optimal_hyperparameters, compute_ar1_plot,
    ts_plot, ts_plot_manual_ylim
)
from g_economic_intuition import (
    combine_portfolio_weights,
    calculate_and_plot_weight_differences,
    calculate_and_plot_weight_differences_rel,
    portfolio_analysis
)
from g_feature_importance import (
    calculate_feature_importance,
    plot_counterfactual
)
from i1_Main import (settings, pf_set, features, pf_order, pf_order_new, main_types,
                     cluster_order, feat_excl)
from g_implementable_efficient_frontier import run_ief
from g_plot_splits import plot_training_val_test_split
from g_plot_distr import plot_combined_return_distributions


# # -------------------- CONFIGURATION (local) --------------------
# data_path = r"C:\Master"
# base_path = r"C:\Master\Data\Generated\Portfolios"
# output_path = r"C:\Master\Data\Generated\Analysis"
# latest_folder = r"C:\Master\Data\Generated\Portfolios\demo"
# model_folder = r"C:\Master\Outputs"

# -------------------- CONFIGURATION --------------------
data_path = "/work/frontier_ml_g10"
base_path = "/work/frontier_ml_g10/data/Portfolios"
output_path = "/work/frontier_ml_g10/data/Analysis"
latest_folder = "/work/frontier_ml_g10/data/Portfolios/demo"
model_folder = "/work/frontier_ml_g10/Outputs"

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

# Load Market data
market_data = load_market_data(settings)

# Load risk-free rate
risk_free = load_risk_free(data_path)

# Load wealth data
wealth_path = os.path.join(data_path, "wealth_processed.pkl")
wealth = pd.read_pickle(wealth_path)

# Extract dates and configuration parameters
dates_oos = dates_data["dates_oos"]
dates_hp = dates_data["dates_hp"]
gamma_rel = pf_set["gamma_rel"]
colours_theme = ["steelblue", "darkorange", "gray"]


# -------------------- LOAD DATA --------------------
print("Loading base case portfolios...")
base_case = load_base_case_portfolios(base_path)

pfml = base_case["pfml"]
static = base_case["static"]
bm_pfs = base_case["bm_pfs"]
tpf = base_case["tpf"]
factor_ml = base_case["factor_ml"]
mkt = base_case["mkt"]

# -------------------- PORTFOLIOS --------------------
# Combine portfolios
pfs = combine_portfolios(pfml, static, bm_pfs, pf_order, gamma_rel)

# Compute portfolio summary
pf_summary, filtered_pfs = compute_portfolio_summary(pfs, main_types, pf_order_new, gamma_rel)
pf_summary.to_csv(os.path.join(output_path, "portfolio_summary.csv"), index=False)
print('Portfolio summary created')


# -------------------- 1) PERFORMANCE TIME SERIES --------------------
print("Computing and plotting performance time series...")

# Dynamically determine start_date and end_date based on pfs['eom_ret']
start_date = pd.to_datetime(pfs['eom_ret'].min()).strftime('%Y-%m-%d')
end_date = pd.to_datetime(pfs['eom_ret'].max()).strftime('%Y-%m-%d')

ts_plot(
        filtered_pfs,
        main_types,
        start_date,
        end_date,
        output_path
        )

ts_plot_manual_ylim(
        filtered_pfs,
        main_types,
        start_date,
        end_date,
        output_path
        )


# -------------------- 2) PROBABILITY OF OUTPERFORMANCE --------------------
print("Computing probability of outperformance...")

prob_outperformance_df = compute_probability_of_outperformance(filtered_pfs, main_types)
prob_outperformance_df.to_csv(os.path.join(output_path, "probability_of_outperformance.csv"), index=False)
print("Probability of outperformance saved as csv")


# -------------------- 3) PORTFOLIO STATISTICS OVER TIME --------------------
print("Computing portfolio statistics over time...")

portfolio_stats_timeseries = compute_and_plot_portfolio_statistics_over_time(
    pfml=pfml,
    tpf=tpf,
    static=static,
    factor_ml=factor_ml,
    pfs=pfs,
    barra_cov=barra_cov,
    dates_oos=dates_oos,
    pf_order=pf_order,
    main_types=main_types,
)
portfolio_stats_timeseries.savefig(os.path.join(output_path, "portfolio_stats_time_series.png"), bbox_inches="tight", dpi=300)


# -------------------- 4) CORRELATION MATRIX --------------------
print("Computing and plotting correlation matrix...")

correlation_matrix = compute_and_plot_correlation_matrix(filtered_pfs, main_types)
correlation_matrix.to_csv(os.path.join(output_path, "correlation_matrix.csv"))


# -------------------- 5) LIQUID VS. ILLIQUID PLOT --------------------
print("Generating plot...")

liquid_id = 10107
illiquid_id = 27983
start_year_liquid = 2000

liquid_vs_illiquid = plot_liquid_vs_illiquid(
    pfml=pfml,
    static=static,
    tpf=tpf,
    factor_ml=factor_ml,
    mkt=mkt,
    pfs=pfs,
    liquid_id=liquid_id,
    illiquid_id=illiquid_id,
    start_year=start_year_liquid
)
liquid_vs_illiquid.savefig(os.path.join(output_path, "liquid_vs_illiquid.png"), bbox_inches="tight", dpi=300)

# -------------------- 6) Optimal Hyper-parameters Plot --------------------
start_year = 1996

fig_hyper, fig_tuning = plot_optimal_hyperparameters(
    model_folder=model_folder,
    static=static["hps"],
    pfml=pfml["best_hps"],
    colours_theme=colours_theme,
    start_year=start_year
)
fig_hyper.savefig(os.path.join(output_path, "optimal_hyperparameters.png"), bbox_inches="tight", dpi=300)
fig_tuning.savefig(os.path.join(output_path, "portfolio_tuning_results.png"), bbox_inches="tight", dpi=300)

# -------------------- 7) Autocorrelation plot --------------------
ar1_plot = compute_ar1_plot(
    chars=chars,
    features=features,
    cluster_labels=cluster_labels,
    output_path=output_path
)
ar1_plot.savefig(os.path.join(output_path, "ar1_plot.png"), bbox_inches="tight", dpi=300)


# -------------------- 8) Implementable  Efficient Frontier --------------------
print("Running initial IEF script...")
ef_all_ss, ef_ss = run_ief(chars, dates_oos, pf_set, wealth, barra_cov, market_data, risk_free, settings, latest_folder, output_path)


# -------------------- 9) PORTFOLIO WEIGHTS ANALYSIS --------------------
print("Analyzing portfolio weights...")

# Combine portfolio weights
weights_combined = combine_portfolio_weights(static, pfml, mkt, tpf, chars, date="2023-11-30")

# Sample once
np.random.seed(settings["seed_no"])
my_ids = np.random.choice(weights_combined["id"].unique(), size=70, replace=False)

# Use for both runs
calculate_and_plot_weight_differences_rel(
    weights=weights_combined,
    types_to_compare=["Portfolio-ML", "Markowitz-ML"],
    settings=settings,
    sample_ids=my_ids,
    save_path=f"{output_path}/rel_weights_1.png"
)

calculate_and_plot_weight_differences_rel(
    weights=weights_combined,
    types_to_compare=["Portfolio-ML", "Static-ML*"],
    settings=settings,
    sample_ids=my_ids,
    save_path=f"{output_path}/rel_weights_2.png"
)

# -------------------- 10) Perform Portfolio Analysis --------------------
print("Performing portfolio analysis...")

# Perform portfolio analysis
portfolio_analysis(static, pfml, mkt, tpf, chars, colours_theme, save_path=f"{output_path}/portfolio_analysis.png")

# -------------------- 11) FEATURE IMPORTANCE IN BASE CASE --------------------
print("Calculating feature importance...")

# Calculate and plot feature importance
calculate_feature_importance(pf_set, latest_folder, save_path=f"{output_path}/feature_importance_demo.png")

# -------------------- 12) COUNTERFACTUAL EF WITHOUT TRADING COSTS --------------------
print("Plotting counterfactual efficient frontier without trading costs...")

# Plot counterfactual efficient frontier
plot_counterfactual(latest_folder, colours_theme, save_path=f"{output_path}/counterfactual_ef_without_tc.png")

# -------------------- 13) Plot training/val/test split  --------------------
print("Plotting training/val/test split...")
plot_training_val_test_split(settings, output_path)

# -------------------- 14) RETURN DISTRIBUTION PLOTS --------------------
print("Plotting return distributions...")

usa_monthly = pd.read_csv(os.path.join(data_path, "world_ret_monthly.csv"))
usa_daily = pd.read_csv(os.path.join(data_path, "usa_dsf.csv"))

plot_combined_return_distributions(usa_monthly, usa_daily)

print("Done.")