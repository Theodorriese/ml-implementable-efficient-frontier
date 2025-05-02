import os
import pandas as pd
from b_prepare_data import (load_cluster_labels,
    load_market_data,
    load_risk_free
)
from g_base_analysis import (
    load_base_case_portfolios, combine_portfolios, compute_portfolio_summary,
    compute_and_plot_performance_time_series, compute_probability_of_outperformance,
    compute_and_plot_portfolio_statistics_over_time, compute_and_plot_correlation_matrix,
    plot_apple_vs_xerox, plot_optimal_hyperparameters, compute_ar1_plot,
    process_features_with_sufficient_coverage,
)
from g_economic_intuition import (
    combine_portfolio_weights,
    calculate_and_plot_weight_differences,
    portfolio_analysis
)
from g_feature_importance import (
    calculate_feature_importance,
    calculate_ief_summary,
    plot_with_trading_costs,
    plot_counterfactual_ef_without_tc,
    plot_feature_importance_for_return_predictions,
    analyze_seasonality_effect
)
from i1_Main import (settings, pf_set, features, pf_order, pf_order_new, main_types,
                     cluster_order, feat_excl)
from g_implementable_efficient_frontier import run_ief


from g_alpha_analysis import (
    plot_alpha_decay_cumulative_continuous,
    plot_alpha_decay_rolling_tstat,
    compute_signal_rank_stability
)

# # -------------------- CONFIGURATION (local) --------------------
# data_path = r"C:\Master"
# base_path = r"C:\Master\Data\Generated\Portfolios"
# output_path = r"C:\Master\Data\Generated\Analysis"
# latest_folder = r"C:\Master\Data\Generated\Portfolios\demo"
# model_folder = r"C:\Master\Outputs"

# -------------------- CONFIGURATION --------------------
data_path = "/work/frontier_ml_europa"
base_path = "/work/frontier_ml_europa/data/Portfolios"
output_path = "/work/frontier_ml_europa/data/Analysis"
latest_folder = "/work/frontier_ml_europa/data/Portfolios/demo"
model_folder = "/work/frontier_ml_europa/Outputs"

# # -------------------- CONFIGURATION (23Y outputs) --------------------
# data_path = r"C:\Master"
# base_path = r"C:\Master\Results_storage\23Y_US_Results_2000_2023_140425\Data\Portfolios"
# output_path = r"C:\Master\Results_storage\23Y_US_Results_2000_2023_140425\Data\Analysis"
# latest_folder = r"C:\Master\Results_storage\23Y_US_Results_2000_2023_140425\Data\Portfolios\demo"
# model_folder = r"C:\Master\Results_storage\23Y_US_Results_2000_2023_140425\Data\Outputs"

# # -------------------- CONFIGURATION (36Y outputs) --------------------
# data_path = r"C:\Master"
# base_path = r"C:\Master\Results_storage\36Y_US_Results_150425\Data\Portfolios"
# output_path = r"C:\Master\Results_storage\36Y_US_Results_150425\Data\Analysis"
# latest_folder = r"C:\Master\Results_storage\36Y_US_Results_150425\Data\Portfolios\demo"
# model_folder = r"C:\Master\Results_storage\36Y_US_Results_150425\Data\Outputs"

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

# Extract dates
dates_oos = dates_data["dates_oos"]
dates_hp = dates_data["dates_hp"]

# Configuration parameters
gamma_rel = pf_set["gamma_rel"]
colours_theme = ["blue", "green", "red", "orange", "purple", "brown"]


# -------------------- LOAD DATA --------------------
print("Loading base case portfolios...")
base_case = load_base_case_portfolios(base_path)

pfml = base_case["pfml"]
static = base_case["static"]
bm_pfs = base_case["bm_pfs"]
tpf = base_case["tpf"]
factor_ml = base_case["factor_ml"]
mkt = base_case["mkt"]

# 0) Extra plots
features=["market_equity", "dolvol_126d", "ami_126d", "prc",
         "rmax1_21d", "rmax5_21d", "beta_dimson_21d"]
# features = features[:30]

plot_alpha_decay_cumulative_continuous(chars, features, output_path)
plot_alpha_decay_rolling_tstat(chars, features, output_path)
compute_signal_rank_stability(chars, features, output_path)

print("Done with the alphas")

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

# Now, plot the performance time series
performance_fig = compute_and_plot_performance_time_series(
        filtered_pfs,
        main_types,
        start_date,
        end_date,
        )
performance_fig.savefig(os.path.join(output_path, "performance_time_series.png"), bbox_inches="tight", dpi=300)


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


# -------------------- 5) APPLE VS. XEROX PLOT --------------------
print("Generating plot...")

liquid_id = 22111 # 22111 is Johnson and Johnson
illiquid_id = 27983 # 27983 is Xerox
start_year = 2010

apple_vs_xerox = plot_apple_vs_xerox(
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

# -------------------- 6) Optimal Hyper-parameters Plot --------------------
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


# -------------------- 8) Coverage plot --------------------
data, coverage = process_features_with_sufficient_coverage(features, feat_excl, settings)


# -------------------- 9) Implementable  Efficient Frontier --------------------
print("Running initial IEF script...")
ef_all_ss, ef_ss = run_ief(chars, dates_oos, pf_set, wealth, barra_cov, market_data, risk_free, settings, latest_folder, output_path)


# -------------------- 10) PORTFOLIO WEIGHTS ANALYSIS --------------------
print("Analyzing portfolio weights...")

# Combine portfolio weights
weights_combined = combine_portfolio_weights(static, pfml, mkt, tpf, chars, date="2020-11-30")

# Calculate and plot weight differences
calculate_and_plot_weight_differences(weights_combined, settings, save_path=f"{output_path}/weight_differences.png")

# -------------------- 11) Perform Portfolio Analysis --------------------
print("Performing portfolio analysis...")

# Perform portfolio analysis
portfolio_analysis(static, pfml, mkt, tpf, chars, colours_theme, save_path=f"{output_path}/portfolio_analysis.png")

# -------------------- 12) FEATURE IMPORTANCE IN BASE CASE --------------------
print("Calculating feature importance...")

# Calculate and plot feature importance
calculate_feature_importance(pf_set, colours_theme, latest_folder, save_path=f"{output_path}/feature_importance_demo.png")

# # -------------------- 13) IEF SUMMARY --------------------
# print("Calculating IEF summary...")

# # Calculate summary statistics for IEF and save the plot
# ief_summary = calculate_ief_summary(latest_folder, ef_ss, pf_set, save_path=f"{output_path}/ief_summary.png")

# # -------------------- 14) PLOT WITH TRADING COSTS --------------------
# print("Plotting excess returns vs volatility with trading costs...")

# # Plot excess returns vs volatility and save the plot
# plot_with_trading_costs(ef_cf_ss, colours_theme, save_path=f"{output_path}/excess_returns_vs_volatility_with_tc.png")

# -------------------- 15) COUNTERFACTUAL EF WITHOUT TRADING COSTS --------------------
print("Plotting counterfactual efficient frontier without trading costs...")

# Plot counterfactual efficient frontier without trading costs and save the plot
plot_counterfactual_ef_without_tc(latest_folder, colours_theme, save_path=f"{output_path}/counterfactual_ef_without_tc.png")

# -------------------- 16) FEATURE IMPORTANCE FOR RETURN PREDICTION MODELS --------------------
print("Plotting feature importance for return prediction models...")

# Plot feature importance for return prediction models and save the plot
plot_feature_importance_for_return_predictions(latest_folder, colours_theme, save_path=f"{output_path}/feature_importance_return_predictions.png")

# -------------------- 17) ANALYZE SEASONALITY EFFECT --------------------
print("Analyzing seasonality effect...")

# Analyze seasonality effect and save the plot
analyze_seasonality_effect(chars, save_path=f"{output_path}/seasonality_effect.png")

print("Done.")