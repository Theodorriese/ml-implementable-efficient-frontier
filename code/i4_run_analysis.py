import os
import pandas as pd
from g_base_analysis import (
    load_base_case_portfolios, combine_portfolios, compute_portfolio_summary,
    compute_and_plot_performance_time_series, compute_probability_of_outperformance,
    compute_and_plot_portfolio_statistics_over_time, compute_and_plot_correlation_matrix,
    plot_apple_vs_xerox
)
from i1_Main import settings, pf_set, features, pf_order, pf_order_new, main_types, cluster_order


# -------------------- CONFIGURATION --------------------
base_path = r"C:\Master\Data\Generated\Portfolios"
output_path = r"C:\Master\Data\Generated\Analysis"
latest_folder = r"C:\Master\Data\Generated\Portfolios\demo"

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
colours_theme = ["blue", "green", "red", "orange", "purple", "brown"]

# -------------------- LOAD DATA --------------------
print("Loading base case portfolios...")

base_case = load_base_case_portfolios(base_path)

mp = base_case["mp"]
pfml = base_case["pfml"]
static = base_case["static"]
bm_pfs = base_case["bm_pfs"]
tpf = base_case["tpf"]
factor_ml = base_case["factor_ml"]


# Combine portfolios
pfs = combine_portfolios(mp, pfml, static, bm_pfs, pf_order, gamma_rel)

# 2) Compute portfolio summary
pf_summary, filtered_pfs = compute_portfolio_summary(pfs, main_types, pf_order_new, gamma_rel)
pf_summary.to_csv(os.path.join(output_path, "portfolio_summary.csv"), index=False)



# -------------------- 1) PERFORMANCE TIME SERIES --------------------
print("Computing and plotting performance time series...")

# Dynamically determine start_date and end_date based on pfs['eom_ret']
start_date = pd.to_datetime(pfs['eom_ret'].min()).strftime('%Y-%m-%d')
end_date = pd.to_datetime(pfs['eom_ret'].max()).strftime('%Y-%m-%d')

# Now, plot the performance time series
performance_fig = compute_and_plot_performance_time_series(filtered_pfs, main_types, start_date, end_date)
performance_fig.savefig(os.path.join(output_path, "performance_time_series.png"))


# -------------------- PROBABILITY OF OUTPERFORMANCE --------------------
print("Computing probability of outperformance...")

prob_outperformance_df = compute_probability_of_outperformance(filtered_pfs, main_types)
prob_outperformance_df.to_csv(os.path.join(output_path, "probability_of_outperformance.csv"), index=False)

# -------------------- PORTFOLIO STATISTICS OVER TIME --------------------
print("Computing portfolio statistics over time...")

compute_and_plot_portfolio_statistics_over_time(
    pfml=pfml,
    tpf=tpf,
    mp=mp,
    static=static,
    factor_ml=factor_ml,
    pfs=pfs,
    barra_cov=barra_cov,
    dates_oos=dates_oos,
    pf_order=pf_order,
    main_types=main_types
)


# -------------------- CORRELATION MATRIX --------------------
print("Computing and plotting correlation matrix...")

correlation_matrix = compute_and_plot_correlation_matrix(filtered_pfs, main_types)
correlation_matrix.to_csv(os.path.join(output_path, "correlation_matrix.csv"))

# -------------------- APPLE VS. XEROX PLOT --------------------
print("Generating Apple vs. Xerox plot...")

liquid_id = 22111  # Johnson & Johnson (example liquid stock ID)
illiquid_id = 27983  # Xerox (example illiquid stock ID)

plot_apple_vs_xerox(
    mp, pfml, static, bm_pfs, filtered_pfs,
    liquid_id, illiquid_id, start_year=2015
)

print("Analysis completed successfully!")
