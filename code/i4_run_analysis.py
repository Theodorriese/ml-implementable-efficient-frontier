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

if not os.path.exists(output_path):
    os.makedirs(output_path)

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

# Set configuration parameters
gamma_rel = config_params["gamma_rel"]
colours_theme = ["blue", "green", "red", "orange", "purple", "brown"]

# -------------------- LOAD DATA --------------------
print("Loading base case portfolios...")

base_case = load_base_case_portfolios(base_path)

mp = base_case["mp"]
pfml = base_case["pfml"]
static = base_case["static"]
bm_pfs = base_case["bm_pfs"]

# 1) Combine portfolios
pfs = combine_portfolios(mp, pfml, static, bm_pfs, pf_order, gamma_rel)

# 2) Compute portfolio summary
pf_summary, filtered_pfs = compute_portfolio_summary(pfs, main_types, pf_order_new, gamma_rel)
pf_summary.to_csv(os.path.join(output_path, "portfolio_summary.csv"), index=False)






# -------------------- PERFORMANCE TIME SERIES --------------------
print("Computing and plotting performance time series...")

start_date = "2015-01-01"
end_date = "2025-12-31"
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
