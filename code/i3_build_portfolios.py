import os
import pandas as pd
from b_prepare_data import load_cluster_labels, load_risk_free
from d_estimate_cov_matrix import prepare_cluster_data
from e_prepare_portfolio_data import run_prepare_portfolio_data
from f_feature_importance_base import run_feature_importance_base
from f_feature_importance_IEF import run_feature_importance_ief
from f_base_case import run_f_base_case
from i1_Main import settings, pf_set, features


# -------------------- CONFIGURATION --------------------
config_params = {
    "size_screen": "perc_low50_high100_min50",
    "wealth": pf_set["wealth"],
    "gamma_rel": pf_set["gamma_rel"],
    "industry_cov": settings["cov_set"]["industries"],
    "update_base": False,
    "update_fi_base": True,
    "update_fi_ief": False,
}

# Print config for verification
print("\nConfiguration Parameters:")
for key, value in config_params.items():
    print(f"  {key}: {value}")

# Assign config to settings
settings["screens"]["size_screen"] = config_params["size_screen"]
settings["cov_set"]["industries"] = config_params["industry_cov"]
pf_set["wealth"] = config_params["wealth"]
pf_set["gamma_rel"] = config_params["gamma_rel"]

# -------------------- DEFINE PATHS --------------------
data_path = settings["data_path"]
portfolios_dir = os.path.join(data_path, "data", "Portfolios")
get_from_path_model = os.path.join(data_path, "Outputs")

latest_folder = os.path.join(portfolios_dir, "demo")
output_path = os.path.join(portfolios_dir, "demo")

# Save config for reproducibility
pd.to_pickle(settings, os.path.join(output_path, "settings.pkl"))
pd.to_pickle(pf_set, os.path.join(output_path, "pf_set.pkl"))


# -------------------- RUN PORTFOLIO CONSTRUCTION --------------------
print("\nStarting portfolio construction...")

# -------------------- Step 1: Load Processed Data --------------------
print("Step 1: Loading processed data...")

# Load preprocessed characteristics
chars_path = os.path.join(data_path, "chars_processed.pkl")
if not os.path.exists(chars_path):
    raise FileNotFoundError(f"Processed chars file not found: {chars_path}")
chars = pd.read_pickle(chars_path)

if 'eom' in chars.columns:
    chars['eom'] = pd.to_datetime(chars['eom'], errors='coerce')
    if chars['eom'].isna().any():
        raise ValueError("Invalid date values in the 'eom' column of chars.")
else:
    raise KeyError("'eom' column is missing from the chars data.")

# Load wealth data
wealth_path = os.path.join(data_path, "wealth_processed.pkl")
if not os.path.exists(wealth_path):
    raise FileNotFoundError(f"Wealth data file not found: {wealth_path}")

wealth = pd.read_pickle(wealth_path)
print("Loaded wealth data.")

# Load cluster labels
cluster_labels = load_cluster_labels(data_path)
print("Loaded cluster labels")

# Load risk-free rate
risk_free = load_risk_free(data_path)
print("Loaded risk-free rate data.")

# Load daily returns
daily_returns = pd.read_pickle(os.path.join(data_path, "daily_returns.pkl"))
print(f"Loaded daily return data from pickle for region: {settings['region']}")

# Load fitted models
fitted_models_path = os.path.join(get_from_path_model, "fitted_models.pkl")
if not os.path.exists(fitted_models_path):
    raise FileNotFoundError(f"Fitted models file not found: {fitted_models_path}")
fitted_models = pd.read_pickle(fitted_models_path)

# -------------------- Step 2: Estimate Covariance Matrix --------------------
print("Step 2: Estimating covariance matrix...")

cov_results = prepare_cluster_data(
    chars=chars,
    cluster_labels=cluster_labels,
    daily=daily_returns,
    settings=settings,
    features=features
)

# Extract components
cluster_data_d = cov_results["cluster_data_d"]
fct_ret = cov_results["fct_ret"]
factor_cov = cov_results["factor_cov"]
spec_risk = cov_results["spec_risk"]
barra_cov = cov_results["barra_cov"]

# Save covariance results
pd.to_pickle(cov_results, os.path.join(output_path, "cov_results.pkl"))

# -------------------- Step 3: Prepare Portfolio Data --------------------
print("Step 3: Preparing portfolio data...")

# Run the portfolio data preparation function and extract its outputs
portfolio_data = run_prepare_portfolio_data(
    chars=chars,
    get_from_path_model=get_from_path_model,
    settings=settings,
    pf_set=pf_set,
    barra_cov=cov_results["barra_cov"]
)

# Save additional processed portfolio data
pd.to_pickle(portfolio_data["chars"], os.path.join(output_path, "chars_with_predictions.pkl"))
pd.to_pickle(portfolio_data["lambda_list"], os.path.join(output_path, "lambda_list.pkl"))
pd.to_pickle(portfolio_data["dates"], os.path.join(output_path, "dates.pkl"))

# Extract dates from portfolio_data
dates_m1, dates_m2, dates_oos, dates_hp, hp_years = (
    portfolio_data["dates"]["dates_m1"],
    portfolio_data["dates"]["dates_m2"],
    portfolio_data["dates"]["dates_oos"],
    portfolio_data["dates"]["dates_hp"],
    portfolio_data["dates"]["hp_years"]
)


# -------------------------- Step 4: Run Base Case ------------------------------
if config_params["update_base"]:
    print("Running Base Case...")

    run_f_base_case(
        chars=portfolio_data["chars"],
        barra_cov=barra_cov,
        wealth=wealth,
        dates_oos=dates_oos,
        pf_set=pf_set,
        settings=settings,
        lambda_list=portfolio_data["lambda_list"],
        risk_free=risk_free,
        features=features,
        dates_m2=dates_m2,
        dates_hp=dates_hp,
        hp_years=hp_years,
        output_path=output_path
    )


# ----------------- Step 5: Run the feature importance scripts -----------------
# -------------------- Feature Importance - Base Case --------------------
if config_params.get('update_fi_base', True):
    print("Running Feature Importance Base Case...")

    run_feature_importance_base(
        chars=portfolio_data['chars'],
        er_models=fitted_models,
        cluster_labels=cluster_labels,
        barra_cov=barra_cov,
        settings=settings,
        pf_set=pf_set,
        wealth=wealth,
        risk_free=risk_free,
        lambda_list=portfolio_data['lambda_list'],
        features=features,
        dates_oos=dates_oos,
        output_path=output_path
    )


# -------------------- Feature Importance - IEF Case --------------------
if config_params.get('update_fi_ief', True):
    print("Running Feature Importance IEF Case...")

    run_feature_importance_ief(
        chars=portfolio_data['chars'],
        barra_cov=barra_cov,
        wealth=wealth,
        dates_oos=dates_oos,
        risk_free=risk_free,
        settings=settings,
        pf_set=pf_set,
        lambda_list=portfolio_data['lambda_list'],
        output_path=output_path,
        cluster_labels=cluster_labels
    )


print(f"Portfolio construction script completed")