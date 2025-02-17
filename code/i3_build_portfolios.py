import os
import time
import pandas as pd
import openpyxl
from datetime import datetime
from b_prepare_data import load_cluster_labels, load_risk_free, load_daily_returns_pkl
from d_estimate_cov_matrix import prepare_cluster_data  # Covariance estimation
from e_prepare_portfolio_data import run_prepare_portfolio_data  # Portfolio prep
from i1_Main import settings, pf_set, features

# -------------------- CONFIGURATION --------------------

config_params = {
    "size_screen": "perc_low50_high100_min40",
    "wealth": pf_set["wealth"],
    "gamma_rel": pf_set["gamma_rel"],
    "industry_cov": settings["cov_set"]["industries"],
    "update_mp": False,
    "update_base": False,
    "update_fi_base": False,
    "update_fi_ief": False,
    "update_fi_ret": False,
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

data_path = r"C:\Master"  # Main data location
get_from_path_model = os.path.join(data_path, "Outputs")  # Model outputs

# Create unique output folder in C:\Master\Data\Generated\Portfolios
timestamp = datetime.now().strftime("%Y%m%d-%H%M")
output_path = os.path.join(data_path, "Data", "Generated", "Portfolios",
                           f"{timestamp}_WEALTH{pf_set['wealth']}_GAMMA{pf_set['gamma_rel']}_SIZE{config_params['size_screen']}_IND{config_params['industry_cov']}")
os.makedirs(output_path, exist_ok=True)

# Save config for reproducibility
pd.to_pickle(settings, os.path.join(output_path, "settings.pkl"))
pd.to_pickle(pf_set, os.path.join(output_path, "pf_set.pkl"))

# -------------------- RUN PORTFOLIO CONSTRUCTION --------------------
start_time = time.time()
print("\nStarting portfolio construction...")

# -------------------- Step 1: Load Processed Data --------------------
print("Step 1: Loading processed data...")

# Load fitted models
fitted_models_path = os.path.join(get_from_path_model, "fitted_models.pkl")
if not os.path.exists(fitted_models_path):
    raise FileNotFoundError(f"Fitted models file not found: {fitted_models_path}")
fitted_models = pd.read_pickle(fitted_models_path)

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

# Load cluster labels
cluster_labels = load_cluster_labels(data_path)
print("Loaded cluster labels:")
print(cluster_labels.head())

# Load risk-free rate
risk_free = load_risk_free(data_path)
print("Loaded risk-free rate data:")
print(risk_free.head())

# Load daily return data
daily_returns = load_daily_returns_pkl(data_path, chars, risk_free)
print("Loaded daily return data:")
print(daily_returns.head())

# -------------------- Step 2: Estimate Covariance Matrix --------------------
print("Step 2: Estimating covariance matrix...")

cov_results = prepare_cluster_data(
    chars=chars,
    cluster_labels=cluster_labels,
    daily=daily_returns,
    settings=settings,
    features=features
)

# Save covariance results
pd.to_pickle(cov_results, os.path.join(output_path, "covariance_matrix.pkl"))

# -------------------- Step 3: Prepare Portfolio Data --------------------
print("Step 3: Preparing portfolio data...")

portfolio_data = run_prepare_portfolio_data(
    chars=chars,
    get_from_path_model=get_from_path_model,
    settings=settings,
    pf_set=pf_set,
    barra_cov=cov_results["factor_cov"]
)

# Save processed portfolio data
pd.to_pickle(portfolio_data["chars"], os.path.join(output_path, "chars_with_predictions.pkl"))
pd.to_pickle(portfolio_data["lambda_list"], os.path.join(output_path, "lambda_list.pkl"))
pd.to_pickle(portfolio_data["dates"], os.path.join(output_path, "dates.pkl"))

# -------------------- Finalization --------------------
elapsed_time = time.time() - start_time
print(f"Portfolio construction script completed in {elapsed_time:.2f} seconds.")
