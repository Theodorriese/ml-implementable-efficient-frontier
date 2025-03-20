import os
import pandas as pd
from i1_Main import settings, pf_set, features
from b_prepare_data import (
    load_risk_free,
    load_market_data,
    load_monthly_data,
    wealth_func,
    preprocess_chars,
    filter_chars,
    feature_screen,
    standardize_features,
    impute_features,
    classify_industry,
    validate_observations,
    apply_size_screen,
    apply_addition_deletion_rule,
    show_investable_universe,
    valid_summary,
)
from c_fit_models import fit_models

# -------------------- Config Parameters and Search grid (Manually Set) --------------------
config_params = {
    "size_screen": "perc_low50_high100_min40",  # Adjusted to min 40 from 50
    "wealth": 1e10,
    "gamma_rel": 10,
    "industry_cov": True,
    "update_mp": False,
    "update_base": False,
    "update_fi_base": False,
    "update_fi_ief": False,
    "update_fi_ret": False
}

search_grid = pd.DataFrame({
    "name": [f"m{i}" for i in range(1, 13)],
    "horizon": [[i] for i in range(1, 13)]
})

# Assign values from config_params to settings and portfolio configurations
settings["screens"]["size_screen"] = config_params["size_screen"]
settings["cov_set"]["industries"] = config_params["industry_cov"]
pf_set["wealth"] = config_params["wealth"]
pf_set["gamma_rel"] = config_params["gamma_rel"]


# -------------------- Run  --------------------
def prepare_and_fit_models(settings, pf_set, features, output_path):
    """
    Prepare data and fit models.

    Parameters:
        settings (dict): Configuration settings.
        pf_set (dict): Portfolio-specific settings.
        features (list): List of features to include in the analysis.
        output_path (str): Directory to save outputs.

    Returns:
        tuple: Processed characteristics dataframe, trained models.
    """
    # -------------------- Step 1: Data Preparation --------------------
    # print("Step 1: Loading risk-free rate data...")
    # risk_free = load_risk_free(settings["data_path"])
    #
    # print("Step 2: Loading market data...")
    # market_data = load_market_data(settings["data_path"])
    #
    # print("Step 3: Computing wealth function...")
    # wealth = wealth_func(
    #     wealth_end=pf_set["wealth"],
    #     end=settings["screens"]["end"],
    #     market=market_data,
    #     risk_free=risk_free
    # )
    #
    # wealth.to_pickle(os.path.join(settings["data_path"], "wealth_processed.pkl"))
    #
    # print("Step 4: Loading monthly return data...")
    # data_ret = load_monthly_data(
    #     settings["data_path"],
    #     settings=settings,
    #     risk_free=risk_free
    # )
    #
    # print("Step 5: Preprocessing characteristics...")
    # chars = preprocess_chars(
    #     settings["data_path"],
    #     features=features,
    #     settings=settings,
    #     data_ret_ld1=data_ret,
    #     wealth=wealth
    # )
    #
    # print("Step 6: Filtering characteristics by date...")
    # chars, n_start, me_start = filter_chars(chars, settings)
    #
    # print("Step 7: Screening features...")
    # chars = feature_screen(chars, features, settings, n_start, me_start, run_sub=False)
    #
    # print("Step 8: Standardizing features...")
    # chars = standardize_features(chars, features, settings)
    #
    # print("Step 9: Imputing missing feature values...")
    # chars = impute_features(chars, features, settings)
    #
    # print("Step 10: Classifying industries...")
    # chars = classify_industry(chars)
    #
    # print("Step 11: Validating observations...")
    # chars = validate_observations(chars, pf_set)
    #
    # print("Step 12: Applying size screen...")
    # chars = apply_size_screen(chars, settings)
    #
    # print("Step 13: Applying addition and deletion rules...")
    # chars = apply_addition_deletion_rule(chars, settings)
    #
    # data_ret.to_pickle(os.path.join(settings["data_path"], "data_ret_processed.pkl"))
    # chars.to_pickle(os.path.join(settings["data_path"], "chars_processed.pkl"))
    #
    # print("Step 14: Showing investable universe and summary...")
    # show_investable_universe(chars)
    # valid_summary(chars)

    ########################################################################

    # -------------------- Step 2: Model Fitting --------------------

    # # To just load them

    print("Loading monthly return data...")
    data_ret = pd.read_csv(os.path.join(settings["data_path"], "data_ret_processed.csv"))

    # Ensure correct date formats for `data_ret`
    data_ret["eom"] = pd.to_datetime(data_ret["eom"])
    data_ret["eom_m"] = pd.to_datetime(data_ret["eom_m"])

    print("Loading preprocessed characteristics data...")
    chars = pd.read_csv(os.path.join(settings["data_path"], "chars_processed.csv"))

    # Ensure correct date formats for `chars`
    chars["eom"] = pd.to_datetime(chars["eom"])

    # To fit them
    print("Step 15: Fitting return prediction models...")
    models = fit_models(
        search_grid=search_grid,
        data_ret=data_ret,
        chars=chars,
        settings=settings,
        features=features,
        output_path=output_path
    )

    print("Data preparation and model fitting complete.")
    return chars, models


# Main Runner
if __name__ == "__main__":
    # -------------------- Configuration and Paths --------------------
    output_dir = "Outputs"
    os.makedirs(output_dir, exist_ok=True)

    print("Starting build portfolio script...")

    # -------------------- Execute Data Preparation and Model Fitting --------------------
    chars, models = prepare_and_fit_models(
        settings=settings,
        pf_set=pf_set,
        features=features,
        output_path=output_dir
    )

    # -------------------- Save Outputs --------------------
    models_file = os.path.join(output_dir, "fitted_models.pkl")
    print(f"Saving fitted models to {models_file}...")
    pd.to_pickle(models, models_file)

    print("Build portfolio script completed successfully.")
