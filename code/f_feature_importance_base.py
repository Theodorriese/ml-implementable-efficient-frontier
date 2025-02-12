import pandas as pd
from tqdm import tqdm
from a_portfolio_choice_functions import tpf_cf_fun, pfml_cf_fun

def implement_markowitz_ml_base(chars, er_models, cluster_labels, dates_oos, barra_cov, settings, pf_set, tpf_cf_wealth, features, output_path):
    """
    Implement Markowitz-ML Base Case.

    Parameters:
        chars (pd.DataFrame): Characteristics data.
        er_models (list): Expected return models.
        cluster_labels (pd.DataFrame): Cluster labels.
        dates_oos (list): Out-of-sample dates.
        barra_cov (dict): Covariance matrices.
        settings (dict): Settings dictionary.
        pf_set (dict): Portfolio settings.
        tpf_cf_wealth (pd.DataFrame): Tangency portfolio wealth data.
        features (list): List of features.
        output_path (str): Path to save the output.
    """
    print("Implementing Markowitz-ML Base Case...")
    cf_clusters = ["bm"] + settings["clusters"]
    tpf_cf_base = []

    for cf_cluster in cf_clusters:
        print(f"Processing cluster: {cf_cluster}")
        cluster_result = tpf_cf_fun(
            data=chars[chars["valid"] == True],
            cf_cluster=cf_cluster,
            er_models=er_models,
            cluster_labels=cluster_labels,
            dates=dates_oos,
            cov_list=barra_cov,
            gamma_rel=100,  # Higher gamma for less extreme TPF
            wealth=tpf_cf_wealth,
            seed=settings["seed_no"],
            features=features,
        )
        tpf_cf_base.append(cluster_result)

    tpf_cf_base_combined = pd.concat(tpf_cf_base, axis=0)
    tpf_cf_base_combined.to_csv(f"{output_path}/tpf_cf_base.csv", index=False)


def implement_portfolio_ml_base(chars, output_path, dates_oos, barra_cov, settings, pf_set, wealth, risk_free, lambda_list, features):
    """
    Implement Portfolio-ML Base Case.

    Parameters:
        chars (pd.DataFrame): Characteristics data.
        output_path (str): Path to load and save the Portfolio-ML base case model.
        dates_oos (list): Out-of-sample dates.
        barra_cov (dict): Covariance matrices.
        settings (dict): Settings dictionary.
        pf_set (dict): Portfolio settings.
        wealth (pd.DataFrame): Wealth data.
        risk_free (pd.DataFrame): Risk-free rate data.
        lambda_list (dict): Lambda list for portfolio-ML.
        features (list): List of features.
    """
    print("Loading Portfolio-ML Base Case model...")
    # Load the Portfolio-ML base model
    pfml_path = f"{output_path}/static-ml.pkl"
    pfml = pd.read_pickle(pfml_path)
    print(f"Portfolio-ML Base Case model loaded from {pfml_path}.")

    print("Implementing Portfolio-ML Base Case...")
    cf_clusters = ["bm"] + settings["clusters"]
    pfml_cf_base = []

    for cf_cluster in cf_clusters:
        print(f"Processing cluster: {cf_cluster}")
        cluster_result = pfml_cf_fun(
            data=chars[chars["valid"] == True],
            cf_cluster=cf_cluster,
            pfml_base=pfml,
            dates=dates_oos,
            cov_list=barra_cov,
            lambda_list=lambda_list,
            scale=settings["pf_ml"]["scale"],
            orig_feat=settings["pf_ml"]["orig_feat"],
            gamma_rel=pf_set["gamma_rel"],
            wealth=wealth,
            risk_free=risk_free,
            mu=pf_set["mu"],
            iter=10,
            seed=settings["seed_no"],
            features=features,
        )
        pfml_cf_base.append(cluster_result)

    pfml_cf_base_combined = pd.concat(pfml_cf_base, axis=0)

    # Save Portfolio-ML Base Case results
    pfml_cf_base_combined.to_csv(f"{output_path}/pfml_cf_base.csv", index=False)
    pfml_cf_base_combined.to_pickle(f"{output_path}/pfml_cf_base.pkl")
    print(f"Portfolio-ML Base Case results saved to {output_path}/pfml_cf_base.csv and {output_path}/pfml_cf_base.pkl")


def run_feature_importance_base(chars, er_models, cluster_labels, barra_cov, settings, pf_set, tpf_cf_wealth, wealth, risk_free, lambda_list,
         features, dates_oos, output_path):
    """
    Main function to execute implementations for Markowitz-ML and Portfolio-ML.

    Parameters:
        chars (pd.DataFrame): Characteristics data.
        er_models (list): Expected return models.
        cluster_labels (pd.DataFrame): Cluster labels.
        barra_cov (dict): Covariance matrices.
        settings (dict): Settings dictionary.
        pf_set (dict): Portfolio settings.
        tpf_cf_wealth (pd.DataFrame): Tangency portfolio wealth data.
        wealth (pd.DataFrame): Wealth data.
        risk_free (pd.DataFrame): Risk-free rate data.
        lambda_list (dict): Lambda list for portfolio-ML.
        features (list): List of features.
        dates_oos (list): Out-of-sample dates.
        output_path (str): Path to save the output.
    """
    # Implement Markowitz-ML Base Case
    print("Running Markowitz-ML Base Case...")
    implement_markowitz_ml_base(
        chars=chars,
        er_models=er_models,
        cluster_labels=cluster_labels,
        dates_oos=dates_oos,
        barra_cov=barra_cov,
        settings=settings,
        pf_set=pf_set,
        tpf_cf_wealth=tpf_cf_wealth,
        features=features,
        output_path=output_path,
    )

    # Implement Portfolio-ML Base Case
    print("Running Portfolio-ML Base Case...")
    implement_portfolio_ml_base(
        chars=chars,
        output_path = output_path,
        dates_oos=dates_oos,
        barra_cov=barra_cov,
        settings=settings,
        pf_set=pf_set,
        wealth=wealth,
        risk_free=risk_free,
        lambda_list=lambda_list,
        features=features,
    )
