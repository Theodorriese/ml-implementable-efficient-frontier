import pandas as pd
from a_portfolio_choice_functions import pfml_cf_fun


def load_pfml_for_gamma(gamma_rel, ief_path, settings):
    """
    Load Portfolio-ML results for a given gamma value.

    Parameters:
        gamma_rel (float): Gamma value for risk aversion.
        ief_path (str): Path to IEF portfolio data.
        settings (dict): Settings dictionary.

    Returns:
        dict: Loaded Portfolio-ML data.
    """
    folder_name = f"20240514-20_WEALTH1e+10_GAMMA{gamma_rel}_SIZEperc_low50_high100_min50_INDTRUE"
    file_path = f"{ief_path}{folder_name}/portfolio-ml.pkl"
    return pd.read_pickle(file_path)


def implement_pfml_cf_clusters(chars, pfml, dates_oos, barra_cov, settings, pf_set, wealth, risk_free,
                               gamma_rel, lambda_list, cluster_labels):
    """
    Implement PFML for specified clusters with lambda_list.

    Parameters:
        chars (pd.DataFrame): Characteristics data.
        pfml (dict): Base PFML results.
        dates_oos (list): Out-of-sample dates.
        barra_cov (dict): Covariance matrices.
        settings (dict): Settings dictionary.
        pf_set (dict): Portfolio settings.
        wealth (pd.DataFrame): Wealth data.
        risk_free (pd.DataFrame): Risk-free rate data.
        gamma_rel (float): Gamma value for risk aversion.
        lambda_list (dict): Lambda values for each date.
        cluster_labels (Any): Cluster labels to be passed to pfml_cf_fun.

    Returns:
        pd.DataFrame: Combined PFML cluster results.
    """
    ief_cf_clusters = ["quality", "value", "momentum", "short_term_reversal"]
    pfml_cf_base = []

    for cf_cluster in ief_cf_clusters:
        print(f"Processing cluster: {cf_cluster} with gamma: {gamma_rel}")
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
            features=pf_set["features"],
            cluster_labels=cluster_labels
        )

        cluster_result["gamma_rel"] = gamma_rel
        pfml_cf_base.append(cluster_result)

    return pd.concat(pfml_cf_base, axis=0)


def portfolio_ml_ief(chars, barra_cov, wealth, dates_oos, risk_free, settings, pf_set, lambda_list,
                     output_path, cluster_labels):
    """
    Implement Portfolio-ML - IEF.

    Parameters:
        chars (pd.DataFrame): Characteristics data.
        barra_cov (dict): Covariance matrices.
        wealth (pd.DataFrame): Wealth data.
        dates_oos (list): Out-of-sample dates.
        risk_free (pd.DataFrame): Risk-free rate data.
        settings (dict): Settings dictionary.
        pf_set (dict): Portfolio settings.
        lambda_list (dict): Lambda values for each date.
        output_path (str): Path to save results.
        cluster_labels (Any): Cluster labels to be passed to implement_pfml_cf_clusters.
    """
    print("Implementing Portfolio-ML - IEF...")
    ief_path = "Data/Generated/Portfolios/IEF/"
    pfml_cf_ief = []

    for gamma_rel in settings["ef"]["gamma_rel"]:
        print(f"Processing gamma: {gamma_rel}")
        pfml = load_pfml_for_gamma(gamma_rel, ief_path, settings)
        pfml_cf_base = implement_pfml_cf_clusters(
            chars=chars,
            pfml=pfml,
            dates_oos=dates_oos,
            barra_cov=barra_cov,
            settings=settings,
            pf_set=pf_set,
            wealth=wealth,
            risk_free=risk_free,
            gamma_rel=gamma_rel,
            lambda_list=lambda_list,
            cluster_labels=cluster_labels
        )
        pfml_cf_ief.append(pfml_cf_base)

    # Combine all results and save
    pfml_cf_ief_combined = pd.concat(pfml_cf_ief, axis=0)
    pfml_cf_ief_combined.to_csv(f"{output_path}/pfml_cf_ief.csv", index=False)


# Main function
def run_feature_importance_IEF(chars, barra_cov, wealth, dates_oos, risk_free, settings, pf_set, lambda_list,
                               output_path, cluster_labels):
    """
    Main function to execute Portfolio-ML IEF.

    Parameters:
        chars (pd.DataFrame): Characteristics data.
        barra_cov (dict): Covariance matrices.
        wealth (pd.DataFrame): Wealth data.
        dates_oos (list): Out-of-sample dates.
        risk_free (pd.DataFrame): Risk-free rate data.
        settings (dict): Settings dictionary.
        pf_set (dict): Portfolio settings.
        lambda_list (dict): Lambda values for each date.
        output_path (str): Path to save the output.
        cluster_labels (Any): Cluster labels to be passed to portfolio_ml_ief.
    """
    print("Starting Portfolio-ML IEF execution...")
    portfolio_ml_ief(
        chars=chars,
        barra_cov=barra_cov,
        wealth=wealth,
        dates_oos=dates_oos,
        risk_free=risk_free,
        settings=settings,
        pf_set=pf_set,
        lambda_list=lambda_list,
        output_path=output_path,
        cluster_labels=cluster_labels
    )