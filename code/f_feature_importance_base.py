import pandas as pd
from tqdm import tqdm
from a_portfolio_choice_functions import tpf_cf_fun, pfml_cf_fun


def implement_markowitz_ml_base(chars, er_models, cluster_labels, dates_oos, barra_cov,
                                settings, tpf_cf_wealth, features, output_path):
    """
    Implement Markowitz-ML Base Case.
    """
    print("Implementing Markowitz-ML Base Case...")
    clusters = sorted(cluster_labels['cluster'].unique())
    cf_clusters = ["bm"] + clusters
    tpf_cf_base = []

    for cf_cluster in tqdm(cf_clusters, desc="Processing Clusters (Markowitz-ML)", unit="cluster"):
        cluster_result = tpf_cf_fun(
            data=chars[chars["valid"] == True],
            cf_cluster=cf_cluster,
            er_models=er_models,
            cluster_labels=cluster_labels,
            dates=dates_oos,
            cov_list=barra_cov,
            gamma_rel=100,
            wealth=tpf_cf_wealth,
            seed=settings["seed_no"],
            features=features,
        )
        tpf_cf_base.append(cluster_result)

    tpf_cf_base_combined = pd.concat(tpf_cf_base, axis=0)
    tpf_cf_base_combined.to_pickle(f"{output_path}/tpf_cf_base.pkl")
    print(f"Markowitz-ML Base Case results saved to {output_path}/tpf_cf_base.pkl.")


def implement_portfolio_ml_base(chars, output_path, dates_oos, barra_cov, settings, pf_set, wealth,
                                risk_free, lambda_list, features, cluster_labels):
    """
    Implement Portfolio-ML Base Case.
    """
    print("Loading Portfolio-ML Base Case model...")
    pfml_path = f"{output_path}/portfolio-ml.pkl"
    pfml = pd.read_pickle(pfml_path)
    print(f"Portfolio-ML Base Case model loaded from {pfml_path}.")

    print("Implementing Portfolio-ML Base Case...")
    clusters = sorted(cluster_labels['cluster'].unique())
    cf_clusters = ["bm"] + clusters
    pfml_cf_base = []

    for cf_cluster in tqdm(cf_clusters, desc="Processing Clusters (Portfolio-ML)", unit="cluster"):
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
            cluster_labels=cluster_labels
        )

        pfml_cf_base.append(cluster_result)

    pfml_cf_base_combined = pd.concat(pfml_cf_base, axis=0)
    pfml_cf_base_combined.to_pickle(f"{output_path}/pfml_cf_base.pkl")
    print(f"Portfolio-ML Base Case results saved to {output_path}/pfml_cf_base.pkl.")


def run_feature_importance_base(chars, er_models, cluster_labels, barra_cov, settings, pf_set, tpf_cf_wealth,
                                wealth, risk_free, lambda_list, features, dates_oos, output_path):
    """
    Main function to execute implementations for Markowitz-ML and Portfolio-ML.
    """
    # # Implement Markowitz-ML Base Case
    # print("Running Markowitz-ML Base Case...")
    # implement_markowitz_ml_base(
    #     chars=chars,
    #     er_models=er_models,
    #     cluster_labels=cluster_labels,
    #     dates_oos=dates_oos,
    #     barra_cov=barra_cov,
    #     settings=settings,
    #     tpf_cf_wealth=tpf_cf_wealth,
    #     features=features,
    #     output_path=output_path,
    # )

    # Implement Portfolio-ML Base Case
    print("Running Portfolio-ML Base Case...")
    implement_portfolio_ml_base(
        chars=chars,
        output_path=output_path,
        dates_oos=dates_oos,
        barra_cov=barra_cov,
        settings=settings,
        pf_set=pf_set,
        wealth=wealth,
        risk_free=risk_free,
        lambda_list=lambda_list,
        features=features,
        cluster_labels=cluster_labels
    )
