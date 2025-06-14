import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from a_portfolio_choice_functions import tpf_cf_fun, pfml_cf_fun


def implement_markowitz_ml_base(chars, er_models, cluster_labels, dates_oos, barra_cov,
                                 settings, wealth, features, output_path, pf_set):
    """
    Implement Markowitz-ML Base Case with optional multiprocessing.
    """
    print("Implementing Markowitz-ML Base Case...")
    clusters = sorted(cluster_labels['cluster'].unique())
    cf_clusters = ["bm"] + clusters

    def process_cluster(cf_cluster):
        return tpf_cf_fun(
            data=chars[chars["valid"] == True],
            cf_cluster=cf_cluster,
            er_models=er_models,
            cluster_labels=cluster_labels,
            dates=dates_oos,
            cov_list=barra_cov,
            gamma_rel=pf_set["gamma_rel"],
            wealth=wealth,
            seed=settings["seed_no"],
            features=features,
        )

    if settings.get("multi_process", False):
        print("Using multiprocessing across clusters...")
        num_cores = max(1, int(cpu_count() - 2))
        tpf_cf_base = Parallel(n_jobs=num_cores)(
            delayed(process_cluster)(cf_cluster) for cf_cluster in cf_clusters
        )
    else:
        print("Using single-core processing across clusters...")
        tpf_cf_base = [process_cluster(cf_cluster) for cf_cluster in tqdm(cf_clusters, desc="Processing Clusters", unit="cluster")]

    tpf_cf_base_combined = pd.concat(tpf_cf_base, axis=0)
    tpf_cf_base_combined.to_pickle(f"{output_path}/tpf_cf_base.pkl")
    print(f"Markowitz-ML Base Case results saved to {output_path}/tpf_cf_base.pkl.")


def implement_portfolio_ml_base(chars, cluster_labels, barra_cov, settings, pf_set,
                                wealth, risk_free, lambda_list, features, dates_oos, output_path):
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

    # I know bad syntax with nested func but it works for now - can be rewritten later
    def process_cluster(cf_cluster):
        return pfml_cf_fun(
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
            iter=100,
            seed=settings["seed_no"],
            features=features,
            cluster_labels=cluster_labels
        )

    if settings.get("multi_process", False):
        print("Using multiprocessing across clusters...")
        num_cores = max(1, int(cpu_count() - 2))
        pfml_cf_base = Parallel(n_jobs=num_cores)(
            delayed(process_cluster)(cf_cluster) for cf_cluster in cf_clusters
        )
    else:
        print("Using single-core processing across clusters...")
        pfml_cf_base = [process_cluster(cf_cluster) for cf_cluster in tqdm(cf_clusters, desc="Processing Clusters", unit="cluster")]

    pfml_cf_base_combined = pd.concat(pfml_cf_base, axis=0)
    pfml_cf_base_combined.to_pickle(f"{output_path}/pfml_cf_base.pkl")
    print(f"Portfolio-ML Base Case results saved to {output_path}/pfml_cf_base.pkl.")


def run_feature_importance_base(chars, er_models, cluster_labels, barra_cov, settings, pf_set,
                                wealth, risk_free, lambda_list, features, dates_oos, output_path):
    """
    Main function to execute implementations for Markowitz-ML and Portfolio-ML.
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
        features=features,
        output_path=output_path,
        pf_set=pf_set,
        wealth=wealth,
    )

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
