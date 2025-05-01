import pandas as pd
import os
from joblib import Parallel, delayed
from multiprocessing import cpu_count

from a_portfolio_choice_functions import pfml_cf_fun
from i1_Main import pf_set, features



def load_pfml(output_path):
    """
    Load Portfolio-ML results from the current dynamic output directory.

    Parameters:
        output_path (str): Path to the dynamically generated portfolio folder.

    Returns:
        dict: Loaded Portfolio-ML data.
    """
    file_path = os.path.join(output_path, "portfolio-ml.pkl")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Portfolio-ML file not found at: {file_path}")
    return pd.read_pickle(file_path)


from joblib import Parallel, delayed
from multiprocessing import cpu_count

def implement_pfml_cf_ief(chars, barra_cov, wealth, dates_oos, risk_free, settings, pf_set, lambda_list,
                          output_path, cluster_labels):
    """
    Implement Portfolio-ML - IEF and PFML for specified clusters with lambda_list.
    """
    print("Implementing Portfolio-ML - IEF...")

    pfml = load_pfml(output_path)
    ief_cf_clusters = ["quality", "value", "momentum", "short_term_reversal"]
    gamma_rel = pf_set["gamma_rel"]

    def process_cluster(cf_cluster):
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
            gamma_rel=gamma_rel,
            wealth=wealth,
            risk_free=risk_free,
            mu=pf_set["mu"],
            iter=10,
            seed=settings["seed_no"],
            features=features,
            cluster_labels=cluster_labels
        )
        cluster_result["gamma_rel"] = gamma_rel
        return cluster_result

    if settings.get("multi_process", False):
        print("Using multiprocessing for IEF clusters...")
        num_cores = max(1, int(cpu_count() * 0.75))
        pfml_cf_base = Parallel(n_jobs=num_cores)(
            delayed(process_cluster)(cf_cluster) for cf_cluster in ief_cf_clusters
        )
    else:
        print("Using single-core processing for IEF clusters...")
        pfml_cf_base = [process_cluster(cf_cluster) for cf_cluster in ief_cf_clusters]

    pfml_cf_ief_combined = pd.concat(pfml_cf_base, axis=0)
    pfml_cf_ief_combined.to_csv(os.path.join(output_path, "pfml_cf_ief.csv"), index=False)
    print(f"✅ Saved PFML-IEF results to {output_path}/pfml_cf_ief.csv")

    return pfml_cf_ief_combined



def run_feature_importance_ief(chars, barra_cov, wealth, dates_oos, risk_free, settings, pf_set, lambda_list,
                               output_path, cluster_labels):
    """
    Main function to execute Portfolio-ML IEF.
    """
    print("Starting Portfolio-ML IEF execution...")
    implement_pfml_cf_ief(
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
