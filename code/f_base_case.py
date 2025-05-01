import pandas as pd
import pickle
import os
from a_portfolio_choice_functions import (tpf_implement, factor_ml_implement, mkt_implement,
                                          ew_implement, rw_implement, mv_implement, static_implement,
                                          pfml_implement, mv_implement_multip, static_implement_multip)


def benchmark_portfolios(chars, barra_cov, wealth, dates_oos, pf_set, settings, output_path):
    """
    Generate benchmark portfolios and save the results.

    Parameters:
        chars (pd.DataFrame): Characteristics data.
        barra_cov (dict): Covariance matrices.
        wealth (pd.DataFrame): Wealth data.
        dates_oos (list): Out-of-sample dates.
        pf_set (dict): Portfolio settings.
        settings (dict): Configuration settings.
        output_path (str): Path to save the output.
    """

    # base_path = "/work/frontier_ml/data/Portfolios/demo"

    # # Load previously saved portfolios
    # tpf = pd.read_pickle(os.path.join(base_path, "tpf.pkl"))
    # factor_ml = pd.read_pickle(os.path.join(base_path, "factor_ml.pkl"))
    # mkt = pd.read_pickle(os.path.join(base_path, "mkt.pkl"))
    # ew = pd.read_pickle(os.path.join(base_path, "ew.pkl"))
    # rw = pd.read_pickle(os.path.join(base_path, "rw.pkl"))
    # mv = pd.read_pickle(os.path.join(base_path, "mv.pkl"))

    print("Generating Markowitz-ML Portfolio...")
    tpf = tpf_implement(chars, cov_list=barra_cov, wealth=wealth, dates=dates_oos, gam=pf_set["gamma_rel"])
    pd.to_pickle(tpf, os.path.join(output_path, "tpf.pkl"))

    print("Generating Factor-ML Portfolio...")
    factor_ml = factor_ml_implement(chars, dates=dates_oos, n_pfs=settings["factor_ml"]["n_pfs"], wealth=wealth)
    pd.to_pickle(factor_ml, os.path.join(output_path, "factor_ml.pkl"))

    print("Generating Market Portfolio...")
    mkt = mkt_implement(chars, dates=dates_oos, wealth=wealth)
    pd.to_pickle(mkt, os.path.join(output_path, "mkt.pkl"))

    print("Generating 1/N Portfolio...")
    ew = ew_implement(chars, dates=dates_oos, wealth=wealth)
    pd.to_pickle(ew, os.path.join(output_path, "ew.pkl"))

    print("Generating Rank-Weighted Portfolio...")
    rw = rw_implement(chars, dates=dates_oos, wealth=wealth)
    pd.to_pickle(rw, os.path.join(output_path, "rw.pkl"))

    print("Generating Minimum Variance Portfolio...")

    if settings["multi_process"]:
        print("Using multiprocessing for Minimum Variance...")
        mv = mv_implement_multip(
            data=chars,
            cov_list=barra_cov,
            wealth=wealth,
            dates=dates_oos,
        )
    else:
        print("Using single-core implementation for Minimum Variance...")
        mv = mv_implement(
            data=chars,
            cov_list=barra_cov,
            wealth=wealth,
            dates=dates_oos
        )

    pd.to_pickle(mv, os.path.join(output_path, "mv.pkl"))


    # Filter out any None values to avoid errors
    portfolio_results = [tpf, factor_ml, ew, mkt, rw, mv]
    valid_portfolios = [pf["pf"] for pf in portfolio_results if pf["pf"] is not None]

    if valid_portfolios:
        print("Combining Benchmark Portfolios...")
        bm_pfs = pd.concat(valid_portfolios, ignore_index=True)

        # Save as CSV
        bm_pfs.to_csv(f"{output_path}/bms.csv", index=False)

        # Save the full dictionary as .pkl
        pd.to_pickle({"bm_pfs": bm_pfs}, os.path.join(output_path, "bms.pkl"))
    else:
        print("No valid benchmark portfolios generated.")


def static_ml(chars, barra_cov, lambda_list, wealth, pf_set, settings, dates_oos, dates_hp,
              output_path):
    """
    Implement and save Static-ML portfolio.

    Parameters:
        chars (pd.DataFrame): Characteristics data.
        barra_cov (dict): Covariance matrices.
        lambda_list (dict): Lambda values.
        wealth (pd.DataFrame): Wealth data.
        pf_set (dict): Portfolio settings.
        settings (dict): Configuration settings.
        dates_oos (list): Out-of-sample dates.
        dates_hp (list): Holding period dates.
        output_path (str): Path to save the output.
    """
    print("Implementing Static-ML...")

    if settings.get("multi_process", False):
        print("Using multiprocessing for Static-ML...")
        static = static_implement_multip(
            data_tc=chars,
            cov_list=barra_cov,
            lambda_list=lambda_list,
            wealth=wealth,
            gamma_rel=pf_set["gamma_rel"],
            dates_oos=dates_oos,
            dates_hp=dates_hp,
            k_vec=settings["pf"]["hps"]["static"]["k"],
            u_vec=settings["pf"]["hps"]["static"]["u"],
            g_vec=settings["pf"]["hps"]["static"]["g"],
            cov_type=settings["pf"]["hps"]["cov_type"],
            validation=None,
            seed=None,
        )
    else:
        print("Using single-core implementation for Static-ML...")
        static = static_implement(
            data_tc=chars,
            cov_list=barra_cov,
            lambda_list=lambda_list,
            wealth=wealth,
            gamma_rel=pf_set["gamma_rel"],
            dates_oos=dates_oos,
            dates_hp=dates_hp,
            k_vec=settings["pf"]["hps"]["static"]["k"],
            u_vec=settings["pf"]["hps"]["static"]["u"],
            g_vec=settings["pf"]["hps"]["static"]["g"],
            cov_type=settings["pf"]["hps"]["cov_type"],
            validation=None,
            seed=None
        )

    # Save the output
    with open(f"{output_path}/static-ml.pkl", 'wb') as file:
        pickle.dump(static, file)


def portfolio_ml(chars, barra_cov, lambda_list, features, risk_free, wealth, pf_set, settings, dates_m2, dates_oos,
                 hp_years, output_path):
    """
    Implement and save Portfolio-ML.

    Parameters:
        chars (pd.DataFrame): Characteristics data.
        barra_cov (dict): Covariance matrices.
        lambda_list (dict): Lambda values.
        features (list): Feature columns.
        risk_free (pd.DataFrame): Risk-free rate data.
        wealth (pd.DataFrame): Wealth data.
        pf_set (dict): Portfolio settings.
        settings (dict): Configuration settings.
        dates_m2 (list): Dates for Portfolio-ML.
        dates_oos (list): Out-of-sample dates.
        hp_years (list): Holding period years.
        output_path (str): Path to save the output.
    """
    print("Implementing Portfolio-ML...")

    ###############################################################################
    # BE AWARE OF THIS
    iter = 100
    ###############################################################################

    pfml = pfml_implement(chars, cov_list=barra_cov, lambda_list=lambda_list, features=features, risk_free=risk_free,
                          wealth=wealth, mu=pf_set["mu"], gamma_rel=pf_set["gamma_rel"],
                          dates_full=dates_m2, dates_oos=dates_oos, lb=pf_set["lb_hor"],
                          hp_years=hp_years, rff_feat=True, g_vec=settings["pf_ml"]["g_vec"],
                          p_vec=settings["pf_ml"]["p_vec"], l_vec=settings["pf_ml"]["l_vec"],
                          scale=settings["pf_ml"]["scale"], orig_feat=settings["pf_ml"]["orig_feat"],
                          iter=iter, hps={}, balanced=False, seed=settings["seed_no"])

    with open(f"{output_path}/portfolio-ml.pkl", 'wb') as file:
        pickle.dump(pfml, file)


def run_f_base_case(chars, barra_cov, wealth, dates_oos, pf_set, settings, lambda_list, risk_free,
                    features, dates_m2, dates_hp, hp_years, output_path):
    """
    Main function to run portfolio implementation steps.

    Parameters:
        chars (pd.DataFrame): Characteristics data.
        barra_cov (dict): Covariance matrices.
        wealth (pd.DataFrame): Wealth data.
        dates_oos (list): Out-of-sample dates.
        pf_set (dict): Portfolio settings.
        settings (dict): Configuration settings.
        config_params (dict): Configuration parameters.
        lambda_list (dict): Lambda values.
        risk_free (pd.DataFrame): Risk-free rate data.
        features (list): Feature columns.
        dates_m2, dates_hp (list): Date ranges for calculations.
        hp_years (list): Holding period years.
        output_path (str): Path to save the output.
    """
    # Run benchmark portfolios
    benchmark_portfolios(chars, barra_cov, wealth, dates_oos, pf_set, settings, output_path)

    # Run Static-ML
    static_ml(chars, barra_cov, lambda_list, wealth, pf_set, settings, dates_oos, dates_hp, output_path)

    # Run Portfolio-ML
    portfolio_ml(chars, barra_cov, lambda_list, features, risk_free, wealth, pf_set, settings, dates_m2,
                dates_oos, hp_years, output_path)
