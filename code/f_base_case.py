import pandas as pd
from a_portfolio_choice_functions import (tpf_implement, factor_ml_implement, mkt_implement, ew_implement, rw_implement,
                                          mv_implement, static_implement, pfml_implement, mp_implement)


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
    print("Generating Markowitz-ML Portfolio...")
    tpf = tpf_implement(chars, cov_list=barra_cov, wealth=wealth, dates=dates_oos, gam=pf_set["gamma_rel"])

    print("Generating Factor-ML Portfolio...")
    factor_ml = factor_ml_implement(chars, dates=dates_oos, n_pfs=settings["factor_ml"]["n_pfs"],
                                    wealth=wealth, gam=pf_set["gamma_rel"])

    print("Generating Market Portfolio...")
    mkt = mkt_implement(chars, dates=dates_oos, wealth=wealth)

    print("Generating 1/N Portfolio...")
    ew = ew_implement(chars, dates=dates_oos, wealth=wealth)

    print("Generating Rank-Weighted Portfolio...")
    rw = rw_implement(chars, dates=dates_oos, wealth=wealth)

    print("Generating Minimum Variance Portfolio...")
    mv = mv_implement(chars, cov_list=barra_cov, dates=dates_oos, wealth=wealth)

    print("Combining Benchmark Portfolios...")
    bm_pfs = pd.concat([tpf["pf"], factor_ml["pf"], ew["pf"], mkt["pf"], rw["pf"], mv["pf"]])
    bm_pfs.to_csv(f"{output_path}/bms.csv", index=False)


def static_ml(chars, barra_cov, lambda_list, risk_free, wealth, pf_set, settings, dates_m1, dates_oos, dates_hp,
              hp_years, output_path):
    """
    Implement and save Static-ML portfolio.

    Parameters:
        chars (pd.DataFrame): Characteristics data.
        barra_cov (dict): Covariance matrices.
        lambda_list (dict): Lambda values.
        risk_free (pd.DataFrame): Risk-free rate data.
        wealth (pd.DataFrame): Wealth data.
        pf_set (dict): Portfolio settings.
        settings (dict): Configuration settings.
        dates_m1 (list): Full dates for Static-ML.
        dates_oos (list): Out-of-sample dates.
        dates_hp (list): Holding period dates.
        hp_years (list): Holding period years.
        output_path (str): Path to save the output.
    """
    print("Implementing Static-ML...")
    static = static_implement(chars, cov_list=barra_cov, lambda_list=lambda_list, rf=risk_free,
                               wealth=wealth, mu=pf_set["mu"], gamma_rel=pf_set["gamma_rel"],
                               dates_full=dates_m1, dates_oos=dates_oos, dates_hp=dates_hp,
                               hp_years=hp_years, k_vec=settings["pf"]["hps"]["static"]["k"],
                               u_vec=settings["pf"]["hps"]["static"]["u"], g_vec=settings["pf"]["hps"]["static"]["g"],
                               cov_type=settings["pf"]["hps"]["cov_type"], validation=None)

    if isinstance(static, dict) and "pf" in static:
        static_df = static["pf"]
    else:
        static_df = pd.DataFrame(static)

    static_df.to_pickle(f"{output_path}/static-ml.pkl")


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
    pfml = pfml_implement(chars, cov_list=barra_cov, lambda_list=lambda_list, features=features, risk_free=risk_free,
                          wealth=wealth, mu=pf_set["mu"], gamma_rel=pf_set["gamma_rel"],
                          dates_full=dates_m2, dates_oos=dates_oos, lb=pf_set["lb_hor"],
                          hp_years=hp_years, rff_feat=True, g_vec=settings["pf_ml"]["g_vec"],
                          p_vec=settings["pf_ml"]["p_vec"], l_vec=settings["pf_ml"]["l_vec"],
                          scale=settings["pf_ml"]["scale"], orig_feat=settings["pf_ml"]["orig_feat"],
                          iter=10, hps={}, balanced=False, seed=settings["seed_no"])

    if isinstance(pfml, dict) and "pf" in pfml:
        pfml_df = pfml["pf"]
    else:
        pfml_df = pd.DataFrame(pfml)

    pfml_df.to_pickle(f"{output_path}/portfolio-ml.pkl")


def multiperiod_ml(config_params, chars, barra_cov, lambda_list, risk_free, wealth, pf_set, settings, dates_m1,
                   dates_oos, dates_hp, hp_years, output_path):
    """
    Implement and save Multiperiod-ML if enabled.

    Parameters:
        config_params (dict): Configuration parameters.
        chars (pd.DataFrame): Characteristics data.
        barra_cov (dict): Covariance matrices.
        lambda_list (dict): Lambda values.
        risk_free (pd.DataFrame): Risk-free rate data.
        wealth (pd.DataFrame): Wealth data.
        pf_set (dict): Portfolio settings.
        settings (dict): Configuration settings.
        dates_m1 (list): Full dates for Multiperiod-ML.
        dates_oos (list): Out-of-sample dates.
        dates_hp (list): Holding period dates.
        hp_years (list): Holding period years.
        output_path (str): Path to save the output.
    """
    if config_params["update_mp"]:
        print("Implementing Multiperiod-ML...")
        mp = mp_implement(chars, cov_list=barra_cov, lambda_list=lambda_list, rf=risk_free, wealth=wealth,
                          mu=pf_set["mu"], gamma_rel=pf_set["gamma_rel"], dates_full=dates_m1,
                          dates_oos=dates_oos, dates_hp=dates_hp, hp_years=hp_years,
                          k_vec=settings["pf"]["hps"]["m1"]["k"], u_vec=settings["pf"]["hps"]["m1"]["u"],
                          g_vec=settings["pf"]["hps"]["m1"]["g"], cov_type=settings["pf"]["hps"]["cov_type"],
                          validation=None, iter_=10, K=settings["pf"]["hps"]["m1"]["K"])

        if isinstance(mp, dict) and "pf" in mp:
            mp_df = mp["pf"]
        else:
            mp_df = pd.DataFrame(mp)

        mp_df.to_pickle(f"{output_path}/multiperiod-ml.pkl")


def run_f_base_case(chars, barra_cov, wealth, dates_oos, pf_set, settings, config_params, lambda_list, risk_free, features,
         dates_m1, dates_m2, dates_hp, hp_years, output_path):
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
        dates_m1, dates_m2, dates_hp (list): Date ranges for calculations.
        hp_years (list): Holding period years.
        output_path (str): Path to save the output.
    """
    benchmark_portfolios(chars, barra_cov, wealth, dates_oos, pf_set, settings, output_path)
    static_ml(chars, barra_cov, lambda_list, risk_free, wealth, pf_set, settings, dates_m1, dates_oos,
              dates_hp, hp_years, output_path)
    portfolio_ml(chars, barra_cov, lambda_list, features, risk_free, wealth, pf_set, settings, dates_m2,
                 dates_oos, hp_years, output_path)
    multiperiod_ml(config_params, chars, barra_cov, lambda_list, risk_free, wealth, pf_set, settings, dates_m1,
                   dates_oos, dates_hp, hp_years, output_path)

