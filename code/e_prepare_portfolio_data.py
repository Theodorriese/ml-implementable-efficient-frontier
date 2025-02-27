import pandas as pd
from datetime import timedelta
from dateutil.relativedelta import relativedelta


def add_return_predictions(chars, get_from_path_model, settings):
    """
    Add return predictions to the chars dataframe.

    Parameters:
        chars (pd.DataFrame): Characteristics dataframe.
        get_from_path_model (str): Path to the folder containing model data.
        settings (dict): Configuration settings.

    Returns:
        pd.DataFrame: Updated chars dataframe with return predictions.
    """
    for h in range(1, settings["pf"]["hps"]["m1"]["K"] + 1):
        model_path = f"{get_from_path_model}/model_{h}.pkl"
        model_data = pd.read_pickle(model_path)

        pred_list = []

        for val_end, model_entry in model_data.items():
            if isinstance(model_entry, dict) and "pred" in model_entry:
                pred_df = model_entry["pred"]

                if isinstance(pred_df, pd.DataFrame) and {"id", "eom", "pred"}.issubset(pred_df.columns):
                    pred_df = pred_df[["id", "eom", "pred"]].copy()
                    pred_df.rename(columns={"pred": f"pred_ld{h}"}, inplace=True)
                    pred_list.append(pred_df)

        if pred_list:
            pred_data = pd.concat(pred_list, ignore_index=True)
            chars = pd.merge(chars, pred_data, on=["id", "eom"], how="left")

    return chars


def create_lambda_list(chars):
    """
    Create a dictionary mapping lambda values to dates with `pd.Timestamp` keys.

    Parameters:
        chars (pd.DataFrame): Characteristics dataframe.

    Returns:
        dict: Dictionary of lambda values keyed by `pd.Timestamp` dates.
    """
    lambda_dates = chars["eom"].unique()
    return {
        pd.Timestamp(d): chars.loc[chars["eom"] == d, ["id", "lambda"]]
        .sort_values("id")
        .set_index("id")["lambda"]
        .to_dict()
        for d in lambda_dates
    }



def calculate_dates(settings, pf_set, barra_cov):
    """
    Calculate important date ranges for the portfolio.

    Parameters:
        settings (dict): Configuration settings.
        pf_set (dict): Portfolio settings.
        barra_cov (dict): Barra covariance data.

    Returns:
        dict: Dictionary of calculated date ranges.
    """
    first_cov_date = min(pd.to_datetime(list(barra_cov.keys())))
    hp_years = list(range(settings["pf"]["dates"]["start_year"], settings["pf"]["dates"]["end_yr"] + 1))
    start_oos = settings["pf"]["dates"]["start_year"] + settings["pf"]["dates"]["split_years"]

    dates_m1 = pd.date_range(
        start=settings["split"]["train_end"] + timedelta(days=1),
        end=settings["split"]["test_end"] + relativedelta(months=-1),
        freq="M"
    ) + pd.DateOffset(days=1)
    dates_m1 = dates_m1 - pd.offsets.MonthEnd(1)

    dates_m2 = pd.date_range(
        start=first_cov_date + relativedelta(months=(pf_set["lb_hor"] + 2)),
        end=settings["split"]["test_end"] + relativedelta(months=-1),
        freq="M"
    ) + pd.DateOffset(days=1)
    dates_m2 = dates_m2 - pd.offsets.MonthEnd(1)

    dates_oos = pd.date_range(
        start=pd.Timestamp(f"{start_oos}-01-01"),
        end=settings["split"]["test_end"] + relativedelta(months=-1),
        freq="M"
    ) + pd.DateOffset(days=1)
    dates_oos = dates_oos - pd.offsets.MonthEnd(1)

    dates_hp = pd.date_range(
        start=pd.Timestamp(f"{min(hp_years)}-01-01"),
        end=settings["split"]["test_end"] + relativedelta(months=-1),
        freq="M"
    ) + pd.DateOffset(days=1)
    dates_hp = dates_hp - pd.offsets.MonthEnd(1)

    return {
        "dates_m1": dates_m1,
        "dates_m2": dates_m2,
        "dates_oos": dates_oos,
        "dates_hp": dates_hp,
        "hp_years": hp_years
    }


def run_prepare_portfolio_data(chars, get_from_path_model, settings, pf_set, barra_cov):
    """
    Main function to execute portfolio preparation steps.

    Parameters:
        chars (pd.DataFrame): Characteristics dataframe.
        get_from_path_model (str): Path to the folder containing model data.
        settings (dict): Configuration settings.
        pf_set (dict): Portfolio settings.
        barra_cov (dict): Barra covariance data.

    Returns:
        dict: Dictionary containing updated chars, lambda list, and calculated dates.
    """
    print("Adding return predictions...")
    chars = add_return_predictions(chars, get_from_path_model, settings)

    print("Creating lambda list...")
    lambda_list = create_lambda_list(chars)

    print("Calculating important dates...")
    dates = calculate_dates(settings, pf_set, barra_cov)

    return {
        "chars": chars,
        "lambda_list": lambda_list,
        "dates": dates,
    }

