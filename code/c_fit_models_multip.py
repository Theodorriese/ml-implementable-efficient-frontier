from joblib import Parallel, delayed
from pathlib import Path
import pandas as pd
from a_return_prediction_functions import rff_hp_search, data_split

def fit_single_model(row, data_ret, chars, settings, features, output_path):
    """
    Train and save a predictive model for a given prediction horizon.

    Parameters:
        row (pd.Series): A row from the search grid, containing 'name' and 'horizon'.
        data_ret (pd.DataFrame): Return data with 'ret_ldX' columns.
        chars (pd.DataFrame): Stock characteristics with validity flags.
        settings (dict): Model settings, including split rules and hyperparameters.
        features (list): List of feature names to use.
        output_path (str): Path to save the trained model.

    Returns:
        tuple: (horizon_str, model dictionary)
    """
    h = row["horizon"] if isinstance(row["horizon"], list) else [row["horizon"]]
    pred_y = data_ret[[f"ret_ld{h_i}" for h_i in h]].mean(axis=1)
    pred_y = data_ret[["id", "eom_m"]].assign(
        eom_pred_last=(data_ret["eom_m"] + pd.DateOffset(months=max(h))).apply(
            lambda x: x.replace(day=1) + pd.offsets.MonthEnd(0)),
        ret_pred=pred_y
    )

    data_pred = pd.merge(pred_y, chars[chars["valid"]], left_on=["id", "eom_m"], right_on=["id", "eom"], how="right")
    print(f"Processing horizons: {h}")

    val_ends = pd.date_range(
        start=settings["split"]["train_end"],
        end=settings["split"]["test_end"],
        freq="YE"
    )
    test_inc = 1

    op = {}
    for val_end in val_ends:
        train_test_val = data_split(
            data=data_pred,
            val_end=val_end,
            val_years=settings["split"]["val_years"],
            train_start=settings["screens"]["start"],
            train_lookback=settings["split"]["train_lookback"],
            retrain_lookback=settings["split"]["retrain_lookback"],
            test_inc=test_inc,
            test_end=settings["split"]["test_end"]
        )

        model_op = rff_hp_search(
            data=train_test_val,
            feat=features,
            p_vec=settings["rff"]["p_vec"],
            g_vec=settings["rff"]["g_vec"],
            l_vec=settings["rff"]["l_vec"],
            seed=settings["seed_no"]
        )
        op[val_end] = model_op

    horizon_str = "_".join(map(str, h))
    output_file = Path(output_path) / f"model_{horizon_str}.pkl"
    pd.to_pickle(op, output_file)

    return horizon_str, op


def fit_models(search_grid, data_ret, chars, settings, features, output_path, n_jobs=-1):
    """
    Fit models for each combination of settings in the search grid using multiprocessing.

    Parameters:
        search_grid (pd.DataFrame): Grid with 'name' and 'horizon' columns.
        data_ret (pd.DataFrame): Return data with columns like 'ret_ld1', 'ret_ld3', etc.
        chars (pd.DataFrame): Characteristics data with a 'valid' boolean flag.
        settings (dict): Dictionary with hyperparameters and split rules.
        features (list): List of feature names.
        output_path (str): Path where models will be saved.
        n_jobs (int): Number of parallel jobs. Use -1 for all available cores.

    Returns:
        dict: Dictionary with keys as horizon strings and values as trained model dictionaries.
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(fit_single_model)(row, data_ret, chars, settings, features, output_path)
        for _, row in search_grid.iterrows()
    )

    models = dict(results)
    full_model_path = Path(output_path) / "model_full.pkl"
    pd.to_pickle(models, full_model_path)

    return models
