import pandas as pd
from pathlib import Path
import time
from a_return_prediction_functions import rff_hp_search, data_split


def fit_models(search_grid, data_ret, chars, settings, features, output_path):
    """
    Fit models based on the given search grid and settings.

    Parameters:
        search_grid (pd.DataFrame): DataFrame with 'name' and 'horizon' columns.
        data_ret (pd.DataFrame): DataFrame containing return data.
        chars (pd.DataFrame): DataFrame with characteristics and validity flags.
        settings (dict): Dictionary containing settings for model fitting.
        features (list): List of feature columns.
        output_path (str): Path to save model outputs.

    Returns:
        dict: Dictionary containing models for each grid.
    """
    start_time = time.process_time()
    models = {}

    for i, row in search_grid.iterrows():
        # Prepare y variable
        h = row["horizon"] if isinstance(row["horizon"], list) else [row["horizon"]]
        pred_y = data_ret[[f"ret_ld{h_i}" for h_i in h]].mean(axis=1)
        pred_y = data_ret[["id", "eom_m"]].assign(  # Use eom_m here instead of eom

            # Add `max(h)` months to `eom_m` and adjust to the last day of the resulting month
            eom_pred_last=(data_ret["eom_m"] + pd.DateOffset(months=max(h))).apply(
                lambda x: x.replace(day=1) + pd.offsets.MonthEnd(0)),
            ret_pred=pred_y
        )

        data_pred = pd.merge(pred_y, chars[chars["valid"]], left_on=["id", "eom_m"], right_on=["id", "eom"],
                             how="right")

        print(f"Processing horizons: {h}")

        # Define validation endpoints and test increment
        val_ends, test_inc = None, None  # Ensure initialization

        if settings["split"]["model_update_freq"] == "once":
            val_ends = [settings["split"]["train_end"]]
            test_inc = 1000
        elif settings["split"]["model_update_freq"] == "yearly":
            val_ends = pd.date_range(
                start=settings["split"]["train_end"],
                end=settings["split"]["test_end"] - pd.DateOffset(years=settings["split"]["val_years"]),
                freq="Y"
            )
            test_inc = 1
        elif settings["split"]["model_update_freq"] == "decade":
            val_ends = pd.date_range(
                start=settings["split"]["train_end"],
                end=settings["split"]["test_end"],
                freq="10Y"
            )
            test_inc = 10

        if val_ends is None or test_inc is None:
            raise ValueError("Validation endpoints or test increment not properly initialized.")

        op = {}

        # Train models for each validation endpoint
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

            # Fit models
            print("Fitting models...")
            model_op = rff_hp_search(
                data=train_test_val,
                feat=features,
                p_vec=settings["rff"]["p_vec"],
                g_vec=settings["rff"]["g_vec"],
                l_vec=settings["rff"]["l_vec"],
                seed=settings["seed_no"]
            )
            op[val_end] = model_op

        # Save model output for the current horizon
        horizon_str = "_".join(map(str, h))
        output_file = Path(output_path) / f"model_{horizon_str}.pkl"
        pd.to_pickle(op, output_file)
        models[horizon_str] = op

    # Save the full model output
    full_model_path = Path(output_path) / "model_full.pkl"
    pd.to_pickle(models, full_model_path)

    elapsed_time = time.process_time() - start_time
    print(f"Total runtime: {elapsed_time:.2f} seconds")

    return models
