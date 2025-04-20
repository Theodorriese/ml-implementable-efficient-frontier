import os
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from a_return_prediction_functions import rff


def run_feature_importance_ret(chars, cluster_labels, features, settings, output_path, model_folder):
    """
    Estimate counterfactual predictions for return prediction models.

    Parameters:
        chars (pd.DataFrame): Characteristics data.
        cluster_labels (pd.DataFrame): Cluster definitions, with 'cluster' and 'characteristic' columns.
        features (list): List of features used in prediction.
        settings (dict): Settings dict with 'seed_no'.
        output_path (str): Where to save output CSV.
        model_folder (str): Folder with model_[h].pkl files.
    """
    results = []

    for h in tqdm(range(1, 13), desc="Processing Horizons", unit="horizon"):
        model_path = os.path.join(model_folder, f'model_{h}.pkl')

        if not os.path.exists(model_path):
            print(f"Model file not found for horizon {h}: {model_path}")
            continue

        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        pred_y = chars[['id', 'eom', f'ret_ld{h}']].rename(columns={f'ret_ld{h}': 'ret_pred'})
        ret_cf_data = pd.merge(pred_y, chars, on=['id', 'eom'], how='left')

        preds_list = [m['pred'] for m in model.values()]
        if preds_list:
            preds = pd.concat(preds_list, ignore_index=True)
            ret_cf_data = pd.merge(ret_cf_data, preds[['id', 'eom', 'pred']], on=['id', 'eom'], how='left')

        ret_cf_data = ret_cf_data.dropna(subset=['ret_pred'])

        clusters = sorted(cluster_labels['cluster'].unique())

        for cls in ['bm'] + clusters:
            np.random.seed(settings['seed_no'])

            if cls == 'bm':
                cf = ret_cf_data.copy()
            else:
                cf = ret_cf_data.copy()
                cf = cf.drop(columns=['pred'], errors='ignore')

                cf['id_shuffle'] = cf.groupby('eom')['id'].transform(
                    lambda x: x.sample(frac=1, random_state=settings['seed_no']).values
                )

                cls_chars = (
                    cluster_labels[
                        (cluster_labels['cluster'] == cls) &
                        (cluster_labels['characteristic'].isin(features))
                    ]['characteristic'].tolist()
                )

                cols = ['id', 'eom'] + cls_chars
                chars_data = cf[cols].copy().rename(columns={'id': 'id_shuffle'})

                cf = cf.drop(columns=cls_chars, errors='ignore')
                cf = pd.merge(cf, chars_data, on=['id_shuffle', 'eom'], how='left')

                for m in model.values():
                    if isinstance(m, dict) and isinstance(m.get('pred'), pd.DataFrame):
                        sub_dates = m['pred']['eom'].unique()
                        mask = cf['eom'].isin(sub_dates)
                        if mask.any():
                            cf_x = cf.loc[mask, features].values
                            rff_result = rff(cf_x, W=m['W'])
                            s = np.hstack([rff_result['X_cos'], rff_result['X_sin']])
                            s *= m['opt_hps']['p'] ** -0.5
                            predictions = m['fit'].predict(s)
                            cf.loc[mask, 'pred'] = predictions

            summary = cf.groupby('eom').apply(
                lambda g: pd.Series({
                    'cluster': cls,
                    'n': len(g),
                    'mse': np.mean((g['pred'] - g['ret_pred']) ** 2)
                })
            ).reset_index()
            summary['h'] = h
            summary = summary.dropna(subset=['mse'])
            results.append(summary)

    ret_cf = pd.concat(results, ignore_index=True)
    output_file = os.path.join(output_path, "ret_cf.csv")
    ret_cf.to_csv(output_file, index=False)
    print(f"Saved counterfactual prediction results to {output_file}")

    return ret_cf
