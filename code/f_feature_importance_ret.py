import os
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from a_return_prediction_functions import rff

def run_feature_importance_ret(chars, cluster_labels, features, settings, output_path, model_folder):
    """
    Estimate counterfactual predictions.

    Parameters:
        chars (pd.DataFrame): Characteristics data.
        cluster_labels (pd.DataFrame): DataFrame with columns like 'cluster' and 'characteristic'
                                       (and possibly others) that define clusters.
        features (list): Feature names used for predictions.
        settings (dict): Settings containing seed information (e.g., settings['seed']).
        output_path (str): Path where the counterfactual predictions CSV will be saved.
        model_folder (str): Folder path containing the trained model pickle files.
    """
    results = []

    # Loop over horizons
    for h in tqdm(range(1, 13), desc="Processing Horizons", unit="horizon"):
        model_path = os.path.join(model_folder, f'model_{h}.pkl')

        # Load the model for the current horizon
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        pred_y = chars[['id', 'eom', f'ret_ld{h}']].rename(columns={f'ret_ld{h}': 'ret_pred'})
        ret_cf_data = pd.merge(pred_y, chars, on=['id', 'eom'], how='left')

        # Combine predictions from all sub-models in the loaded model that contain a 'pred' key.
        preds_list = []
        for key, m in model.items():
            if isinstance(m, dict) and 'pred' in m:
                preds_list.append(m['pred'])
        if preds_list:
            preds = pd.concat(preds_list, ignore_index=True)
            ret_cf_data = pd.merge(ret_cf_data, preds[['id', 'eom', 'pred']], on=['id', 'eom'], how='left')

        ret_cf_data = ret_cf_data.dropna(subset=['ret_pred'])
        pred_columns = [f'pred_ld{h}' for h in range(1, 13)]
        mask = ret_cf_data[pred_columns].isna().all(axis=1)
        ret_cf_data = ret_cf_data.loc[~mask]

        clusters = sorted(cluster_labels['cluster'].unique())

        # Process benchmark ("bm") and each cluster separately
        for cls in ['bm'] + clusters:
            np.random.seed(settings['seed_no'])

            if cls == 'bm':
                # For benchmark, use the data as is.
                cf = ret_cf_data.copy()
            else:
                # For a counterfactual cluster:
                cf = ret_cf_data.copy()
                # Remove the 'pred' column (if present)
                if 'pred' in cf.columns:
                    cf = cf.drop(columns=['pred'])

                # Shuffle the 'id' within each 'eom' group.
                # Here we add a new column 'id_shuffle' which holds a random permutation of 'id'
                def shuffle_group(x):
                    return x.sample(frac=1, replace=False, random_state=settings['seed_no']).values
                cf['id_shuffle'] = cf.groupby('eom')['id'].transform(shuffle_group)

                # Determine which characteristics to replace based on the cluster.
                cls_chars = (cluster_labels[(cluster_labels['cluster'] == cls) &
                                            (cluster_labels['characteristic'].isin(features))]
                             ['characteristic'].tolist())

                cols = ['id', 'eom'] + cls_chars
                chars_data = cf[cols].copy().rename(columns={'id': 'id_shuffle'})

                cf = cf.drop(columns=cls_chars)
                cf = pd.merge(cf, chars_data, on=['id_shuffle', 'eom'], how='left')

                # For each sub-model in the loaded model, update the expected returns.
                for key, m_sub in model.items():
                    if isinstance(m_sub, dict) and 'pred' in m_sub and m_sub['pred'] is not None:
                        sub_dates = m_sub['pred']['eom'].unique()
                        mask = cf['eom'].isin(sub_dates)
                        if mask.sum() > 0:
                            cf_x = cf.loc[mask, features].values
                            rff_result = rff(cf_x, W=m_sub['W'])
                            s = np.hstack([rff_result['X_cos'], rff_result['X_sin']])
                            factor = m_sub['opt_hps']['p'] ** (-0.5)
                            s = factor * s
                            predictions = m_sub['fit'].predict(s)
                            cf.loc[mask, 'pred'] = predictions

            summary = cf.groupby('eom').apply(lambda g: pd.Series({
                'cluster': cls,
                'n': len(g),
                'mse': np.mean((g['pred'] - g['ret_pred'])**2)
            })).reset_index()
            summary['h'] = h
            results.append(summary)

    # Combine all results (across horizons and clusters) and save to CSV.
    ret_cf = pd.concat(results, ignore_index=True)
    ret_cf.to_pickle(os.path.join(output_path, "ret_cf.pkl"))



