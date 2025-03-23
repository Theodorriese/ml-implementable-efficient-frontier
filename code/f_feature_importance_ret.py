import os
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm


def run_counterfactuals(chars, cluster_labels, features, settings, output_path):
    """
    Estimate counterfactual predictions.

    Parameters:
        chars (pd.DataFrame): Characteristics data.
        cluster_labels (pd.DataFrame): Cluster labels with associated characteristics.
        features (list): Feature names used for predictions.
        settings (dict): Settings containing seed information.
        output_path (str): Path to save the counterfactual predictions.
    """
    results = []

    for h in tqdm(range(1, 13), desc="Processing Horizons", unit="horizon"):
        model_path = os.path.join(output_path, f'model_{h}.pkl')

        # Load the model
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        # Prepare data
        pred_y = chars[['id', 'eom', f'ret_ld{h}']].rename(columns={f'ret_ld{h}': 'ret_pred'})
        ret_cf_data = pd.merge(pred_y, chars, on=['id', 'eom'], how='left')

        preds = pd.concat([m['pred'] for m in model if 'pred' in m])
        ret_cf_data = pd.merge(ret_cf_data, preds[['id', 'eom', 'pred']], on=['id', 'eom'], how='left')
        ret_cf_data = ret_cf_data.dropna(subset=['ret_pred'])

        # Implement counterfactual
        clusters = sorted(cluster_labels['cluster'].unique())
        for cls in ['bm'] + clusters:
            np.random.seed(settings['seed'])
            print(f"Processing Cluster: {cls}")

            if cls == 'bm':
                cf = ret_cf_data.copy()
            else:
                cf = ret_cf_data.drop(columns=['pred'])  # Counterfactual
                cf['id_shuffle'] = cf.groupby('eom')['id'].transform(lambda x: np.random.permutation(x))
                cls_chars = cluster_labels.loc[(cluster_labels['cluster'] == cls) & (cluster_labels['characteristic'].isin(features)), 'characteristic']
                chars_data = cf[['id_shuffle', 'eom'] + list(cls_chars)].rename(columns={'id_shuffle': 'id'})
                cf.drop(columns=cls_chars, inplace=True)
                cf = pd.merge(cf, chars_data, on=['id', 'eom'], how='left')

            cf['mse'] = (cf['pred'] - cf['ret_pred']) ** 2
            cf_summary = cf.groupby('eom').agg(cluster=(cls, 'first'), n=('id', 'size'), mse=('mse', 'mean'))
            cf_summary['h'] = h
            results.append(cf_summary)

    # Save results
    ret_cf = pd.concat(results).reset_index()
    ret_cf.to_pickle(os.path.join(output_path, "ret_cf.pkl"))
    print("Counterfactual predictions saved.")
