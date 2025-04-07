import pandas as pd
import os

# Path to your saved model file
model_path = r"C:\Master\Outputs\fitted_models.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

models = pd.read_pickle(model_path)
print(f"✅ Loaded models from {model_path}")

# Store all predictions into a single DataFrame
all_preds = []

for horizon_key, model_group in models.items():
    print(f"\n🔍 Horizon: {horizon_key}")

    for val_end, model_obj in model_group.items():
        if not isinstance(model_obj, dict):
            continue

        test_df = model_obj.get("test", None)
        if isinstance(test_df, pd.DataFrame) and "pred" in test_df.columns:
            df = test_df.copy()
            df["horizon"] = int(horizon_key)
            df["val_end"] = pd.to_datetime(val_end)
            all_preds.append(df[["id", "eom_m", "pred", "val_end", "horizon"]])

# Combine all into a single dataframe
if not all_preds:
    print("⚠️ No predictions found in any model.")
else:
    pred_df = pd.concat(all_preds, ignore_index=True)
    pred_df["eom_m"] = pd.to_datetime(pred_df["eom_m"])

    print(f"\n✅ Total predictions: {len(pred_df):,}")
    print("\n📋 Sample predictions:")
    print(pred_df.sample(10).sort_values("eom_m"))

    # Optional: Save to CSV for inspection
    pred_df.to_csv("model_predictions_sample.csv", index=False)
    print("\n📁 Saved predictions sample to 'model_predictions_sample.csv'")
