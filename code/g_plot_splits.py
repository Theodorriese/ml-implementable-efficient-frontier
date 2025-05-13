import matplotlib.pyplot as plt
import pandas as pd
from pandas.tseries.offsets import DateOffset
from matplotlib import rcParams


def plot_training_val_test_split(settings, output_path=None):
    """
    Visualize rolling window splits over validation years.

    Parameters:
        settings (dict): Dictionary containing 'split' and 'screens' configuration.
        output_path (str): Optional path to save the plot.
    """
    rcParams.update({
    'font.size': 14,
    'axes.titlesize': 15,
    'axes.labelsize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 15,
    'figure.titlesize': 18
    })

    val_years = settings["split"]["val_years"]
    train_lookback = settings["split"]["train_lookback"]
    retrain_lookback = settings["split"]["retrain_lookback"]
    screen_start = settings["screens"]["start"]
    train_end = settings["split"]["train_end"]
    test_end = settings["split"]["test_end"]

    val_ends = pd.date_range(start=train_end, end=test_end, freq="YE")
    all_segments = []

    for t in val_ends:
        train_start = max(t - DateOffset(years=train_lookback), screen_start)
        train_end_point = t - DateOffset(years=val_years)
        val_start = train_end_point
        val_end = t
        retrain_start = max(t - DateOffset(years=retrain_lookback), screen_start)
        retrain_end = t
        test_start = t
        test_end_point = t + DateOffset(years=1)

        all_segments.append({
            "t": t,
            "Training": (train_start, train_end_point),
            "Validation": (val_start, val_end),
            "Retraining": (retrain_start, retrain_end),
            "Test": (test_start, test_end_point),
        })

    fig, ax = plt.subplots(figsize=(14, 14))
    y_labels = ["Training", "Validation", "Retraining", "Test"]
    colors = {"Training": "C0", "Validation": "C1", "Retraining": "C2", "Test": "C3"}

    for i, seg in enumerate(all_segments):
        for j, label in enumerate(y_labels):
            start, end = seg[label]
            ax.plot([start, end], [j + 1 + i * 5] * 2, lw=6, color=colors[label])

    ax.set_yticks([1 + i * 5 for i in range(len(all_segments))])
    ax.set_yticklabels([str(seg["t"].year) for seg in all_segments])
    ax.set_title("Expanding Windows")
    ax.set_xlabel("Time")
    ax.grid(False)
    ax.legend(handles=[plt.Line2D([0], [0], color=colors[l], lw=6, label=l) for l in y_labels])
    plt.tight_layout()

    if output_path:
        fig.savefig(f"{output_path}/train_val_test_splits.png", bbox_inches="tight", dpi=300)

    plt.show()
