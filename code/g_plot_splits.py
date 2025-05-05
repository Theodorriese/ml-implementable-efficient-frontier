import pandas as pd
from pandas.tseries.offsets import DateOffset
import matplotlib.pyplot as plt
from i1_Main import settings

from matplotlib import rcParams
rcParams.update({
    'font.size': 14,           # Base font size
    'axes.titlesize': 15,      # Axes title
    'axes.labelsize': 15,      # Axes labels
    'xtick.labelsize': 15,     # X tick labels
    'ytick.labelsize': 15,     # Y tick labels
    'legend.fontsize': 15,     # Legend text
    'figure.titlesize': 18     # Figure title
})
colours_theme = ["steelblue", "darkorange", "gray"]


# Extract values from settings
val_years = settings["split"]["val_years"]
train_lookback = settings["split"]["train_lookback"]
retrain_lookback = settings["split"]["retrain_lookback"]
screen_start = settings["screens"]["start"]
train_end = settings["split"]["train_end"]
test_end = settings["split"]["test_end"]

# Generate all validation endpoints (end of each year)
val_ends = pd.date_range(start=train_end, end=test_end, freq="YE")

# Collect all segments for plotting
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

# Plotting
fig, ax = plt.subplots(figsize=(14, 14))
y_labels = ["Training", "Validation", "Retraining", "Test"]
colors = {"Training": "C0", "Validation": "C1", "Retraining": "C2", "Test": "C3"}

for i, seg in enumerate(all_segments):
    for j, label in enumerate(y_labels):
        start, end = seg[label]
        ax.plot([start, end], [j + 1 + i * 5] * 2, lw=6, color=colors[label])

# Adjust plot
ax.set_yticks([1 + i * 5 for i in range(len(all_segments))])
ax.set_yticklabels([str(seg["t"].year) for seg in all_segments])
ax.set_title("Rolling Window Splits Over Validation Years")
ax.set_xlabel("Time")
ax.grid(False)
ax.legend(handles=[plt.Line2D([0], [0], color=colors[l], lw=6, label=l) for l in y_labels])
plt.tight_layout()
plt.show()
