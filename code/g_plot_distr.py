import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import rcParams


def plot_combined_return_distributions(usa_monthly, usa_daily, clip_bounds=(-1, 1)):
    """
    Plot side-by-side histograms of monthly and daily stock returns.

    Parameters:
        usa_monthly (pd.DataFrame): DataFrame with 'RET' column (monthly returns).
        usa_daily (pd.DataFrame): DataFrame with 'RET' column (daily returns).
        clip_bounds (tuple): Bounds for clipping extreme outliers for visual clarity.
    """
    colours_theme = ["steelblue", "darkorange", "gray"]

    rcParams.update({
    'font.size': 14,
    'axes.titlesize': 15,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14,
    'figure.titlesize': 18
    })

    # Prepare monthly returns
    usa_monthly["RET"] = pd.to_numeric(usa_monthly["RET"], errors="coerce")
    monthly_returns = usa_monthly["RET"].dropna()
    monthly_clipped = monthly_returns.clip(*clip_bounds)
    monthly_stats = {
        'mean': monthly_returns.mean(),
        'std': monthly_returns.std(),
        'median': monthly_returns.median(),
        'min': monthly_returns.min(),
        'max': monthly_returns.max()
    }

    # Prepare daily returns
    usa_daily["RET"] = pd.to_numeric(usa_daily["RET"], errors="coerce")
    daily_returns = usa_daily["RET"].dropna()
    daily_clipped = daily_returns.clip(*clip_bounds)
    daily_stats = {
        'mean': daily_returns.mean(),
        'std': daily_returns.std(),
        'median': daily_returns.median(),
        'min': daily_returns.min(),
        'max': daily_returns.max()
    }

    # Print statistics
    print("\n--- Monthly Returns ---")
    for k, v in monthly_stats.items():
        print(f"{k.capitalize()}: {v:.4f}")

    print("\n--- Daily Returns ---")
    for k, v in daily_stats.items():
        print(f"{k.capitalize()}: {v:.4f}")

    # Plot setup
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    y_formatter = mtick.FuncFormatter(lambda x, _: f'{int(x):,}')

    # Monthly histogram
    axes[0].hist(monthly_clipped, bins=100, edgecolor='black', color=colours_theme[0])
    axes[0].axvline(monthly_stats['mean'], color='red', linestyle='dashed', linewidth=1.5,
                    label=f"Mean = {monthly_stats['mean']:.2%}")
    axes[0].set_title("Monthly Stock Returns")
    axes[0].set_xlabel("Monthly Return")
    axes[0].set_ylabel("Frequency")
    axes[0].xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[0].yaxis.set_major_formatter(y_formatter)
    axes[0].legend()

    # Daily histogram
    axes[1].hist(daily_clipped, bins=400, edgecolor='black', color=colours_theme[0])
    axes[1].axvline(daily_stats['mean'], color='red', linestyle='dashed', linewidth=1.5,
                    label=f"Mean = {daily_stats['mean']:.2%}")
    axes[1].set_title("Daily Stock Returns")
    axes[1].set_xlabel("Daily Return")
    axes[1].xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[1].yaxis.set_major_formatter(y_formatter)
    axes[1].set_xlim(-0.25, 0.25)
    axes[1].legend()

    plt.tight_layout()
    plt.show()
