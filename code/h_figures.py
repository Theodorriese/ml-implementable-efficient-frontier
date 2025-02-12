import os
import matplotlib.pyplot as plt


def generate_output():
    """
    Dummy function to generate a dictionary of Matplotlib figures.
    Replace this with your actual figure generation logic.
    """
    figs = {}
    # Create a list of keys corresponding to all the figures referenced in save_all_figures
    figure_keys = [
        "ef", "ef_all", "ts", "comp_stats", "example", "feature_importance",
        "er_tuning", "portfolio_tuning", "fi_returns", "ar1", "cf_ef_tc",
        "cf_ef_markowitz", "by_size", "shorting", "w_liq", "simulations",
        "rff_specific0"
    ]

    # Generate dummy figures for each key
    for key in figure_keys:
        fig, ax = plt.subplots()
        ax.set_title(f"Dummy Figure for {key}")
        figs[key] = fig

    return figs


def save_figure(fig, filename, fig_w, fig_h, txt_size=None):
    """
    Saves a Matplotlib figure in PDF format.

    Parameters:
        fig (plt.Figure): Matplotlib figure object.
        filename (str): File path to save the figure.
        fig_w (float): Figure width in inches.
        fig_h (float): Figure height in inches.
        txt_size (int, optional): Text size for figure labels.
    """
    # Adjust text size if requested
    if txt_size:
        fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
        for ax in fig.axes:
            ax.tick_params(axis="both", labelsize=txt_size)

    # Set figure size
    fig.set_size_inches(fig_w, fig_h)
    # Save as PDF
    fig.savefig(f"{filename}.pdf", format="pdf", dpi=300, bbox_inches="tight")


def save_figure_format(fig, path, name, fmt, width, height):
    """
    Saves a figure in various formats.

    Parameters:
        fig (plt.Figure): Matplotlib figure object.
        path (str): Directory path to save the file.
        name (str): File name (without extension).
        fmt (str): File format (e.g., "pdf", "jpg", "tiff", "eps").
        width (float): Figure width in inches.
        height (float): Figure height in inches.
    """
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, name)

    # Set figure size
    fig.set_size_inches(width, height)

    if fmt == "pdf":
        fig.savefig(f"{file_path}.pdf", format="pdf", dpi=300, bbox_inches="tight")
    elif fmt == "jpg":
        fig.savefig(f"{file_path}.jpg", format="jpg", dpi=300, bbox_inches="tight")
    elif fmt == "eps":
        fig.savefig(f"{file_path}.eps", format="eps", dpi=300, bbox_inches="tight")
    elif fmt == "tiff":
        fig.savefig(f"{file_path}.tiff", format="tiff", dpi=500, bbox_inches="tight")


def save_all_figures(output, output_path="Figures", fig_w=8, fig_h=6, txt_size=12):
    """
    Saves all required figures from the output dictionary.

    Parameters:
        output (dict): Dictionary containing all generated figures.
        output_path (str): Path to save figures.
        fig_w (float): Figure width in inches.
        fig_h (float): Figure height in inches.
        txt_size (int): Text size for figure labels.
    """
    os.makedirs(output_path, exist_ok=True)

    # 1. Efficient Frontier
    save_figure(output["ef"], f"{output_path}/ief_by_wealth", fig_w, fig_h, txt_size)

    # 2. Efficient Frontier - All Methods
    save_figure(output["ef_all"], f"{output_path}/ief_by_method", fig_w, fig_h, txt_size)

    # 3. Portfolios: Cumulative Performance
    save_figure(output["ts"], f"{output_path}/cumret_pf", fig_w, fig_h)

    # 4. Portfolios: Stats Over Time
    save_figure(output["comp_stats"], f"{output_path}/stats_ts", fig_w, fig_h * 1.5, txt_size)

    # 5. Example: Apple vs. Xerox
    save_figure(output["example"], f"{output_path}/example_weights", fig_w, fig_h * 1.5, txt_size)

    # 6. Feature Importance
    save_figure(output["feature_importance"], f"{output_path}/feat_imp", fig_w, fig_h, txt_size)

    # 7. Optimal Hyper-parameters (Expected Return)
    save_figure(output["er_tuning"], f"{output_path}/optimal_hps_er", fig_w, fig_h)

    # 8. Optimal Hyper-parameters (Portfolio Tuning)
    save_figure(output["portfolio_tuning"], f"{output_path}/optimal_hps", fig_w, fig_h)

    # 9. Feature Importance - Returns
    fig = output["fi_returns"]
    # Equivalent to fig.tight_layout(), but done explicitly
    fig.set_tight_layout(True)
    fig.axes[0].tick_params(axis="x", rotation=-20, labelsize=8)
    fig.axes[0].tick_params(axis="y", labelsize=7)
    # Dummy legend call
    fig.legend(fontsize=5)
    save_figure(fig, f"{output_path}/fi_returns", fig_w, fig_h)

    # 10. Feature Autocorrelation
    save_figure(output["ar1"], f"{output_path}/feature_ar1", fig_w, fig_h * 2)

    # 11. Efficient Frontier with Trading Costs
    save_figure(output["cf_ef_tc"], f"{output_path}/ef_cf_tc", fig_w, fig_h)

    # 12. Efficient Frontier without Trading Costs
    save_figure(output["cf_ef_markowitz"], f"{output_path}/ef_cf_no_tc", fig_w, fig_h)

    # 13. Performance Across Size Distribution
    save_figure(output["by_size"], f"{output_path}/by_size", fig_w, fig_h * 1.5)

    # 14. Shorting Costs
    save_figure(output["shorting"], f"{output_path}/shorting", fig_w, fig_h)

    # 15. Relation Between Liquidity and Portfolio Weight
    save_figure(output["w_liq"], f"{output_path}/w_liq", fig_w, fig_h)

    # 16. Simulations
    save_figure(output["simulations"], f"{output_path}/simulations", fig_w, fig_h, txt_size)

    # 17. RFF Example (Special Handling)
    save_figure(output["rff_specific0"], f"{output_path}/rff_ex", fig_w, fig_h * 1.5)


def main():
    """
    Main function to generate and save figures.
    """
    output = generate_output()  # Replace with your real figure generation code
    save_all_figures(output, output_path="Figures", fig_w=8, fig_h=6, txt_size=12)


if __name__ == "__main__":
    main()
