import os

import numpy as np
import matplotlib.pyplot as plt

TUM_BLUE = (0 / 255, 101 / 255, 189 / 255)
TUM_ORAN = (227 / 255, 114 / 255, 34 / 255)
TUM_GREEN = (162 / 255, 173 / 255, 0 / 255)
TUM_BLACK = (0, 0, 0)
GRAY = (104 / 255, 104 / 255, 104 / 255)
BOUND_COL = (204 / 255, 204 / 255, 204 / 255)

COLUMNWIDTH = 3.5
PAGEWIDTH = COLUMNWIDTH * 2.0
FONTSIZE = 10  # IEEE is 8


def update_matplotlib(fontsize=None):
    """Update matplotlib font style."""
    if fontsize is None:
        fontsize = 10
    plt.rcParams.update(
        {
            "font.size": fontsize,
            "font.family": "Times New Roman",
            "text.usetex": True,
            "figure.autolayout": True,
            "xtick.labelsize": fontsize * 1.0,
            "ytick.labelsize": fontsize * 1.0,
        }
    )


def plt_rmse_over_horizon_model(rmse, opt_rmse, base_path, current_model):
    """Plots RMSE over time horizon."""
    plt.plot(np.arange(0.1, 5.05, 0.1), rmse, label="RMSE")
    if current_model == "g_sel":
        plt.plot(np.arange(0.1, 5.05, 0.1), opt_rmse, label="RMSE opt")

    plt.legend()

    _ = save_plt_fusion(
        base_path=base_path,
        file_name="{}_rmse_over_horizon.svg".format(current_model),
    )


def save_plt_fusion(base_path, file_name, file_type="svg"):
    """Saves plot to save_path."""
    save_path = os.path.join(base_path, "plots")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = os.path.join(save_path, file_name)
    try:
        plt.savefig(save_path, format=file_type)
    except Exception as er:
        print("\n\nNo plt.savefig, error: {}\n\n".format(er))
    plt.close()


def get_pretty_violin(
    ax,
    v_in,
    vi,
    col,
    plot_pairs=False,
    showmedians=False,
    quantiles=[],
    n_in=[],
):
    """Create violin plot either unified or splitted."""
    violin_parts = ax.violinplot(
        v_in,
        showextrema=False,
        showmedians=showmedians,
        quantiles=quantiles,
    )
    ax.set_title(None)
    ax.set_ylabel(vi[1])
    ax.grid(True)
    ax.set_xticks([1])

    ax.set_xticklabels([vi[2]])
    if plot_pairs:
        if len(n_in) > 1:
            print(
                vi[2]
                + ",\n$n_{\mathrm{points}}$ = "
                + ", ".join([str(nn) for nn in n_in])
            )
    else:
        print(vi[2] + "\n($n_{\mathrm{points}}$ = " + "{})".format(len(v_in)))

    for key, b in violin_parts.items():
        if isinstance(b, list):
            b = b[0]

        if plot_pairs:
            m = np.mean(b.get_paths()[0].vertices[:, 0])
            if plot_pairs == "right":
                # modify the paths to not go further right than the center
                b.get_paths()[0].vertices[:, 0] = np.clip(
                    b.get_paths()[0].vertices[:, 0], m, np.inf
                )
            elif plot_pairs == "left":
                b.get_paths()[0].vertices[:, 0] = np.clip(
                    b.get_paths()[0].vertices[:, 0], -np.inf, m
                )

        if key == "bodies":
            b.set_facecolor(col)
            b.set_edgecolor(col)
            b.set_alpha(1.0)
        else:
            b.set_edgecolor("k")
            b.set_linewidth(3)
