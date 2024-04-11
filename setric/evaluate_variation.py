"""Evaluation variation of error threshold."""

import os
import sys
import argparse
import pickle

import numpy as np
import matplotlib.pyplot as plt

repo_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(repo_path)

from setric.evaluate_fusion import eval_main
from utils.visualization import update_matplotlib
from utils.processing import Namespace

TUM_BLUE = (0 / 255, 101 / 255, 189 / 255)
TUM_LIGHT_BLUE = (100 / 255, 160 / 255, 200 / 255)
TUM_LIGHTER_BLUE = (152 / 255, 198 / 255, 234 / 255)
TUM_ORAN = (227 / 255, 114 / 255, 34 / 255)
TUM_GREEN = (162 / 255, 173 / 255, 0 / 255)
TUM_BLACK = (0, 0, 0)
GRAY = (104 / 255, 104 / 255, 104 / 255)
BOUND_COL = (204 / 255, 204 / 255, 204 / 255)

COLOR_LIST = [GRAY, TUM_BLACK, TUM_ORAN, TUM_GREEN, TUM_BLUE]
# order: benchmark, cv, l_lstm, dg_lstm, g_sel
COLOR_LIST_RED = [GRAY, TUM_ORAN, TUM_GREEN, TUM_BLUE]

COLUMNWIDTH = 3.5
PAGEWIDTH = COLUMNWIDTH * 2.0
FONTSIZE = 7  # IEEE is 8
FIGSIZE = (16, 9)


def save_plt(base_path, plt_key, ax, with_legend=True):
    """Save plot to save_path."""
    update_matplotlib(fontsize=FONTSIZE)

    # save wo legend
    save_path = os.path.join(base_path, plt_key + ".svg")
    plt.savefig(save_path, format="svg")
    plt.savefig(save_path.replace(".svg", ".pdf"), format="pdf")

    if with_legend:
        ax.legend(fontsize=FONTSIZE)
        # save w legend
        save_path = os.path.join(base_path, plt_key + "_legend.svg")
        plt.savefig(save_path, format="svg")
        plt.savefig(save_path.replace(".svg", ".pdf"), format="pdf")

    return save_path


def plot_rmse_distributions(
    input_tuples,
    plt_path,
    plt_key="rmse_over_samples",
    step_size=1.0,
    n_bin=21,
    is_logbins=True,
):
    """Evaluate distribution of RMSE error over samples."""
    strstr = ""
    for j, (items, model_tag) in enumerate(input_tuples):
        dist_values = items[plt_key]
        if len(dist_values.shape) == 0:
            continue
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=FIGSIZE)
        if is_logbins:
            bin_min = np.floor(np.log10(min(dist_values) + 1e-3))
            bin_max = max(2.0, np.ceil(np.log10(max(dist_values))))
            plt_bins = np.logspace(bin_min, bin_max, n_bin)
        else:
            max_err = max(20, int(max(dist_values)) + 1)
            plt_bins = np.arange(0.001, max_err + step_size / 2, step_size)

        for n_i in range(len(items["10q_" + plt_key])):
            qq = np.round(0.1 + n_i / 10.0, 1)
            val = items["10q_" + plt_key][n_i]
            strstr += "\bModel: {} | q{} : {:.02f} m \n".format(
                model_tag.upper(), qq, val
            )
        strstr += "ADE = {:.02f} m, FDE = {:.02f} m, FDE / ADE = {:.02f} \n".format(
            items["rmse_over_horizon"].mean(),
            items["rmse_over_horizon"][-1],
            items["rmse_over_horizon"][-1] / items["rmse_over_horizon"].mean(),
        )

        plt.hist(dist_values, bins=plt_bins, color=COLOR_LIST[j], ec=GRAY)
        plt.xscale("log")

        ax.set_xlabel(
            "RMSE in m",
        )
        ax.set_ylabel(
            "$N_{\mathrm{abs}}$",
        )

        save_path = save_plt(
            plt_path, plt_key + "_dist_" + model_tag, ax, with_legend=False
        )

    assert input_tuples[0][1] == "benchmark"
    assert input_tuples[-1][1] == "g_sel"

    [ip[0] for ip in input_tuples[1:-1]]
    rmse_over_sample_stack = np.stack(
        [ip[0]["rmse_over_samples"] for ip in input_tuples[1:-1]]
    )

    min_vals = np.clip(np.min(rmse_over_sample_stack, axis=0), 1e-4, 10000)
    second_min_vals = np.clip(
        [
            np.min(
                rmse_over_sample_stack[
                    rmse_over_sample_stack[:, kk] != min_vals[kk], kk
                ]
            )
            for kk in range(rmse_over_sample_stack.shape[1])
        ],
        1e-4,
        10000,
    )
    max_vals = np.max(rmse_over_sample_stack, axis=0)

    strstr += "\nDeviation between max and min predictor: mean {:.03f} m, std {:.03f} m\n".format(
        np.mean((max_vals - min_vals)), np.std((max_vals - min_vals))
    )
    strstr += "\nDeviation between second min and min predictor: mean {:.03f} m, std {:.03f} m\n".format(
        np.mean((second_min_vals - min_vals)), np.std((second_min_vals - min_vals))
    )
    strstr += "\nmin vals: mean {:.03f} m, std {:.03f} m\n".format(
        np.mean(min_vals), np.std(min_vals)
    )

    for j in range(len(input_tuples) - 1):
        strstr += "Quantile vs g_sel: MODEL = {}, RED Q0.75 = {:.04f}, RED Q0.8 = {:.04f}, REDQ0.9 = {:.04f}\n".format(
            input_tuples[j][1].upper(),
            1
            - (
                np.quantile(input_tuples[-1][0]["rmse_over_samples"], 0.75)
                / np.quantile(input_tuples[j][0]["rmse_over_samples"], 0.75)
            ),
            1
            - (
                np.quantile(input_tuples[-1][0]["rmse_over_samples"], 0.8)
                / np.quantile(input_tuples[j][0]["rmse_over_samples"], 0.8)
            ),
            1
            - (
                np.quantile(input_tuples[-1][0]["rmse_over_samples"], 0.9)
                / np.quantile(input_tuples[j][0]["rmse_over_samples"], 0.9)
            ),
        )

    strstr += "MissRate_2_1: "
    for vals, fm_key in input_tuples:
        strstr += "{} = {:.03f} % | ".format(fm_key.upper(), vals["miss2rate"] * 100.0)
    strstr += "\n\n"

    with open(
        os.path.join(os.path.dirname(save_path), "00_data_1.txt"), "w", encoding="utf-8"
    ) as ff:
        ff.write(strstr)


def plot_RMSE_over_horizon(
    input_tuples,
    plt_path,
    plt_key="rmse_over_horizon",
    dt_s=0.1,
    show=False,
    with_str=True,
):
    """Plot RMSE over time steps."""
    strstr = ""
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=FIGSIZE)

    for j, (items, model_tag) in enumerate(input_tuples):
        t_list = np.arange(0, len(items[plt_key]) * dt_s, step=dt_s) + dt_s
        values = items[plt_key]
        ax.plot(
            t_list,
            values,
            label=model_tag.upper(),
            color=COLOR_LIST[j],
        )
        strstr += "Model: {} | RMSE: {:.02f} m | MissRate_2_1: {:.02f} % \n".format(
            model_tag.upper(), values.mean(), items.get("miss2rate", 0.00) * 100
        )

    ax.grid("True")

    ax.set_xlim(0)
    ax.set_ylim(0)

    ax.set_xlabel(
        "$t_{\mathrm{pred}}$ in s",
    )
    ax.set_ylabel("RMSE in m")
    plt.tight_layout()

    save_path = save_plt(plt_path, plt_key + "_" + "n_{}".format(len(input_tuples)), ax)

    if show:
        plt.show()
    plt.close()

    if with_str:
        # write to .txt file
        with open(
            os.path.join(os.path.dirname(save_path), "00_data_2.txt"),
            "w",
            encoding="utf-8",
        ) as ff:
            ff.write(strstr)

    return


def plot_boxplots(
    input_tuples, plt_path, plt_key="rmse_over_samples", exclude_list=["cv"]
):
    """Plot RMSE over time steps."""
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=FIGSIZE)

    data = [items[0][plt_key] for items in input_tuples if items[1] not in exclude_list]
    model_tag = [
        items[1].upper() for items in input_tuples if items[1] not in exclude_list
    ]

    bp = ax.boxplot(data, showfliers=False, patch_artist=True)
    ax.set_xticklabels(model_tag)

    # changing color and linewidth of
    # medians
    for _, median in enumerate(bp["medians"]):
        median.set(color=(1, 1, 1))

    # fill with colors
    for patch, color in zip(bp["boxes"], COLOR_LIST_RED):
        patch.set_facecolor(color)

    ax.grid("True")

    ax.set_ylabel("RMSE in m")
    plt.tight_layout()

    _ = save_plt(plt_path, "box_plot", ax, with_legend=False)

    plt.close()

    return


def visualization_vs_benchmark(args):
    """Visualizes one fusion model vs. benchmark."""
    plt_path = os.path.join(args.path, "result_plots")
    if not os.path.exists(plt_path):
        os.mkdir(plt_path)

    # Load benchmark model
    with open(args.benchmark, "rb") as fp:
        benchmark_model = pickle.load(fp)

    # Load benchmark model
    with open(os.path.join(args.path, "evaluation_results.pkl"), "rb") as fp:
        fusion_model = pickle.load(fp)

    # Plot RMSE over t
    # Also write values to txt for final evaluation
    for data_tag in benchmark_model:
        input_tuples = [(benchmark_model[data_tag]["wale-net"], "benchmark")]

        for sub_model in fusion_model[data_tag].keys():
            input_tuples.append((fusion_model[data_tag][sub_model], sub_model))

    for nn in range(len(input_tuples)):
        plot_RMSE_over_horizon(
            input_tuples[: nn + 1],
            plt_path=plt_path,
            with_str=(nn == len(input_tuples) - 1),
        )

    # Plot distribution
    plot_rmse_distributions(input_tuples, plt_path=plt_path)

    # Plot box plots
    plot_boxplots(input_tuples, plt_path=plt_path)

    # Plot Confusion Matrix
    # See in 'plots' folder, extract from there


def visualization_selectors(sel_model_list):
    """Visualizes selector results."""
    data_tag_list = None
    plt_path = None
    selector_values = {}
    strstr = ""
    iter_keys = ["e_0", "e_5", "e_10"]
    rmse_keys = ["rmse", "rmse_opt"]

    for threshold_val, selector_path in sel_model_list:
        # get plt path
        if plt_path is None:
            plt_path = os.path.join(os.path.dirname(selector_path), "00_g_sel_results")
            if not os.path.exists(plt_path):
                os.mkdir(plt_path)

        # Load benchmark model
        with open(os.path.join(selector_path, "evaluation_results.pkl"), "rb") as fp:
            fusion_model = pickle.load(fp)

        if data_tag_list is None:
            data_tag_list = list(fusion_model.keys())
            selector_values = {data_tag: {} for data_tag in data_tag_list}
            sel_val_plt = {
                data_tag: {i_k: [] for i_k in iter_keys + rmse_keys}
                for data_tag in data_tag_list
            }
            {
                data_tag: {i_k: [] for i_k in iter_keys + rmse_keys}
                for data_tag in data_tag_list
            }
            for data_tag in data_tag_list:
                sel_val_plt[data_tag]["rmse_single"] = 1e6

        for data_tag in data_tag_list:
            selector_values[data_tag][threshold_val] = fusion_model[data_tag]["g_sel"]

            sel_val_plt[data_tag]["e_0"].append(
                selector_values[data_tag][threshold_val]["num_correct_selections"]
                .cpu()
                .detach()
                .numpy()
                * 100.0
            )
            sel_val_plt[data_tag]["e_5"].append(
                selector_values[data_tag][threshold_val]["num_correct_selections_05"]
                .cpu()
                .detach()
                .numpy()
                * 100.0
            )
            sel_val_plt[data_tag]["e_10"].append(
                selector_values[data_tag][threshold_val]["num_correct_selections_10"]
                .cpu()
                .detach()
                .numpy()
                * 100.0
            )
            sel_val_plt[data_tag]["rmse"].append(
                selector_values[data_tag][threshold_val]["rmse_over_samples"].mean()
            )
            sel_val_plt[data_tag]["rmse_opt"].append(
                selector_values[data_tag][threshold_val]["opt_rmse_over_samples"].mean()
            )
            sel_val_plt[data_tag]["rmse_single"] = min(
                sel_val_plt[data_tag]["rmse_single"],
                min(
                    [
                        val["opt_rmse_over_samples"].mean()
                        for key, val in fusion_model[data_tag].items()
                        if key != "g_sel"
                    ]
                ),
            )

            predictions_stack = np.stack(
                (
                    fusion_model[data_tag]["cv"]["rmse_over_samples"],
                    fusion_model[data_tag]["l_lstm"]["rmse_over_samples"],
                    fusion_model[data_tag]["dg_lstm"]["rmse_over_samples"],
                    fusion_model[data_tag]["dg_lstm"]["rmse_over_samples"] * 0.0 - 1.0,
                )
            ).T

            predictions = np.array(
                [np.random.choice(pred_i) for pred_i in predictions_stack]
            )
            sel_val_plt[data_tag]["rmse_random"] = np.mean(
                predictions[predictions > -0.5]
            )

            strstr += "\n\nMissRate_2_1: "
            for fm_key in fusion_model[data_tag].keys():
                fusion_model[data_tag][fm_key]["miss2rate"]
                strstr += "{} = {:.03f} % | ".format(
                    fm_key.upper(), fusion_model[data_tag][fm_key]["miss2rate"] * 100.0
                )

        # Plot Selections
        strstr += "Threshold: {:.02f} | correct: {:.02f} %, RMSE: {:.02f} m, OPT_RMSE: {:.02f} m | 0%: {:.02f} % | 5%: {:.02f} % | 10%: {:.02f} %\n".format(
            threshold_val,
            selector_values[data_tag][threshold_val]["num_correct_selections"] * 100,
            selector_values[data_tag][threshold_val]["rmse_over_horizon"].mean(),
            selector_values[data_tag][threshold_val]["opt_rmse_over_horizon"].mean(),
            selector_values[data_tag][threshold_val].get(
                "num_correct_selections_00", -1
            )
            * 100,
            selector_values[data_tag][threshold_val]["num_correct_selections_05"] * 100,
            selector_values[data_tag][threshold_val]["num_correct_selections_10"] * 100,
        )

    # plot correct selections over qunatile (with deltas)
    labels = [r"$\Delta_{0\%}$", r"$\Delta_{5\%}$", r"$\Delta_{10\%}$"]
    marker_list = ["o", "x", "d"]
    color_list_2 = [TUM_BLUE, TUM_LIGHT_BLUE, TUM_LIGHTER_BLUE]
    for data_tag in data_tag_list:
        th_vals = list(selector_values[data_tag].keys())

        for nn in range(len(iter_keys)):
            _, ax = plt.subplots(nrows=1, ncols=1, figsize=FIGSIZE)
            for kk, iter_k in enumerate(iter_keys[: nn + 1]):
                plt.plot(
                    th_vals,
                    sel_val_plt[data_tag][iter_k],
                    label=labels[kk],
                    color=color_list_2[kk],
                    marker=marker_list[kk],
                    linestyle="solid",
                )

            ax.grid("True")
            ax.set_xlim(min(th_vals) - 0.01, max(th_vals) + 0.01)
            ax.set_ylim(80, 100)
            ax.set_xticks(th_vals)
            # ax.set_ylim(0, 100)
            ax.set_xlabel(
                r"$\varepsilon_{\mathrm{rel}}$",
            )
            ax.set_ylabel(r"$\Phi$ in \%")
            plt.tight_layout()
            save_path = save_plt(plt_path, "g_sel" + "_" + data_tag + "_" + iter_k, ax)

    # plot RMSE over quantile vs optimal selections
    labels = [r"G_SEL", r"OPTIMAL"]
    marker_list = ["o", "s"]
    color_list_2 = [TUM_BLUE, (254 / 255, 215 / 255, 2 / 255)]
    for data_tag in data_tag_list:
        th_vals = list(selector_values[data_tag].keys())
        for nn in range(len(iter_keys)):
            _, ax = plt.subplots(nrows=1, ncols=1, figsize=FIGSIZE)
            for kk, iter_k in enumerate(rmse_keys[: nn + 1]):
                plt.plot(
                    th_vals,
                    sel_val_plt[data_tag][iter_k],
                    label=labels[kk],
                    color=color_list_2[kk],
                    marker=marker_list[kk],
                    linestyle="solid",
                )
            plt.plot(
                [th_vals[0], th_vals[-1]],
                [
                    sel_val_plt[data_tag]["rmse_single"],
                    sel_val_plt[data_tag]["rmse_single"],
                ],
                color=TUM_ORAN,
                label="L_LSTM",
            )
            plt.plot(
                [th_vals[0], th_vals[-1]],
                [
                    sel_val_plt[data_tag]["rmse_random"],
                    sel_val_plt[data_tag]["rmse_random"],
                ],
                color=GRAY,
                label="RANDOM",
            )

            ax.grid("True")
            ax.set_xlim(min(th_vals) - 0.01, max(th_vals) + 0.01)
            ax.set_xticks(th_vals)
            ax.set_ylim(0)
            ax.set_xlabel(
                r"$\varepsilon_{\mathrm{rel}}$",
            )
            ax.set_ylabel(r"RMSE in m")
            plt.tight_layout()
            save_path = save_plt(plt_path, "g_sel" + "_" + data_tag + "_" + iter_k, ax)

    # write to .txt file
    with open(
        os.path.join(os.path.dirname(save_path), "00_data.txt"), "w", encoding="utf-8"
    ) as ff:
        ff.write(strstr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=str,
        default=os.path.join(
            repo_path, "results_benchmark", "evaluation_results_wale_net.pkl"
        ),
        help="path to stored training",
    )

    args = parser.parse_args()

    # must be sorted ascending (ascending quantil)
    # first entry: quantil of error threshhold (between 0 and zero)
    # second entry: Path to trained model
    # if len of list == 1, i.e. only the default selector is given, it will evaluated seperatly
    sel_model_list = [
        (
            0.8,
            os.path.join(repo_path, "results_variation", "cr_fusion_08"),
        ),
        (
            0.85,
            os.path.join(repo_path, "results_variation", "cr_fusion_085"),
        ),
        (
            0.9,
            os.path.join(repo_path, "results_variation", "cr_fusion_09"),
        ),
        (
            0.95,
            os.path.join(repo_path, "results_variation", "cr_fusion_095"),
        ),
        (
            1.0,
            os.path.join(repo_path, "results_variation", "cr_fusion_1"),
        ),
    ]

    args.path = sel_model_list[0][1]

    for base_path_in in [sm[1] for sm in sel_model_list]:
        if os.path.exists(os.path.join(base_path_in, "evaluation_results.pkl")):
            continue
        args_eval = Namespace(debug=False, path=base_path_in)
        eval_main(args=args_eval)

    update_matplotlib(fontsize=FONTSIZE)
    visualization_vs_benchmark(args=args)

    update_matplotlib(fontsize=FONTSIZE)
    visualization_selectors(sel_model_list=sel_model_list)
