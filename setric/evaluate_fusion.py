"""Evaluation of fusion model."""

import os
import sys
import copy
import time
import json
import pickle
import argparse
import datetime

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sn
import pandas as pd
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix
from torch_geometric import seed_everything
from utils.scheduling import print_overview, print_quantiles, adjust_batch_size
from utils.map_processing import get_rdp_map, get_sc_img_batch

from utils.Dataset_OpenDD import Dataset_OpenDD
from utils.Dataset_CR import Dataset_CR
from utils.metrics import (
    get_mse_over_horizon,
    get_rmse_over_samples,
    get_mis_pred_over_samples,
)
from utils.scheduling import permute_input
from utils.processing import cr_labels
from utils.visualization import (
    save_plt_fusion,
    plt_rmse_over_horizon_model,
    update_matplotlib,
)
from models.Fusion_Model import Fusion_Model

TUM_BLUE = (0 / 255, 101 / 255, 189 / 255)
TUM_LIGHT_BLUE = (100 / 255, 160 / 255, 200 / 255)
TUM_LIGHTER_BLUE = (152 / 255, 198 / 255, 234 / 255)
TUM_ORAN = (227 / 255, 114 / 255, 34 / 255)
TUM_GREEN = (162 / 255, 173 / 255, 0 / 255)
TUM_BLACK = (0, 0, 0)
GRAY = (104 / 255, 104 / 255, 104 / 255)
BOUND_COL = (204 / 255, 204 / 255, 204 / 255)

COLUMNWIDTH = 3.5
PAGEWIDTH = COLUMNWIDTH * 2.0
FIGSIZE = (PAGEWIDTH * 0.7, 0.7 * 9 / 16 * PAGEWIDTH)
FONTSIZE = 9  # IEEE is 8
update_matplotlib(fontsize=FONTSIZE)


def eval_main(args, model_seed_list=[10], get_all_res_dict=False):
    """Evaluate fusion model on dataset (CR or OpenDD)."""
    dataset, train_test_zip, cfg_train, net_config = prepare_val(args)

    all_res = {}
    for train_split, test_split in train_test_zip:
        for seed in model_seed_list:
            # write to file and to terminal
            orig_stdout = sys.stdout
            sys.stdout = Logger(
                file_path=os.path.join(cfg_train["path"], "eval_logs.txt")
            )

            # seed and get dict key to store results
            seed_everything(seed)
            dict_key = get_dict_key(
                train_split, test_split, seed, is_opendd="open" in cfg_train["data"]
            )
            all_res[dict_key] = {}

            cfg_train.update({"seed": seed})

            # get dataloader and model path
            (
                test_dataloader,
                processed_file_folder,
                model_path,
                rdp_map_dict,
            ) = get_val_content(dataset, cfg_train, train_split, test_split)

            # load model from model path with cfg_train + net_config
            g_fusion = get_fusion_model(model_path, cfg_train, net_config)
            g_fusion.eval()

            # iter through models
            iterator = copy.deepcopy(g_fusion.model_tag_list)
            if not cfg_train.get("no_g_sel", False):
                iterator.append("g_sel")
            for model_tag in iterator:
                g_fusion.update_model_tag(model_tag)

                result_dict = evaluate_model(
                    g_fusion,
                    test_dataloader,
                    train_split,
                    test_split,
                    cfg_train,
                    rdp_map_dict,
                    processed_file_folder,
                )

                all_res[dict_key][model_tag] = result_dict

            if args.miss_rates:
                plt_miss_rate_eval(data_tag_dict=all_res[dict_key], cfg_train=cfg_train)

            # put back the original state of stdout
            sys.stdout = orig_stdout

    _ = save_eval(cfg_train["path"], all_res)

    if get_all_res_dict:
        return all_res


def plt_miss_rate_eval(data_tag_dict, cfg_train):
    """Plot miss rate."""
    base_path = os.path.abspath(cfg_train["path"])
    plt_keys = [
        (
            "mr_t_pred",
            "num_miss2_var_dict",
            "$t_{\mathrm{pred}}$ in s",
            r"MR$_{2_1}$ in \%",
        ),
        (
            "mr_n_obj",
            "num_obj_miss2_list",
            r"$n_{\mathrm{obj}}$",
            r"$n_{\mathrm{mr, rel}}$ in \%",
        ),
        (
            "mr_v_obj",
            "avg_speed_miss2_list",
            r"$v_{\mathrm{obj}}$ in m/s",
            r"$n_{\mathrm{mr, rel}}$ in \%",
        ),
        (
            "mr_obj_type",
            "obj_type_miss2_list",
            "objtype",
            r"$n_{\mathrm{mr, rel}}$ in \%",
        ),
    ]
    model_col_list = [TUM_BLACK, TUM_ORAN, TUM_GREEN, TUM_BLUE]
    type_col_list = [
        GRAY,
        TUM_BLUE,
        TUM_BLACK,
        TUM_ORAN,
        TUM_GREEN,
        TUM_LIGHT_BLUE,
        TUM_LIGHTER_BLUE,
    ]
    num_labels = cr_labels(None, get_len=True)
    category_names = cr_labels(None, get_label_str=True)
    category_colors = type_col_list + list(
        plt.colormaps["RdYlGn"](
            np.linspace(0.15, 0.85, num_labels - len(type_col_list))
        )[:, :3]
    )
    category_colors[11] = BOUND_COL
    print("\n\n\n#### Miss Rate Analysis ####")
    for save_name, dict_key, x_key, y_key in plt_keys:
        if dict_key == "obj_type_miss2_list":
            nrow = len(data_tag_dict)
            fig_s_in = (PAGEWIDTH * 1, PAGEWIDTH * 1 / 8 * len(data_tag_dict))
            legend_dict = {}
            fs_in = 16
        else:
            nrow = 1
            fig_s_in = FIGSIZE
            fs_in = 10
        _, ax = plt.subplots(
            nrows=nrow,
            ncols=1,
            figsize=fig_s_in,
        )
        update_matplotlib(fontsize=fs_in)

        print("dict_key: {}".format(dict_key))
        for j, (model_tag, result_dict) in enumerate(data_tag_dict.items()):
            if dict_key == "num_miss2_var_dict":
                x_val = np.array(list(result_dict[dict_key].keys())) / 10.0
                y_val = np.array(list(result_dict[dict_key].values())) * 100.0
                if model_tag == "g_sel":
                    y_val *= 10
            else:
                counts = sum(result_dict[dict_key], [])
                if len(counts) == 0:
                    continue
                x_val = []
                y_val = []
                if dict_key == "avg_speed_miss2_list":
                    dv = 1
                    for c_val in np.arange(0, 31, dv):
                        x_val.append(c_val)
                        y_val.append(
                            100.0
                            / len(counts)
                            * sum(abs(np.array(counts) - c_val) < dv / 2)
                        )
                else:
                    for c_val in range(int(max(counts)) + 1):
                        x_val.append(c_val)
                        y_val.append(
                            100.0 * (np.array(counts) == c_val).sum() / len(counts)
                        )

            print(
                "Model Tag: {}  ||  ".format(model_tag.upper())
                + " | ".join(
                    [
                        "x: {}, y: {:.04f}".format(xx, yy)
                        for xx, yy in zip(x_val, y_val)
                        if yy > 0
                    ]
                )
            )

            if dict_key == "obj_type_miss2_list":
                count_list = np.array(y_val) / 100.0 * len(counts)
                data_cum = np.array(count_list).cumsum()
                data_tot = sum(count_list)

                for j_i, width in zip(x_val, y_val):
                    if width == 0:
                        continue
                    catname = cr_labels(int(j_i))
                    color = category_colors[j_i]
                    width *= len(counts) / 100.0
                    start = data_cum[j_i] - width
                    _ = ax[j].barh(
                        model_tag.upper(),
                        width,
                        left=start,
                        height=1.0,
                        label=catname,
                        color=color,
                    )
                    if catname not in legend_dict:
                        legend_dict.update(
                            {
                                catname: Patch(
                                    facecolor=color,
                                    edgecolor=color,
                                    label=catname.lower(),
                                )
                            }
                        )

                tick_vals = np.linspace(0, data_tot, 6)
                tick_labels = ["{:.01f}".format((k * 2) / 10) for k in range(6)]
                ax[j].invert_xaxis()  # labels read top-to-bottom
                ax[j].set_xlim(left=0, right=data_tot)
                ax[j].set_xticks(tick_vals, tick_labels)
                # ax[j].xaxis.set_visible(False)
                ax[j].grid(False)
            else:
                label_in = model_tag.upper()
                if dict_key == "num_miss2_var_dict" and model_tag == "g_sel":
                    label_in += " x10"
                ax.plot(
                    x_val,
                    y_val,
                    label=label_in,
                    color=model_col_list[j],
                )

        if dict_key == "obj_type_miss2_list":
            print(
                "CR CATEGORY \n"
                + "  |  ".join(
                    [
                        "idx: {} - cat: {}".format(idx, cat)
                        for idx, cat in enumerate(category_names)
                    ]
                )
            )
            ax[-1].legend(
                handles=legend_dict.values(),
                loc="upper center",
                bbox_to_anchor=(0.5, -0.8),
                ncols=5,
                fontsize=fs_in*0.5,
            )
        else:
            ax.grid("True")
            ax.set_xlim(0)
            ax.set_ylim(-2.0)
            ax.set_xlabel(x_key)
            ax.set_ylabel(y_key)
            plt.tight_layout()

        print("\n\n")

        # save wo legend
        save_path = os.path.join(base_path, "mr_plots", save_name)
        if not os.path.exists(os.path.dirname(save_path)):
            os.mkdir(os.path.dirname(save_path))
        update_matplotlib(fontsize=fs_in)
        plt.savefig(save_path + ".svg", format="svg")
        plt.savefig(save_path + ".pdf", format="pdf")
        plt.savefig(save_path + ".png", format="png")

        if dict_key != "obj_type_miss2_list":
            save_path += "_legend"
            ax.legend()
            plt.savefig(save_path + ".svg", format="svg")
            plt.savefig(save_path + ".pdf", format="pdf")
            plt.savefig(save_path + ".png", format="png")

        plt.close()


def prepare_val(args):
    """Prepare validation."""
    with open(os.path.join(args.path, "train_config.json"), "r") as f:
        cfg_train = json.load(f)
    with open(os.path.join(args.path, "net_config.json"), "r") as f:
        net_config = json.load(f)

    cfg_train.update(args.__dict__)

    if args.debug:
        device = "cpu"
    else:
        device = cfg_train.get(
            "device",
            "cuda:0" if torch.cuda.is_available() else "cpu",
        )
    cfg_train["device"] = device

    _ = adjust_batch_size(cfg_train)

    if "open" in cfg_train["data"]:
        train_test_zip = [
            ("r_1", "r_A"),
            ("r_1", "r_B"),
        ]
        dataset = Dataset_OpenDD
    else:
        train_test_zip = [("cr", "cr")]
        dataset = Dataset_CR

    return dataset, train_test_zip, cfg_train, net_config


def get_dict_key(train_split, test_split, seed, is_opendd=False):
    """Get dict key."""
    if is_opendd:
        dict_key = "{}_{}_{}".format(train_split, test_split, seed)
    else:
        dict_key = "cr_{}".format(seed)
    return dict_key


def get_val_content(dataset, cfg_train, train_split, test_split, split_type="test"):
    """Get val content."""
    print("\n\n" + "#" * 80)
    print("Creating Test Dataloader (data = {}) ...".format(cfg_train["data"]))

    time_test_dataloader_start = datetime.datetime.now()

    is_shuffle = True
    data_split = test_split

    is_opendd = "open" in cfg_train["data"]
    base_path = os.path.abspath(cfg_train["path"])

    if cfg_train["debug"]:
        is_shuffle = False
        data_split = train_split
        split_type = "train"

    test_dataset = dataset(
        split=data_split,
        split_type=split_type,
        debug=cfg_train["debug"],
    )

    dataloader = DataLoader(
        test_dataset,
        batch_size=cfg_train["batch_size"],
        shuffle=is_shuffle,
        num_workers=cfg_train["num_workers"],
        pin_memory=cfg_train["pin_memory"],
    )
    print(
        "Completed after (Hours:Minutes:Seconds:Microseconds): "
        + str(datetime.datetime.now() - time_test_dataloader_start)
        + "\n"
    )

    """get map dict"""
    if is_opendd and cfg_train["sc_img"]:
        valid_rdbs = [1, 2, 3, 4, 5, 6, 7]
        rdp_map_dict = {
            rdb_int: get_rdp_map(rdb_int, data_path=test_dataset.processed_file_folder)
            for rdb_int in valid_rdbs
        }
    else:
        rdp_map_dict = None

    if cfg_train.get("no_g_sel", False):
        model_tag = "g_fusion_" + cfg_train["model_tag_list"][-1]
    else:
        model_tag = "g_fusion_g_sel"

    if cfg_train.get("load_model", False):
        last_model_base_path = list(cfg_train["model_path"].values())[-1]
        model_path = os.path.join(
            last_model_base_path,
            train_split,
            model_tag,
            str(cfg_train["seed"]),
            "model_parameters_{}.pth.tar".format(cfg_train["seed"]),
        )
    else:
        model_path = os.path.join(
            base_path,
            train_split,
            model_tag,
            str(cfg_train["seed"]),
            "model_parameters_{}.pth.tar".format(cfg_train["seed"]),
        )

    return dataloader, test_dataset.processed_file_folder, model_path, rdp_map_dict


def get_fusion_model(model_path, cfg_train, net_config):
    """Get fusion model."""
    checkpoint = torch.load(model_path, map_location=cfg_train["device"])
    model_stat_dict = checkpoint["model_state_dict"]
    g_fusion = Fusion_Model(
        cfg=cfg_train,
        net_config=net_config,
    )
    g_fusion.load_state_dict(model_stat_dict)
    g_fusion.to(cfg_train["device"])
    return g_fusion


def save_eval(base_path, all_res):
    """Save evaluation."""
    store_path = os.path.join(
        base_path,
        "evaluation_results.pkl",
    )
    with open(store_path, "wb") as fp:
        pickle.dump(all_res, fp)
    print("Saved to {}\n\n".format(store_path))


def calc_conf_matrix(
    y_true,
    y_pred,
    n_samples,
    sel_list,
    base_path,
):
    """Calculate Confusion matrix of G_sel."""
    conf_mat = confusion_matrix(y_true, y_pred, labels=list(range(len(sel_list))))
    print(
        "\n\nCONFUSION MATRIX (CM)\n"
        + "Vertical: Selection, horizontal: Ground truth\nSelector CM in percent (n = {})".format(
            n_samples
        )
    )

    print(conf_mat / sum(sum(conf_mat)) * 100)

    mat_size = len(sel_list)
    gt_mat = np.zeros((mat_size, mat_size))

    for kk in range(mat_size):
        gt_mat[kk, kk] = y_true.count(kk)
    print("Ground truth CM in percent (n = {})".format(n_samples))
    print(gt_mat / sum(sum(conf_mat)) * 100)

    # plot and save
    for mat, file_str in [(conf_mat, "pred"), (gt_mat, "gt")]:
        df_cm = pd.DataFrame(mat / sum(sum(conf_mat)), index=sel_list, columns=sel_list)
        plt.figure()
        sn.set(font_scale=1.4)  # for label size
        try:
            sn.heatmap(df_cm, annot=True)  # font size
        except Exception as er:
            print("\n\nNo sn.heatmap, error: {}\n\n".format(er))
            continue
        file_name = "g_sel_conf_mat_" + file_str + ".png"
        _ = save_plt_fusion(base_path, file_name, file_type="png")


def eval_distribution(
    rmse_over_samples,
    base_path,
    current_model,
    step_size=1.0,
    n_bin=21,
    is_logbins=True,
):
    """Evaluate distribution of RMSE error over samples."""
    if len(rmse_over_samples) == 0:
        print("\n\nrmse_over_samples is empty, no evaluation of distribution\n\n")
        return

    if is_logbins:
        bin_min = np.floor(np.log10(min(rmse_over_samples) + 1e-3))
        bin_max = max(2.0, np.ceil(np.log10(max(rmse_over_samples))))
        plt_bins = np.logspace(bin_min, bin_max, n_bin)
    else:
        max_err = max(20, int(max(rmse_over_samples)) + 1)
        plt_bins = np.arange(0.001, max_err + step_size / 2, step_size)

    plt.hist(rmse_over_samples, bins=plt_bins)
    plt.xscale("log")

    _ = save_plt_fusion(
        base_path=base_path,
        file_name="{}_histogram_rmse.svg".format(current_model),
    )


def classification_analysis(
    true_negatives, false_negatives, true_positives, false_positives
):
    """Evaluate classification performance.

    true_negatives: all correct valid selections.
    false_negatives: all invalids classified as valid (no matter which class).
    true_positives: all correct invalid selections.
    false_negative: all valid classified as invalid (no matter which class).
    """
    false_positive_rate = false_positives / max(
        (false_positives + true_negatives), 1e-6
    )
    false_negative_rate = false_negatives / max(
        (false_negatives + true_positives), 1e-6
    )

    precision = true_positives / max((true_positives + false_positives), 1e-6)
    sensitivity = true_positives / max((true_positives + false_negatives), 1e-6)
    specificity = true_negatives / max((true_negatives + false_positives), 1e-6)

    youden_index = sensitivity + specificity - 1

    accuracy = (true_positives + true_negatives) / max(
        (true_positives + true_negatives + false_positives + false_negatives), 1e-6
    )

    f_1_score = 2 / (1 / sensitivity + 1 / precision)

    print(
        "FPR = {:.03f} | FNR = {:.03f} | Precision = {:.03f} | Recall (Sensitivity) = {:.03f} | Specificity = {:.03f} | Youden Index = {:.03f} | F1-Score = {:.03f} | Accuracy = {:.03f}".format(
            false_positive_rate,
            false_negative_rate,
            precision,
            sensitivity,
            specificity,
            youden_index,
            f_1_score,
            accuracy,
        )
    )


def eval_edge_cases(
    pred_rmse,
    pred_pred,
    pred_data_y,
    pred_data_x,
    pred_data_obj_ref,
    pred_data_sc_img,
    base_path,
    pref="l_lstm_best",
    filter_zeros=False,
):
    """Evaluate best and worst prediction of a given model. Predictions are stored to plots."""
    pred_rmse = pred_rmse.to("cpu")
    pred_pred = pred_pred.to("cpu")
    pred_data_y = pred_data_y.to("cpu")
    pred_data_x = pred_data_x.to("cpu")
    pred_data_obj_ref = pred_data_obj_ref.to("cpu")
    pred_data_sc_img = pred_data_sc_img.to("cpu")

    if "best" in pref and filter_zeros:
        pref += "_non_zero"

    update_matplotlib(fontsize=FONTSIZE)

    for nn in range(len(pred_rmse)):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=FIGSIZE)

        # plot object: hist, fut, gt
        ax[0].plot(
            pred_data_x[nn, :, 0],
            pred_data_x[nn, :, 1],
            label="hist",
            color=TUM_BLUE,
        )
        ax[0].plot(
            pred_data_y[nn, :, 0],
            pred_data_y[nn, :, 1],
            label="gt",
            color=TUM_ORAN,
        )
        ax[0].plot(
            pred_pred[nn, :, 0],
            pred_pred[nn, :, 1],
            label="pred",
            color=TUM_GREEN,
        )

        ax[0].grid("True")

        ax[0].set_xlabel(
            "x",
        )
        ax[0].set_ylabel(
            "y",
        )
        ax[0].axis("equal")
        ax[0].legend()

        rdb_num, rdb_scene, timestamp, objid = pred_data_obj_ref[nn][0][:4]
        rdb_num = int(rdb_num)
        rdb_scene = int(rdb_scene)
        timestamp = float(timestamp)
        objid = int(objid)
        fig.suptitle(
            "RDB = {}, scene = {}, timestamp = {}, obj_id = {}, RMSE = {:.02f} m".format(
                rdb_num,
                rdb_scene,
                timestamp,
                objid,
                pred_rmse[nn],
            )
        )

        _ = save_plt_fusion(
            base_path=base_path,
            file_name=pref + "_pred_{}.svg".format(nn),
        )


def get_all_traj_loss(model, model_output, gt):
    """Get all trajectory loss."""
    test_data_pred, _ = model.get_selections(model_output)

    # get loss of best selection
    all_trajectory_loss = torch.pow(model_output[1] - gt[:, :, :2], 2.0)
    return test_data_pred, all_trajectory_loss


def evaluate_model(
    model,
    test_dataloader,
    train_split,
    test_split,
    cfg_train,
    rdp_map_dict,
    processed_file_folder,
    filter_zeros=True,
):
    """Evaluate submodel of Fusion Model on test data set."""
    is_opendd = "open" in cfg_train["data"]
    base_path = os.path.abspath(cfg_train["path"])

    # Init variables
    time_list = []
    if cfg_train["device"] == "cpu":
        mse_over_horizon = torch.zeros(model.output_length)
        opt_mse_over_horizon = torch.zeros(model.output_length)
    else:
        mse_over_horizon = torch.zeros(model.output_length, device="cuda:0")
        opt_mse_over_horizon = torch.zeros(model.output_length, device="cuda:0")

    num_miss_2_var_len = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    rmse_over_samples = []
    opt_rmse_over_samples = []

    selections_list = []
    opt_selections_list = []

    num_correct_selections = 0.0
    num_correct_selections_00 = 0.0
    num_correct_selections_05 = 0.0
    num_correct_selections_10 = 0.0
    num_miss2 = 0.0
    num_miss2_var_dict = {val: 0.0 for val in num_miss_2_var_len}
    num_obj_miss2_list = []
    obj_type_miss2_list = []
    avg_speed_miss2_list = []
    num_selections_valid = 0

    iter_loss_sum = 0.0
    num_samples = 0
    num_zero_samples = 0

    best_pred_rmse_tensor = torch.tensor([]).to("cpu")
    worst_pred_rmse_tensor = torch.tensor([]).to("cpu")
    best_pred_pred = torch.tensor([]).to("cpu")
    worst_pred_pred = torch.tensor([]).to("cpu")
    best_pred_data_y = torch.tensor([]).to("cpu")
    worst_pred_data_y = torch.tensor([]).to("cpu")
    best_pred_data_x = torch.tensor([]).to("cpu")
    worst_pred_data_x = torch.tensor([]).to("cpu")
    best_pred_data_obj_ref = torch.tensor([]).to("cpu")
    worst_pred_data_obj_ref = torch.tensor([]).to("cpu")
    best_pred_data_sc_img = torch.tensor([]).to("cpu")
    worst_pred_data_sc_img = torch.tensor([]).to("cpu")

    if cfg_train["debug"]:
        strstr = "DEBUG on train dataset " + str(train_split)
    else:
        strstr = "train " + str(train_split) + " - test " + str(test_split)
    print("\n")

    for test_data in tqdm(
        test_dataloader,
        desc=strstr,
    ):
        # Permute input
        temp_class = torch.clone(test_data.obj_class.detach()).to(cfg_train["device"])
        permute_input(data_batch=test_data)

        # Get sc images
        if is_opendd and cfg_train["sc_img"]:
            test_data.sc_img = get_sc_img_batch(
                batch_obj_ref=test_data.obj_ref,
                rdp_map_dict=rdp_map_dict,
                data_path=processed_file_folder,
                cfg=cfg_train,
                mod="val",
            )

        # Send data to device
        test_data.to(cfg_train["device"])

        # Predict
        with torch.no_grad():
            t0 = time.time()
            model_output = model(test_data)
            time_list.append(time.time() - t0)

        # Get loss
        (
            iter_loss_batch,
            sel_model_mse,
            selections,
            opt_model_mse,
            opt_selections,
        ) = model.loss(model_output, test_data.y[:, :, :2])

        # Selections
        selections_list.extend(selections.tolist())
        opt_selections_list.extend(opt_selections.tolist())
        num_correct_selections += torch.sum(selections == opt_selections)
        valid_selection_list = selections < len(model.model_tag_list)
        num_selections_valid += int(sum(valid_selection_list))

        # (R)mse
        mse_over_horizon += get_mse_over_horizon(sel_model_mse)
        opt_mse_over_horizon += get_mse_over_horizon(opt_model_mse)
        rmse_batch = get_rmse_over_samples(sel_model_mse)
        rmse_over_samples += rmse_batch.tolist()
        opt_rmse_over_samples += get_rmse_over_samples(opt_model_mse).tolist()

        # Miss 2 rate
        bool_mis_predictions = get_mis_pred_over_samples(
            sel_model_mse, horz_len=sel_model_mse.shape[1]
        )
        num_miss2 += torch.sum(bool_mis_predictions)
        if cfg_train.get("miss_rates", False):
            num_obj_vec = torch.tensor(
                sum(
                    [
                        [
                            test_data.ptr[j + 1] - test_data.ptr[j]
                            for _ in range(test_data.ptr[j + 1] - test_data.ptr[j])
                        ]
                        for j in range(len(test_data.ptr) - 1)
                    ],
                    [],
                ),
                device=cfg_train["device"],
            )
            num_obj_vec = num_obj_vec[valid_selection_list]
            num_obj_miss2_list.append(
                num_obj_vec[bool_mis_predictions].detach().cpu().tolist()
            )

            obj_class_filt = temp_class[valid_selection_list, 0, 0]
            obj_type_miss2_list.append(
                obj_class_filt[bool_mis_predictions].detach().cpu().tolist()
            )

            t_hist = 3.0
            objs_x_pos = test_data.x[valid_selection_list][:, :, :2]
            avg_hist_speed_filt = ((objs_x_pos.diff(dim=1) ** 2).sum(dim=2) ** 0.5).sum(
                dim=1
            ) / t_hist
            avg_speed_miss2_list.append(
                avg_hist_speed_filt[bool_mis_predictions].detach().cpu().tolist()
            )

            for var_len in num_miss_2_var_len:
                bool_mis_predictions_var = get_mis_pred_over_samples(
                    sel_model_mse, horz_len=var_len
                )
                num_miss2_var_dict[var_len] += (
                    torch.sum(bool_mis_predictions_var).detach().cpu().numpy()
                )

        # Sum over loss
        iter_loss_sum += iter_loss_batch

        # num samples
        num_samples += test_data.x.shape[0]

        # count number of zero samples
        num_zero_samples += sum(
            torch.sum(torch.sum(test_data.x[:, :, :2], dim=1), dim=1) == 0
        )

        # concatenate and sort rmse errors
        err_cat = torch.concat(
            [
                best_pred_rmse_tensor,
                rmse_batch.to("cpu"),
                worst_pred_rmse_tensor,
            ],
            dim=0,
        )
        values, indices = err_cat.sort()

        # Get traj prediction and gt data
        if "g_sel" in model.tag:
            test_data_pred, all_trajectory_loss = get_all_traj_loss(
                model, model_output, test_data.y
            )

            _, opt_selections_non_filtered = all_trajectory_loss.sum((2, 3)).min(dim=0)
            opt_model_mse_non_filtered = all_trajectory_loss[
                opt_selections_non_filtered, opt_selections_non_filtered > -1
            ]
            opt_rmse_over_samples_valid = get_rmse_over_samples(
                opt_model_mse_non_filtered[valid_selection_list]
            )

            error_ratio = rmse_batch / torch.clip(opt_rmse_over_samples_valid, min=1e-6)
            num_correct_selections_00 += torch.sum(error_ratio <= 1.00)
            num_correct_selections_05 += torch.sum(error_ratio < 1.05)
            num_correct_selections_10 += torch.sum(error_ratio < 1.10)
        else:
            test_data_pred = model_output
            num_correct_selections_00 += torch.sum(rmse_batch > -1)  # dummy
            num_correct_selections_05 += torch.sum(rmse_batch > -1)  # dummy
            num_correct_selections_10 += torch.sum(rmse_batch > -1)  # dummy

        pred_cat = torch.concat(
            [best_pred_pred, test_data_pred.to("cpu"), worst_pred_pred], dim=0
        )
        if "g_sel" in model.tag:
            test_data_y = test_data.y[valid_selection_list]
            test_data_x = test_data.x[valid_selection_list]
            test_data_obj_ref = test_data.obj_ref[valid_selection_list]
            test_data_sc_img = test_data.sc_img[valid_selection_list]
        else:
            test_data_y = test_data.y
            test_data_x = test_data.x
            test_data_obj_ref = test_data.obj_ref
            test_data_sc_img = test_data.sc_img

        data_cat_y = torch.concat(
            [best_pred_data_y, test_data_y.to("cpu"), worst_pred_data_y], dim=0
        )
        data_cat_x = torch.concat(
            [best_pred_data_x, test_data_x.to("cpu"), worst_pred_data_x], dim=0
        )
        data_cat_obj_ref = torch.concat(
            [
                best_pred_data_obj_ref,
                test_data_obj_ref.to("cpu"),
                worst_pred_data_obj_ref,
            ],
            dim=0,
        )
        data_cat_sc_img = torch.concat(
            [best_pred_data_sc_img, test_data_sc_img.to("cpu"), worst_pred_data_sc_img],
            dim=0,
        )

        # get 10 best predictions
        best_pred_rmse = values[:10]
        best_pred_rmse_tensor = err_cat[indices][:10]
        best_pred_pred = pred_cat[indices][:10]
        best_pred_data_y = data_cat_y[indices][:10]
        best_pred_data_x = data_cat_x[indices][:10]
        best_pred_data_obj_ref = data_cat_obj_ref[indices][:10]
        best_pred_data_sc_img = data_cat_sc_img[indices][:10]

        # filter zeros values out of best predictions
        if filter_zeros:
            non_zeros_idx = best_pred_rmse > 1e-2
            best_pred_rmse = best_pred_rmse[non_zeros_idx]
            best_pred_rmse_tensor = best_pred_rmse_tensor[non_zeros_idx]
            best_pred_pred = best_pred_pred[non_zeros_idx]
            best_pred_data_y = best_pred_data_y[non_zeros_idx]
            best_pred_data_x = best_pred_data_x[non_zeros_idx]
            best_pred_data_obj_ref = best_pred_data_obj_ref[non_zeros_idx]
            best_pred_data_sc_img = best_pred_data_sc_img[non_zeros_idx]

        # get 10 worst predicitions
        worst_pred_rmse = values[-10:]
        worst_pred_rmse_tensor = err_cat[indices][-10:]
        worst_pred_pred = pred_cat[indices][-10:]
        worst_pred_data_y = data_cat_y[indices][-10:]
        worst_pred_data_x = data_cat_x[indices][-10:]
        worst_pred_data_obj_ref = data_cat_obj_ref[indices][-10:]
        worst_pred_data_sc_img = data_cat_sc_img[indices][-10:]

    # End of iteration of test data, get overall values
    rmse_over_horizon = (
        torch.pow(mse_over_horizon / num_selections_valid, 0.5).cpu().detach().numpy()
    )
    opt_rmse_over_horizon = (
        torch.pow(opt_mse_over_horizon / num_selections_valid, 0.5)
        .cpu()
        .detach()
        .numpy()
    )

    # Overall model loss
    iter_loss_sum /= num_samples

    # Overall correct selections
    num_correct_selections /= num_samples
    num_correct_selections_00 /= num_selections_valid
    num_correct_selections_05 /= num_selections_valid
    num_correct_selections_10 /= num_selections_valid

    # miss rate only in case of valid output
    num_miss2 /= num_selections_valid
    for var_key in num_miss2_var_dict.keys():
        num_miss2_var_dict[var_key] /= num_selections_valid

    # classification analysis
    opt_selections_array = np.array(opt_selections_list)
    selections_array = np.array(selections_list)
    # check ratio of correct selection within the invalids only:
    opt_valid_selection_list = opt_selections_array < len(model.model_tag_list)
    selections_in_invalid = selections_array[opt_valid_selection_list is False]
    opt_selections_in_invalid = opt_selections_array[opt_valid_selection_list is False]
    true_positives = np.sum(selections_in_invalid == opt_selections_in_invalid)
    false_positives = np.sum(selections_in_invalid < len(model.model_tag_list))
    selections_in_valid = selections_array[opt_valid_selection_list]
    true_negatives = np.sum(selections_in_valid < len(model.model_tag_list))
    false_negatives = np.sum(
        selections_in_valid == len(model.model_tag_list)
    )  # not the same as all_valid - true_negatives

    if not cfg_train.get("miss_rates", False):
        # calc confusion matrix
        if "g_sel" in model.tag:
            _ = calc_conf_matrix(
                y_true=opt_selections_list,
                y_pred=selections_list,
                n_samples=num_samples,
                sel_list=model.model_tag_list + ["invalid"],
                base_path=base_path,
            )

        # plot rmse over horizon
        _ = plt_rmse_over_horizon_model(
            rmse_over_horizon,
            opt_rmse_over_horizon,
            base_path,
            model.current_model,
        )

        # get historgram
        _ = eval_distribution(
            rmse_over_samples,
            base_path,
            model.current_model,
        )

        # eval best predictions
        _ = eval_edge_cases(
            best_pred_rmse,
            best_pred_pred,
            best_pred_data_y,
            best_pred_data_x,
            best_pred_data_obj_ref,
            best_pred_data_sc_img,
            base_path,
            pref=str(model.current_model) + "_best",
            filter_zeros=filter_zeros,
        )

        # eval worst predictions
        _ = eval_edge_cases(
            worst_pred_rmse,
            worst_pred_pred,
            worst_pred_data_y,
            worst_pred_data_x,
            worst_pred_data_obj_ref,
            worst_pred_data_sc_img,
            base_path,
            pref=str(model.current_model) + "_worst",
        )

    # Start print
    print("\n" + "#" * 80 + "\n")
    print("EVALUATION OF {}".format(model.current_model.upper()))

    # Print zero entries
    print(
        "data, zero entries: {} / {} = {:.02f} %".format(
            num_zero_samples, num_samples, 100 * num_zero_samples / num_samples
        )
    )

    # Print execution time
    print(
        "execution time (device = {}): {:.0f} ms ({:.02f} objects in avg.)".format(
            cfg_train["device"],
            sum(time_list) / len(time_list) * 1000,
            num_samples / len(test_dataloader),
        )
    )

    # Print
    _ = print_overview(
        model.current_model,
        len(test_dataloader) + 1,
        iter_loss_sum,
        rmse_over_horizon.mean(),
        opt_rmse_over_horizon.mean(),
        num_correct_selections,
        end_str="\n",
    )
    if "g_sel" in model.tag:
        print(
            "correct selection: {:.02f} | correct selections (0%): {:.02f} % | correct selections (5%): {:.02f} % | correct selections (10%): {:.02f} %".format(
                num_correct_selections * 100,
                num_correct_selections_00 * 100,
                num_correct_selections_05 * 100,
                num_correct_selections_10 * 100,
            )
        )

        classification_analysis(
            true_negatives, false_negatives, true_positives, false_positives
        )

    print_quantiles(rmse_over_samples)
    print("MissRate-2-1: {:.02f} %".format(num_miss2 * 100.0))

    # print RMSE
    if "open" in cfg_train["data"]:
        raise NotImplementedError("time steps for rmse output for openDD missing")
    else:
        n_s = list(range(5))
        idx = list(range(9, 50, 10))
        for s_step, idx_step in zip(n_s, idx):
            strstr = "rmse {0:d}s: {1:.2f} m".format(
                s_step + 1, rmse_over_horizon[idx_step]
            )
            if "g_sel" in model.tag:
                strstr += " (min.: {:.2f} m)".format(opt_rmse_over_horizon[idx_step])
            print(strstr)

    # end print
    print("#" * 80 + "\n\n")

    results_dict = {
        "time_list": time_list,
        "mse_over_horizon": mse_over_horizon,
        "opt_mse_over_horizon": opt_mse_over_horizon,
        "rmse_over_horizon": rmse_over_horizon,
        "opt_rmse_over_horizon": opt_rmse_over_horizon,
        "rmse_over_samples": np.array(rmse_over_samples),
        "opt_rmse_over_samples": np.array(opt_rmse_over_samples),
        "selections_list": selections_list,
        "opt_selections_list": opt_selections_list,
        "num_selections_valid": num_selections_valid,
        "num_correct_selections": num_correct_selections,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "true_positives": true_positives,
        "num_correct_selections_00": num_correct_selections_00,
        "num_correct_selections_05": num_correct_selections_05,
        "num_correct_selections_10": num_correct_selections_10,
        "miss2rate": num_miss2,
        "iter_loss_sum": iter_loss_sum,
        "num_samples": num_samples,
        "num_zero_samples": num_zero_samples,
        "n_params": model.get_tot_params(),
    }

    if len(rmse_over_samples) > 0:
        results_dict.update(
            {
                "10q_rmse_over_samples": np.quantile(
                    rmse_over_samples, q=np.arange(0.1, 1.05, 0.1)
                ),
            }
        )
    if len(opt_rmse_over_samples) > 0:
        results_dict.update(
            {
                "10q_opt_rmse_over_samples": np.quantile(
                    opt_rmse_over_samples, q=np.arange(0.1, 1.05, 0.1)
                ),
            }
        )

    if cfg_train.get("miss_rates", False):
        results_dict.update(
            {
                "num_miss2_var_dict": num_miss2_var_dict,
                "num_obj_miss2_list": num_obj_miss2_list,
                "obj_type_miss2_list": obj_type_miss2_list,
                "avg_speed_miss2_list": avg_speed_miss2_list,
            }
        )

    return results_dict


class Logger(object):
    """Logger class to print to terminal and write to logfile."""

    def __init__(self, file_path):
        """Initialize logger class."""
        self.terminal = sys.stdout
        self.file_path = file_path
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
        if not os.path.exists(os.path.dirname(self.file_path)):
            os.makedirs(os.path.dirname(self.file_path))

    def write(self, message):
        """Write to logger."""
        with open(self.file_path, "a", encoding="utf-8") as self.log:
            self.log.write(message)
        self.terminal.write(message)

    def flush(self):
        """Flush method.

        Flush method is needed for python 3 compatibility.
        this handles the flush command by doing nothing.
        you might want to specify some extra behavior here.
        """
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default=os.path.join(repo_path, "results", "cr_fusion_08"),
        help="path to stored training",
    )
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--miss_rates", default=False, action="store_true")

    args = parser.parse_args()

    if not os.path.exists(args.path):
        raise ValueError("Results path does not exist")

    eval_main(args)
