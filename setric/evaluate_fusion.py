import os
import sys
import time
import json
import pickle
import argparse
import datetime

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

from tqdm import tqdm
import matplotlib.pyplot as plt
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
from utils.metrics import get_mse_over_horizon, get_rmse_over_samples

from models.Fusion_Model import Fusion_Model

TUM_BLUE = (0 / 255, 101 / 255, 189 / 255)
TUM_ORAN = (227 / 255, 114 / 255, 34 / 255)
TUM_GREEN = (162 / 255, 173 / 255, 0 / 255)
TUM_BLACK = (0, 0, 0)
GRAY = (104 / 255, 104 / 255, 104 / 255)
BOUND_COL = (204 / 255, 204 / 255, 204 / 255)

COLUMNWIDTH = 3.5
PAGEWIDTH = COLUMNWIDTH * 2.0
FONTSIZE = 10  # IEEE is 8


def update_matplotlib():
    """Update matplotlib font style."""
    plt.rcParams.update(
        {
            "font.size": FONTSIZE,
            "font.family": "Times New Roman",
            "text.usetex": True,
            "figure.autolayout": True,
            "xtick.labelsize": FONTSIZE * 1.0,
            "ytick.labelsize": FONTSIZE * 1.0,
        }
    )


def eval_main(base_path, debug, model_seed_list=[10]):
    """Evaluates fusion model on dataset (CR or OpenDD)."""

    # write to file and to terminal
    orig_stdout = sys.stdout
    sys.stdout = Logger(file_path=os.path.join(base_path, "eval_logs.txt"))

    with open(os.path.join(base_path, "train_config.json"), "r") as f:
        cfg_train = json.load(f)
    with open(os.path.join(base_path, "net_config.json"), "r") as f:
        net_config = json.load(f)

    if debug:
        device = "cpu"
        cfg_train["device"] = device
    else:
        device = cfg_train.get(
            "device", "cuda" if torch.cuda.is_available() and not debug else "cpu"
        )

    cfg_train["batch_size"] = adjust_batch_size(cfg_train)

    all_res = {}

    if "open" in cfg_train["data"]:
        train_test_zip = [
            ("r_1", "r_A"),
            ("r_1", "r_B"),
        ]
        dataset = Dataset_OpenDD
        is_opendd = True
    else:
        train_test_zip = [("cr", "cr")]
        dataset = Dataset_CR
        is_opendd = False

    for train_split, test_split in train_test_zip:
        for seed in model_seed_list:
            seed_everything(seed)
            if is_opendd:
                dict_key = "{}_{}_{}".format(train_split, test_split, seed)
            else:
                dict_key = "cr_{}".format(seed)
            all_res[dict_key] = {}

            print("Creating Test Dataloader (data = {})...".format(cfg_train["data"]))
            time_test_dataloader_start = datetime.datetime.now()
            if debug:
                data_split = train_split
                split_type = "train"
                is_shuffle = False
            else:
                data_split = test_split
                split_type = "test"
                is_shuffle = True
            test_dataset = dataset(split=data_split, split_type=split_type, debug=debug)

            test_dataloader = DataLoader(
                test_dataset,
                batch_size=cfg_train["batch_size"],
                shuffle=is_shuffle,
                num_workers=cfg_train.get("num_workers", 2),
                pin_memory=cfg_train.get("pin_memory", True),
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
                    rdb_int: get_rdp_map(
                        rdb_int, data_path=test_dataset.processed_file_folder
                    )
                    for rdb_int in valid_rdbs
                }
            else:
                rdp_map_dict = None

            model_path = os.path.join(
                base_path,
                train_split,
                "g_fusion_g_sel",
                str(seed),
                "model_parameters_{}.pth.tar".format(seed),
            )
            if not os.path.exists(model_path):
                model_path = os.path.join(
                    base_path,
                    "r_1",
                    "g_fusion_g_sel",
                    str(seed),
                    "model_parameters_{}.pth.tar".format(seed),
                )

            checkpoint = torch.load(model_path, map_location=device)
            model_stat_dict = checkpoint["model_state_dict"]
            g_fusion = Fusion_Model(
                cfg=cfg_train,
                net_config=net_config,
            )
            g_fusion.load_state_dict(model_stat_dict)
            g_fusion.to(device)
            g_fusion.eval()

            for n_iter, model_tag in enumerate(g_fusion.model_tag_list + ["g_sel"]):
                g_fusion.update_model_tag(n_iter, model_tag)

                result_dict = evaluate_model(
                    g_fusion,
                    test_dataloader,
                    train_split,
                    test_split,
                    cfg_train,
                    rdp_map_dict,
                    test_dataset.processed_file_folder,
                    device,
                    base_path,
                    debug=debug,
                    is_opendd=is_opendd,
                )

                all_res[dict_key][model_tag] = result_dict

    store_path = os.path.join(
        base_path,
        "evaluation_results.pkl",
    )
    with open(store_path, "wb") as fp:
        pickle.dump(all_res, fp)
    print("Saved to {}\n\n".format(store_path))

    # put back the original state of stdout
    sys.stdout = orig_stdout


def plt_rmse_over_horizon(rmse, opt_rmse, base_path, current_model):
    """Plots RMSE over time horizon."""
    plt.plot(np.arange(0.1, 5.05, 0.1), rmse, label="RMSE")
    if current_model == "g_sel":
        plt.plot(np.arange(0.1, 5.05, 0.1), opt_rmse, label="RMSE opt")

    plt.legend()

    _ = save_plt(
        base_path=base_path, file_name="{}_rmse_over_horizon.svg".format(current_model)
    )


def calc_conf_matrix(y_true, y_pred, n_samples, sel_list, base_path):
    """Calculate Confusion matrix of G_sel."""
    conf_mat = confusion_matrix(y_true, y_pred)
    print(
        "CONFUSION MATRIX in percent (n = {}), vertical: SELECTION, horizontal: BEST PREDICTION".format(
            n_samples
        )
    )
    print(conf_mat / sum(sum(conf_mat)) * 100)

    mat_size = len(set(y_true))
    gt_mat = np.zeros((mat_size, mat_size))
    for kk in set(y_true):
        gt_mat[kk, kk] = y_true.count(kk)
    print("GROUND TRUTH MATRIX in percent (n = {})".format(n_samples))
    print(gt_mat / sum(sum(conf_mat)) * 100)

    # plot and save
    for mat, file_str in [(conf_mat, "pred"), (gt_mat, "gt")]:
        df_cm = pd.DataFrame(mat / sum(sum(conf_mat)), index=sel_list, columns=sel_list)
        plt.figure()
        sn.set(font_scale=1.4)  # for label size
        sn.heatmap(df_cm, annot=True)  # font size
        file_name = "g_sel_conf_mat_" + file_str + ".png"
        _ = save_plt(base_path, file_name, file_type="png")


def save_plt(base_path, file_name, file_type="svg"):
    """Saves plot to save_path."""
    save_path = os.path.join(base_path, "plots")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = os.path.join(save_path, file_name)
    plt.savefig(save_path, format=file_type)
    plt.close()


def eval_distribution(
    rmse_over_samples,
    base_path,
    current_model,
    step_size=1.0,
    n_bin=21,
    is_logbins=True,
):
    "Evaluates distribution of RMSE error over samples."
    if is_logbins:
        bin_min = np.floor(np.log10(min(rmse_over_samples) + 1e-3))
        bin_max = max(2.0, np.ceil(np.log10(max(rmse_over_samples))))
        plt_bins = np.logspace(bin_min, bin_max, n_bin)
    else:
        max_err = max(20, int(max(rmse_over_samples)) + 1)
        plt_bins = np.arange(0.001, max_err + step_size / 2, step_size)

    plt.hist(rmse_over_samples, bins=plt_bins)
    plt.xscale("log")

    _ = save_plt(
        base_path=base_path, file_name="{}_histogram_rmse.svg".format(current_model)
    )


def classification_analysis(
    true_negatives, false_negatives, true_positives, false_positives
):
    """
    true_negatives: all correct valid selections.
    false_negatives: all invalids classified as valid (no matter which class).
    true_positives: all correct invalid selections.
    false_negative: all valid classified as invalid (no matter which class).
    """
    false_positive_rate = false_positives / (false_positives + true_negatives)
    false_negative_rate = false_negatives / (false_negatives + true_positives)

    precision = true_positives / (true_positives + false_positives)
    sensitivity = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)

    youden_index = sensitivity + specificity - 1

    accuracy = (true_positives + true_negatives) / (
        true_positives + true_negatives + false_positives + false_negatives
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
    "Evaluates best and worst prediction of a given model. Predictions are stored to plots."
    pred_rmse = pred_rmse.to("cpu")
    pred_pred = pred_pred.to("cpu")
    pred_data_y = pred_data_y.to("cpu")
    pred_data_x = pred_data_x.to("cpu")
    pred_data_obj_ref = pred_data_obj_ref.to("cpu")
    pred_data_sc_img = pred_data_sc_img.to("cpu")

    if "best" in pref and filter_zeros:
        pref += "_non_zero"

    update_matplotlib()
    if os.getlogin() == "ubuntu":
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "text.usetex": False,
            }
        )

    for nn in range(len(pred_rmse)):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))

        ax[1].imshow(pred_data_sc_img[nn, 0])
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

        plt.tight_layout()

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

        _ = save_plt(base_path=base_path, file_name=pref + "_pred_{}.svg".format(nn))


def evaluate_model(
    model,
    test_dataloader,
    train_split,
    test_split,
    cfg_train,
    rdp_map_dict,
    processed_file_folder,
    device,
    base_path,
    filter_zeros=True,
    debug=False,
    is_opendd=False,
):
    "Evaluates submodel of Fusion Model on test data set."
    # Init variables
    time_list = []
    if device == "cpu":
        mse_over_horizon = torch.zeros(model.output_length)
        opt_mse_over_horizon = torch.zeros(model.output_length)
    else:
        mse_over_horizon = torch.zeros(model.output_length, device="cuda")
        opt_mse_over_horizon = torch.zeros(model.output_length, device="cuda")

    rmse_over_samples = []
    opt_rmse_over_samples = []

    selections_list = []
    opt_selections_list = []

    num_correct_selections = 0.0
    num_correct_selections_00 = 0.0
    num_correct_selections_05 = 0.0
    num_correct_selections_10 = 0.0
    num_miss2 = 0.0
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

    if debug:
        strstr = "DEBUG on train dataset " + str(train_split)
    else:
        strstr = "train " + str(train_split) + " - test " + str(test_split)
    print("\n")

    for test_data in tqdm(
        test_dataloader,
        desc=strstr,
    ):
        # Permute input
        test_data.x = test_data.x.permute(0, 2, 1)
        test_data.y = test_data.y.permute(0, 2, 1)

        # Add object class to the time series data - obj_class shape: (N, seq_length, 1)
        obj_class = test_data.obj_class.repeat(1, test_data.x.shape[1], 1)
        test_data.x = torch.cat((test_data.x, obj_class), dim=2)

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
        test_data.to(device)

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
        val, _ = torch.max(
            torch.pow(torch.sum(sel_model_mse, dim=-1), 0.5) > 2.0, dim=-1
        )
        num_miss2 += torch.sum(val)

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
            test_data_pred, _ = model.get_selections(model_output)

            # get loss of best selection
            all_trajectory_loss = torch.pow(
                model_output[1] - test_data.y[:, :, :2], 2.0
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

    # classification analysis
    opt_selections_array = np.array(opt_selections_list)
    selections_array = np.array(selections_list)
    # check ratio of correct selection within the invalids only:
    opt_valid_selection_list = opt_selections_array < len(model.model_tag_list)
    selections_in_invalid = selections_array[opt_valid_selection_list == False]
    opt_selections_in_invalid = opt_selections_array[opt_valid_selection_list == False]
    true_positives = np.sum(selections_in_invalid == opt_selections_in_invalid)
    false_positives = np.sum(selections_in_invalid < len(model.model_tag_list))
    selections_in_valid = selections_array[opt_valid_selection_list]
    opt_selections_in_valid = opt_selections_array[opt_valid_selection_list]
    true_negatives = np.sum(selections_in_valid < len(model.model_tag_list))
    false_negatives = np.sum(
        selections_in_valid == len(model.model_tag_list)
    )  # not the same as all_valid - true_negatives

    try:
        # calc confusion matrix
        if "g_sel" in model.tag:
            _ = calc_conf_matrix(
                y_true=opt_selections_list,
                y_pred=selections_list,
                n_samples=num_samples,
                sel_list=model.model_tag_list + ["g_sel"],
                base_path=base_path,
            )

        # plot rmse over horizon
        _ = plt_rmse_over_horizon(
            rmse_over_horizon, opt_rmse_over_horizon, base_path, model.current_model
        )

        # get historgram
        _ = eval_distribution(rmse_over_samples, base_path, model.current_model)

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
    except Exception:
        print("no eval edge cases. {}".format(Exception))

    # Start print
    print("=" * 80)
    print("evaluation of {}".format(model.current_model))

    # Print zero entries
    print(
        "data, zero entries: {} / {} = {:.02f} %".format(
            num_zero_samples, num_samples, 100 * num_zero_samples / num_samples
        )
    )

    # Print execution time
    print(
        "execution time (device = {}): {:.0f} ms ({:.02f} objects in avg.)".format(
            device,
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
    print("=" * 80)

    results_dict = {
        "time_list": time_list,
        "mse_over_horizon": mse_over_horizon,
        "opt_mse_over_horizon": opt_mse_over_horizon,
        "rmse_over_horizon": rmse_over_horizon,
        "opt_rmse_over_horizon": opt_rmse_over_horizon,
        "rmse_over_samples": np.array(rmse_over_samples),
        "opt_rmse_over_samples": np.array(opt_rmse_over_samples),
        "10q_rmse_over_samples": np.quantile(
            rmse_over_samples, q=np.arange(0.1, 1.05, 0.1)
        ),
        "10q_opt_rmse_over_samples": np.quantile(
            opt_rmse_over_samples, q=np.arange(0.1, 1.05, 0.1)
        ),
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
    }

    return results_dict


class Logger(object):
    """Logger class to print to terminal and write to logfile."""

    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.file_path = file_path
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    def write(self, message):
        with open(self.file_path, "a", encoding="utf-8") as self.log:
            self.log.write(message)
        self.terminal.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
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

    args = parser.parse_args()

    if not os.path.exists(args.path):
        raise ValueError("Results path does not exist")

    eval_main(args.path, args.debug)
e