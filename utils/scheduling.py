import os
import sys
import datetime
import json

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

import names
import torch
import numpy as np


def store_current_config(
    base_path, cfg_train, net_config, args, n_iter=0, add_best_str=None
):
    if n_iter > 0:
        return
    if add_best_str is None:
        strstr = ""
    else:
        strstr = add_best_str
    with open(
        os.path.join(base_path, strstr + os.path.basename(args.config)), "w"
    ) as f:
        json.dump(cfg_train, f, indent=4)
    with open(
        os.path.join(base_path, strstr + os.path.basename(args.net_config)), "w"
    ) as f:
        json.dump(net_config, f, indent=4)


def adapt_to_CR(net_config, cfg_train, split_list):
    if "open" in cfg_train["data"]:
        net_config["output_length"] = 10
        net_config["input_features"] = 8
        net_config["dt_step_s"] = 0.2
        return net_config, split_list

    net_config["output_length"] = 50
    net_config["input_features"] = 4
    net_config["dt_step_s"] = 0.1
    return net_config, ["cr"]


def adjust_batch_size(cfg):
    hostname = os.uname()[1]
    if hostname == "gpu-vm" and cfg["device"] != "cpu":
        return min(16, cfg["batch_size"])

    return cfg["batch_size"]


def get_config(args, results_dir_path, is_fusion=False):
    # Init config
    # get name of net
    is_bayesian = args.__dict__.get("bayesian", False)
    if is_bayesian:
        net_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        net_name = get_random_name()  # get name before seeding

    cfg = {
        "net_name": net_name,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    # add args
    cfg.update(args.__dict__)

    # load train config
    with open(os.path.join(repo_path, args.config), "r") as f:
        cfg_json = json.load(f)
    cfg.update(cfg_json)

    _ = get_model_tag_list(cfg)

    # load net config
    if cfg["load_model"]:
        last_model_tag = cfg["model_tag_list"][-1]
        last_model_path = cfg["model_path"][last_model_tag]
        with open(os.path.join(last_model_path, "net_config.json"), "r") as f:
            net_config = json.load(f)
    else:
        with open(os.path.join(repo_path, args.net_config), "r") as f:
            net_config = json.load(f)

    cfg["batch_size"] = adjust_batch_size(cfg)

    # Init results path
    if cfg["debug"]:
        results_dir_path = os.path.join(
            os.path.dirname(results_dir_path), "results_debug"
        )
    if is_bayesian:
        results_dir_path = results_dir_path.replace("results", "results_bayesian")
        results_dir_path = os.path.join(results_dir_path, args.time_stamp_str)
    if not os.path.exists(results_dir_path):
        os.mkdir(results_dir_path)

    strstr = ""
    if not "open" in cfg["data"]:
        strstr += "cr_"
    if is_fusion:
        strstr += "fusion_"

    base_path = os.path.join(results_dir_path, strstr + cfg["net_name"])
    cfg["path"] = base_path
    if os.path.exists(base_path):
        raise ValueError("Path already exists: '{}'".format(base_path))

    return net_config, cfg, base_path


def get_random_name():
    return names.get_full_name(gender="female").lower().replace(" ", "_")


def get_model_tag_list(cfg):
    if cfg["model_list"] == 0:
        cfg["model_tag_list"] = ["l_lstm", "dg_lstm"]
    elif cfg["model_list"] == 1:
        cfg["model_tag_list"] = ["cv", "l_lstm", "dg_lstm"]
    elif cfg["model_list"] == 2:
        cfg["model_tag_list"] = ["cv", "l_lstm"]
    else:
        raise ValueError(
            "invalid choise of 'args.model_list' = {}".format(cfg["model_tag_list"])
        )


def update_configs(config_update, net_config, cfg_train):
    if config_update is not None:
        for update_key in config_update.keys():
            if update_key in net_config:
                net_config[update_key] = config_update[update_key]
            if update_key in cfg_train:
                cfg_train[update_key] = config_update[update_key]


def modify_for_debug(cfg_train):
    if not cfg_train["debug"]:
        return

    if cfg_train.get("bayesian", False):
        n_epochs = 120
    else:
        n_epochs = 200

    update_dict = {
        "epochs": n_epochs,
        "step_size": 300,
        # "num_workers": 1,
        # "pin_memory": False,
        # "device": "cpu"
    }

    cfg_train.update(update_dict)


def get_log_root_path(base_path, split, tag):
    log_root_path = os.path.join(base_path, split, tag)
    if not os.path.exists(log_root_path):
        os.makedirs(log_root_path)
    return log_root_path


def update_model_path(cfg):
    cfg["model_path"] = {key: cfg["path"] for key in cfg["model_path"].keys()}


def load_net_params(model, cfg):
    iterator = model.model_tag_list + ["g_sel"]

    if not cfg["load_model"]:
        _ = update_model_path(cfg)
        return iterator

    last_model_tag = cfg["model_tag_list"][-1]
    last_model_path = cfg["model_path"][last_model_tag]

    if not last_model_path:
        raise ValueError("No model path given for '{}'".format(last_model_tag))

    if not os.path.exists(last_model_path):
        raise ValueError("No model found at '{}'".format(last_model_path))

    model.update_model_tag(n_iter=0, model_tag=last_model_tag)
    model_load_path = os.path.join(
        last_model_path,
        cfg["split"],
        model.tag,
        str(cfg["seed"]),
        "model_parameters_{}.pth.tar".format(cfg["seed"]),
    )
    checkpoint = torch.load(model_load_path, map_location=cfg["device"])
    model_stat_dict = checkpoint["model_state_dict"]
    model.load_state_dict(model_stat_dict)
    model.to(cfg["device"])

    model.encoder_trained = True
    model.sc_img_trained = True

    iterator = ["g_sel"]

    return iterator


def print_quantiles(rmse_over_sample, log_string=None):
    strstr = "rmse_over_samples quantiles: 0.8-error: {:.04f} m | 0.85-error: {:.04f} m | 0.9-error: {:.04f} m | 0.95-error: {:.04f} m | 1.0-error: {:.04f} m".format(
        np.quantile(rmse_over_sample, q=0.8),
        np.quantile(rmse_over_sample, q=0.85),
        np.quantile(rmse_over_sample, q=0.9),
        np.quantile(rmse_over_sample, q=0.95),
        np.quantile(rmse_over_sample, q=1.0),
    )
    if log_string is None:
        print(strstr)
    else:
        log_string(strstr)


def print_overview(
    model_tag,
    n_iter,
    iter_loss,
    avg_rmse_over_horizon,
    avg_min_rmse_over_horizon,
    rel_correct_selections,
    end_str,
    string_only=False,
):
    if model_tag == "g_sel":
        factor = 1000
    else:
        factor = 1

    strstr = "model: {} | iter: {:03d} | avg. loss (x{:04d}): {:.03f} | RMSE: {:.03f} | min. RMSE: {:.03f} | correct selections: {:.02f} %".format(
        model_tag.upper(),
        n_iter,
        factor,
        iter_loss,
        avg_rmse_over_horizon,
        avg_min_rmse_over_horizon,
        rel_correct_selections * 100,
    )

    if string_only:
        return strstr

    print(strstr, end=end_str)


def print_datastats(dataloader, split_type):
    n_zeros = 0
    n_total = 0

    for batch in dataloader:
        batch.x = batch.x.permute(0, 2, 1)
        batch.y = batch.y.permute(0, 2, 1)
        n_zeros += sum(torch.sum(torch.sum(batch.x[:, :, :2], dim=1), dim=1) == 0)
        n_total += batch.x.shape[0]

    print(
        "### Datastats, type = {}: total = {} | zeros = {} ({:.02f} %)".format(
            split_type,
            n_total,
            n_zeros,
            100.0 * n_zeros / n_total,
        )
    )
