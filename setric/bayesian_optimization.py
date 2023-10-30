import os
import sys
import argparse
import datetime
import json
import numpy as np

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

from bayes_opt.logger import JSONLogger, ScreenLogger
from bayes_opt.event import Events
from bayes_opt import BayesianOptimization

# Custom imports
from utils.scheduling import store_current_config
from setric.main_fusion import main_train_fusion

# Args parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_list",
    type=int,
    default=1,
    help="0: 'l_lstm', 'dg_lstm'; 1: 'cv', 'l_stm', 'dg_lstm'",
)
parser.add_argument("--config", type=str, default="config/train_config.json")
parser.add_argument("--net_config", type=str, default="config/net_config.json")
parser.add_argument("--debug", default=False, action="store_true")
parser.add_argument("--data", type=str, default="cr")
parser.add_argument("--load_model", default=False, action="store_true")
args = parser.parse_args()

args.bayesian = True

time_stamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def convert_to_config_vals(
    gnn_distance_n_10,
    gnn_message_size_n_16,
    gnn_aggr_choice,
    gnn_embedding_size_n_32,
    gnn_num_hidden_layers_add,
    gnn_hidden_size_n_16,
    dyn_embedding_size_n_16,
    decoder_size_n_64,
    g_sel_dyn_emb_choice,
    g_sel_output_linear_size_n_32,
    num_img_filters_n_8,
    dec_img_size_n_8,
    weight,
):
    """Converts parameters from normalized Bayesian hyperparams to net input."""
    add_lin_output = 64 if g_sel_dyn_emb_choice < 0.5 else 0
    return {
        "gnn_distance": 20 + 10 * int(np.round(gnn_distance_n_10)),
        "gnn_message_size": 16 + 16 * int(np.round(gnn_message_size_n_16)),
        "gnn_aggr": "max" if gnn_aggr_choice < 0.5 else "add",
        "gnn_embedding_size": 32 + 32 * int(np.round(gnn_embedding_size_n_32)),
        "gnn_num_hidden_layers": 1 + int(np.round(gnn_num_hidden_layers_add)),
        "gnn_hidden_size": 16 + 16 * int(np.round(gnn_hidden_size_n_16)),
        "dyn_embedding_size": 16 + 16 * int(np.round(dyn_embedding_size_n_16)),
        "decoder_size": 64 + 64 * int(np.round(decoder_size_n_64)),
        "g_sel_dyn_emb": "no_lstm" if g_sel_dyn_emb_choice < 0.5 else "obj_state",
        "g_sel_output_linear_size": 32
        + add_lin_output
        + 32 * int(np.round(g_sel_output_linear_size_n_32)),
        "num_img_filters": 8 + 8 * int(np.round(num_img_filters_n_8)),
        "dec_img_size": 8 + 8 * int(np.round(dec_img_size_n_8)),
        "weight": float(weight),
    }


def eval_bayes_opt(
    gnn_distance_n_10,
    gnn_message_size_n_16,
    gnn_aggr_choice,
    gnn_embedding_size_n_32,
    gnn_num_hidden_layers_add,
    gnn_hidden_size_n_16,
    dyn_embedding_size_n_16,
    decoder_size_n_64,
    g_sel_dyn_emb_choice,
    g_sel_output_linear_size_n_32,
    num_img_filters_n_8,
    dec_img_size_n_8,
    weight,
):
    """Runs iteration step of Bayesian optimizer, i.e. main_train_fusion function."""
    config_update = convert_to_config_vals(
        gnn_distance_n_10,
        gnn_message_size_n_16,
        gnn_aggr_choice,
        gnn_embedding_size_n_32,
        gnn_num_hidden_layers_add,
        gnn_hidden_size_n_16,
        dyn_embedding_size_n_16,
        decoder_size_n_64,
        g_sel_dyn_emb_choice,
        g_sel_output_linear_size_n_32,
        num_img_filters_n_8,
        dec_img_size_n_8,
        weight,
    )
    config_update.update(
        {
            "epochs": 30,
            "batch_size": 64,
        }
    )

    args.time_stamp_str = time_stamp_str

    best_rmse_val, best_val_correct_selections = main_train_fusion(
        args=args, config_update=config_update
    )

    r_1 = (best_val_correct_selections - 0.7) / (1.0 - 0.7)
    r_2 = 0.3 / best_rmse_val
    target = r_1 + r_2
    target = target.cpu().detach().numpy()

    if np.isnan(target):
        return -0.7 / 0.3 + 0.0

    return target


def get_best_config(json_logger_path):
    """Extracts best config von bayes_log.json and outputs to net_config.json and train_config.json"""
    # encode results
    tot_res_list = []
    with open(json_logger_path, "r") as f:
        for jsonObj in f:
            line_res = json.loads(jsonObj)
            tot_res_list.append(line_res)

    def get_target(line_res):
        return line_res.get("target")

    # Sort results, best (=highest target) first
    tot_res_list.sort(key=get_target, reverse=True)
    best_guess = tot_res_list[0]
    best_config_update = convert_to_config_vals(
        gnn_distance_n_10=best_guess["params"]["gnn_distance_n_10"],
        gnn_message_size_n_16=best_guess["params"]["gnn_message_size_n_16"],
        gnn_aggr_choice=best_guess["params"]["gnn_aggr_choice"],
        gnn_embedding_size_n_32=best_guess["params"]["gnn_embedding_size_n_32"],
        gnn_num_hidden_layers_add=best_guess["params"]["gnn_num_hidden_layers_add"],
        gnn_hidden_size_n_16=best_guess["params"]["gnn_hidden_size_n_16"],
        dyn_embedding_size_n_16=best_guess["params"]["dyn_embedding_size_n_16"],
        decoder_size_n_64=best_guess["params"]["decoder_size_n_64"],
        g_sel_dyn_emb_choice=best_guess["params"]["g_sel_dyn_emb_choice"],
        g_sel_output_linear_size_n_32=best_guess["params"][
            "g_sel_output_linear_size_n_32"
        ],
        num_img_filters_n_8=best_guess["params"]["num_img_filters_n_8"],
        dec_img_size_n_8=best_guess["params"]["dec_img_size_n_8"],
        weight=best_guess["params"]["weight"],
    )

    # search in files for best iteration
    res_dir = os.path.dirname(json_logger_path)
    n = 0
    for sub_res_dir in os.listdir(res_dir):
        if sub_res_dir.endswith(".json") or sub_res_dir.endswith(".txt"):
            continue
        res_net_config_path = os.path.join(
            res_dir, sub_res_dir, os.path.basename(args.net_config)
        )
        with open(res_net_config_path, "r") as f:
            res_net_config = json.load(f)
        res_train_config_path = os.path.join(
            res_dir, sub_res_dir, os.path.basename(args.config)
        )
        with open(res_train_config_path, "r") as f:
            res_train_config = json.load(f)

        same_net_config = [
            best_config_update[keys]
            for keys in best_config_update
            if keys in res_net_config
        ] == [
            res_net_config[keys]
            for keys in best_config_update
            if keys in res_net_config
        ]
        same_train_config = [
            best_config_update[keys]
            for keys in best_config_update
            if keys in res_train_config
        ] == [
            res_train_config[keys]
            for keys in best_config_update
            if keys in res_train_config
        ]

        if same_net_config and same_train_config:
            p2 = json.dumps(best_guess, indent=4)
            strstr = "Found best config at: {},\nopt: {}\n".format(sub_res_dir, p2)
            print(strstr)
            with open(
                os.path.join(res_dir, "best_iteration.txt"), "a", encoding="utf-8"
            ) as f:
                f.write(strstr)
            store_current_config(
                base_path=res_dir,
                cfg_train=res_train_config,
                net_config=res_net_config,
                args=args,
                n_iter=0,
                add_best_str="best_{}_".format(n),
            )
            n += 1

    if n > 0:
        return

    raise ValueError("No matched config found")


if __name__ == "__main__":
    # Define bounds for optimization
    pbounds = {
        "gnn_distance_n_10": (0, 6),
        "gnn_message_size_n_16": (0, 3),
        "gnn_embedding_size_n_32": (0, 5),
        "gnn_aggr_choice": (0, 1),
        "gnn_num_hidden_layers_add": (0, 1),
        "gnn_hidden_size_n_16": (0, 3),
        "dyn_embedding_size_n_16": (0, 3),
        "decoder_size_n_64": (0, 2),
        "g_sel_dyn_emb_choice": (0, 1),
        "g_sel_output_linear_size_n_32": (0, 3),
        "num_img_filters_n_8": (0, 5),
        "dec_img_size_n_8": (0, 5),
        "weight": (1.0, 4.0),
    }

    # Define optimizer
    optimizer = BayesianOptimization(
        f=eval_bayes_opt, pbounds=pbounds, random_state=1, verbose=2
    )

    optimizer.probe(
        params={
            "dec_img_size_n_8": 2.519386983032522,
            "decoder_size_n_64": 1.2805140650861133,
            "dyn_embedding_size_n_16": 1.818744969006035,
            "g_sel_dyn_emb_choice": 0.0,
            "g_sel_output_linear_size_n_32": 2.349751704407859,
            "gnn_aggr_choice": 1.0,
            "gnn_distance_n_10": 5.242212291055031,
            "gnn_embedding_size_n_32": 2.0749750754911496,
            "gnn_hidden_size_n_16": 3.0,
            "gnn_message_size_n_16": 2.6439550673557846,
            "gnn_num_hidden_layers_add": 0.740896170981939,
            "num_img_filters_n_8": 1.1980883364729427,
            "weight": 1.0,
        },
        lazy=True,
    )

    dirname = "results_bayesian"
    if args.debug:
        dirname += "_debug"
    sub_folder = time_stamp_str
    bayesian_results_path = os.path.join(repo_path, dirname, sub_folder)
    os.makedirs(bayesian_results_path)

    # Screen and json logging
    logger = JSONLogger(path=os.path.join(bayesian_results_path, "bayes_logs.json"))
    screen_logger = ScreenLogger(verbose=2)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    optimizer.subscribe(Events.OPTIMIZATION_START, logger)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    optimizer.subscribe(Events.OPTIMIZATION_END, logger)

    # Execute optimizer
    optimizer.maximize(init_points=25, n_iter=75)

    get_best_config(json_logger_path=logger._path)
