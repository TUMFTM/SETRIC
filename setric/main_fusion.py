import os
import sys
import argparse

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

from torch_geometric import seed_everything

from models.Fusion_Model import Fusion_Model
from utils.train import train
from utils.scheduling import (
    get_config,
    adapt_to_CR,
    store_current_config,
    update_configs,
    modify_for_debug,
    load_net_params,
    get_log_root_path,
)


def main_train_fusion(args, seed_list=[10], split_list=["r_1"], config_update=None):
    """Main function to train fusion model."""
    net_config, cfg_train, base_path = get_config(
        args=args,
        results_dir_path=os.path.join(repo_path, "results"),
        is_fusion=True,
    )
    _ = update_configs(config_update, net_config, cfg_train)
    net_config, split_list = adapt_to_CR(net_config, cfg_train, split_list)
    _ = modify_for_debug(cfg_train)

    for seed in seed_list:
        for split in split_list:
            seed_everything(seed)
            cfg_train.update({"split": split, "seed": seed})

            ######################################################################################
            # Load Model
            g_fusion = Fusion_Model(
                cfg=cfg_train,
                net_config=net_config,
            )
            ######################################################################################
            # Load Parameters
            iterator = load_net_params(g_fusion, cfg_train)
            ######################################################################################
            # Start Training
            for n_iter, model_tag in enumerate(iterator):
                g_fusion.update_model_tag(
                    n_iter, model_tag, is_iterative=len(iterator) > 1
                )

                log_root_path = get_log_root_path(
                    base_path, cfg_train["split"], g_fusion.tag
                )

                _ = store_current_config(base_path, cfg_train, net_config, args, n_iter)

                outp = train(
                    model=g_fusion,
                    cfg=cfg_train,
                    log_root_path=log_root_path,
                )

                if model_tag == "g_sel":
                    best_rmse_val = outp[0]
                    best_val_correct_selections = outp[1]
            ######################################################################################

    return best_rmse_val, best_val_correct_selections


if __name__ == "__main__":
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

    main_train_fusion(args=args)
