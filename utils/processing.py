import os
import sys

repo_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(repo_path)

import argparse
import json
import shutil
from tqdm import tqdm
from datetime import datetime
import torch


def label_to_int(label):
    """This function convert the label to a label specific integer. The function can be used with the OpenDD Dataset
    or any other Dataset which has an equivalent stack of possible labels that is equally spelled!
    """

    if label == "Pedestrian":
        return 1
    elif label == "Bicycle":
        return 2
    elif label == "Motorcycle":
        return 3
    elif label == "Car":
        return 4
    elif label == "Van":
        return 5
    elif label == "Medium Vehicle":
        return 6
    elif label == "Truck":
        return 7
    elif label == "Bus":
        return 8
    elif label == "Trailer":
        return 9
    elif label == "Heavy Vehicle":
        return 10
    else:
        raise ValueError(
            "Invalid Label - Label has to be 'Pedestrian', 'Bicycle', 'Motorcycle', 'Car', 'Van', "
            "'Medium Vehicle', 'Truck', 'Bus', 'Trailer', 'Heavy Vehicle' - got: "
            + label
        )


def int_to_label(label_specific_integer):
    """This function convert the label specific integer back to the label. The function can be used with the OpenDD
    Dataset or any other Dataset which has an equivalent stack of possible labels that is equally spelled!
    """

    if label_specific_integer == 1:
        return "Pedestrian"
    elif label_specific_integer == 2:
        return "Bicycle"
    elif label_specific_integer == 3:
        return "Motorcycle"
    elif label_specific_integer == 4:
        return "Car"
    elif label_specific_integer == 5:
        return "Van"
    elif label_specific_integer == 6:
        return "Medium Vehicle"
    elif label_specific_integer == 7:
        return "Truck"
    elif label_specific_integer == 8:
        return "Bus"
    elif label_specific_integer == 9:
        return "Trailer"
    elif label_specific_integer == 10:
        return "Heavy Vehicle"
    else:
        raise ValueError(
            "Invalid Integer - Integer value has to be between 1 and 10 - got: "
            + str(label_specific_integer)
        )


def get_edge_index(agent_num):
    """This function creates an edge index list for all agents with edges to all other agents, excluding self-loop.
    Afterwards the list is cast as a tensor that can directly be stored as the edge_index input of a
    torch_geometric.data.Data object"""

    edge_index = []
    for k in range(agent_num):
        for n in range(agent_num):
            # exclude self loop and prevent edge duplicates from previous iterations
            if n > k:
                edge_index.append([k, n])
                edge_index.append([n, k])
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return edge_index


def extract_CR_from_repo(args):
    source_path = os.path.join(
        args.path, "commonroad-scenarios", "scenarios", "recorded"
    )

    # copy to target dir
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%Y%m%d_%H%M%S")

    target_path = os.path.join(
        args.path, "gnn_data", "raw", timestampStr + "_commonroad-scenarios"
    )
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    print("Copying files to {} ...".format(target_path))
    for path, _, file_list in tqdm(os.walk(source_path)):
        if len(file_list) > 0:
            for file in file_list:
                if "interactive" in path:
                    if ".cr.xml" in file:
                        trgt_file = file.replace(".cr.xml", ".xml")
                    else:
                        continue
                else:
                    trgt_file = file

                file_src = os.path.join(path, file)
                file_dst = os.path.join(target_path, trgt_file)

                shutil.copy(file_src, file_dst)

    if not min([xx.endswith(".xml") for xx in os.listdir(target_path)]):
        raise ValueError("invalid file format detected, all files must be .xml")

    print("Done. Copied {} files".format(len(os.listdir(target_path))))


def get_debug_data(data="cr"):
    from utils.Dataset_OpenDD import Dataset_OpenDD
    from torch_geometric.loader import DataLoader
    from utils.Dataset_CR import Dataset_CR
    from torch_geometric import seed_everything

    from utils.scheduling import get_model_tag_list, adapt_to_CR
    from utils.map_processing import get_rdp_map, get_sc_img_batch

    seed = 10
    if "open" in data:
        split = "r_1"
    else:
        split = "cr"
    seed_everything(seed)

    # get config #
    with open(os.path.join(repo_path, "config/train_config.json"), "r") as f:
        cfg_train = json.load(f)
    cfg_train.update({"split": split, "seed": seed})
    cfg_train["epochs"] = 200

    cfg_train["model_list"] = 1
    cfg_train["data"] = data
    cfg_train["device"] = "cuda"
    _ = get_model_tag_list(cfg_train)

    with open(os.path.join(repo_path, "config/net_config.json"), "r") as f:
        net_config = json.load(f)

    net_config, _ = adapt_to_CR(net_config, cfg_train, [split])

    # get Data #
    if "open" in data:
        train_dataset = Dataset_OpenDD(
            split=cfg_train["split"], split_type="train", debug=True
        )
    else:
        train_dataset = Dataset_CR(
            split=cfg_train["split"], split_type="train", debug=True
        )

    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg_train["batch_size"], shuffle=False
    )
    batch = next(iter(train_dataloader))
    batch.x = batch.x.permute(0, 2, 1)
    batch.y = batch.y.permute(0, 2, 1)

    # Add object class to the time series data
    obj_class = batch.obj_class.repeat(
        1, batch.x.shape[1], 1
    )  # obj_class shape: (N, seq_length, 1)
    batch.x = torch.cat((batch.x, obj_class), dim=2)

    valid_rdbs = [1, 2, 3, 4, 5, 6, 7]
    if cfg_train["sc_img"] and "open" in data:
        rdp_map_dict = {
            rdb_int: get_rdp_map(rdb_int, data_path=train_dataset.processed_file_folder)
            for rdb_int in valid_rdbs
        }
        batch.sc_img = get_sc_img_batch(
            batch.obj_ref,
            rdp_map_dict,
            train_dataset.processed_file_folder,
            cfg=cfg_train,
            mod="train",
        )

    return batch, cfg_train, net_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="/home/ubuntu/data/commonroad/",
        help="path to clone https://gitlab.lrz.de/tum-cps/commonroad-scenarios into",
    )

    args = parser.parse_args()

    extract_CR_from_repo(args=args)
