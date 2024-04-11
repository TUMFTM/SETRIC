import sys
import os
import sqlite3
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

from utils.processing import label_to_int_open_dd, get_edge_index
from utils.geometry import coord_to_coord


class Dataset_OpenDD(InMemoryDataset):
    def __init__(
        self,
        split="r_1",
        split_type="train",
        h_t_steps=90,
        f_t_steps=150,
        h_val_steps=6,
        f_val_steps=15,
        current_pos_idx_0=False,
        processed_file_folder="data",
        debug=False,
    ):
        # In super().__init__(root) the InMemoryDataset Class will automatically call the methods raw_file_names,
        # processed_file_names, download and process. Since in this specific implementation, the inputs are used in
        # this classes, they have to be stored already at this point.

        """
        Input -------
        SPLIT:          Name of the data split (r_1, r_2, r_3)
        SPLIT_TYPE:     Type of the data split (train, val, test)
        H_T_STEPS:      History time used to create input data of one sample in time steps of size 0.033367 seconds.
        F_T_STEPS:      Future time used to create target data of one sample in time steps of size 0.033367 seconds.
        H_VAL_STEPS:    Time distance between considered data entries in history time to create input sample data in
                        steps of size 0.033367 seconds.
        F_VAL_STEPS:    Time distance between considered data entries in future time to create target sample data in
                        steps of size 0.033367 seconds.
        """

        """ Store Hyperparameters"""
        self.t_delta = 0.033367  # Dataset specific time difference between consecutive timestamps in seconds
        self.h_t_steps = h_t_steps
        self.f_t_steps = f_t_steps
        self.h_val_steps = h_val_steps
        self.f_val_steps = f_val_steps
        self.history_time = self.h_t_steps * self.t_delta  # in seconds
        self.future_time = self.f_t_steps * self.t_delta  # in seconds
        self.history_val_time = self.h_val_steps * self.t_delta  # in seconds
        self.future_val_time = self.f_val_steps * self.t_delta  # in seconds
        self.current_pos_idx_0 = current_pos_idx_0
        self.debug = debug

        """ Check if split and split_type is valid """
        self.split = split
        if split in ["r_1", "r_2", "r_3"] and split_type in ["train", "val"]:
            self.split_type = split_type
        elif split in ["r_A", "r_B", "r_C"] and split_type in ["test"]:
            self.split_type = split_type
        else:
            raise ValueError(
                "invalid split or split_type: split can either be 'r_1', 'r_2', 'r_3' with split_type ..."
                "'train', 'val' or 'r_A', 'r_B', 'r_C' with split_type 'test'"
            )

        """  Get processed file path """
        self.repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.processed_file_folder = processed_file_folder

        if self.debug:
            self.processed_file_folder += "_debug"
            print("Running on debug data")

        self.processed_file_name = "OpenDD" + "_" + split_type + "_" + split + ".pt"

        """  Get split definition """
        split_definition_file = None
        if self.split_type == "train":
            split_definition_file = os.path.join(
                self.processed_file_folder,
                "raw",
                "split_definition",
                self.split + "_train.txt",
            )
        elif self.split_type == "val":
            split_definition_file = os.path.join(
                self.processed_file_folder,
                "raw",
                "split_definition",
                self.split + "_val.txt",
            )
        elif self.split_type == "test":
            split_definition_file = os.path.join(
                self.processed_file_folder,
                "raw",
                "split_definition",
                self.split + ".txt",
            )
        tf = open(split_definition_file, "r")
        self.scene_list = [line.rstrip() for line in tf]
        tf.close()

        super().__init__(root=self.processed_file_folder)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [self.processed_file_name]

    def download(self):
        pass

    def process(self):
        """
        This function generates a list of samples from all scenes specified in the split_definition_file. Hereby,
        every sample is a graph with one node for each agent. Each node contains the following node features (data.x):
        X = Past x positions relative to the current x position
        Y = Past y positions relative to the current y position
        ANGLE = Past bounding box angle relative to the current bounding box angle
        V = Velocity in bounding box direction
        ACC_TAN = Tangential acceleration of agent
        ACC_LAT = Lateral acceleration of agent
        CLASS = Integer representing one class e.g. ['car', 'pedestrian', 'bicycle', 'car'] -> [1, 2, 3, 1]
        WIDTH = Width of the object's bounding box
        LENGTH = Length of the object's bounding box

        The target (data.y) is equals the future trajectory of each agent given as:
        X = Future x positions relative to the current x position
        Y = Future y positions relative to the current y position

        The edges (data.edge_index) are constructed from every agent to every other agent within the scene without
        self-loops
        """

        data_list = []

        """  For all roundabouts """
        for rdb in [1, 2, 3, 4, 5, 6, 7]:
            # filter scenes_list for current rdb
            rdb_scene_list = [
                scene for scene in self.scene_list if int(scene[3]) == rdb
            ]
            # Check if scenes of current roundabout are part of the data split
            if len(rdb_scene_list) > 0:
                rdb_path = os.path.join(
                    self.processed_file_folder, "raw", "rdb" + str(rdb)
                )

                def process_scene_sample(t_c_idx):
                    """function to process one scene that can be parallelized"""
                    t_c = scene_t[t_c_idx]
                    t_min = scene_t[t_c_idx - self.h_t_steps]
                    t_max = scene_t[t_c_idx + self.f_t_steps]
                    # filter time interval
                    df_scene_t = df_scene[df_scene["TIMESTAMP"].between(t_min, t_max)]
                    # filter agents that are not present throughout the entire sample
                    scene_t_objid = df_scene_t.drop_duplicates(subset="OBJID")[
                        "OBJID"
                    ].copy()
                    for objid in scene_t_objid:
                        if (
                            df_scene_t[df_scene_t["OBJID"] == objid]["TIMESTAMP"].max()
                            != t_max
                            or df_scene_t[df_scene_t["OBJID"] == objid][
                                "TIMESTAMP"
                            ].min()
                            != t_min
                        ):
                            objid_drop_idx = scene_t_objid[scene_t_objid == objid].index
                            scene_t_objid.drop(objid_drop_idx, inplace=True)
                    # Initialize lists
                    (
                        x_p_l_list,
                        x_p_t_list,
                        x_angle_list,
                        x_v_l_list,
                        x_v_t_list,
                        x_a_l_list,
                        x_a_t_list,
                        x_class_list,
                        y_p_l_list,
                        y_p_t_list,
                        y_angle_list,
                        ref_list,
                    ) = ([] for _ in range(12))
                    # Check if the sample contains agents and whether data is available for the time interval (In
                    # rare occasions, the timestamps make a leap which is either cause by malfunctions during the
                    # drone flight or the processing of the raw data).
                    if (
                        len(scene_t_objid) > 0
                        and t_min - self.t_delta < t_c - self.history_time
                        and t_c + self.future_time < t_max + self.t_delta
                    ):
                        for objid in scene_t_objid:
                            # Obtain history and future data of the object within the respective time interval.
                            # Hereby, t_c is viewed as the last received data
                            df_scene_t_h_objid = df_scene_t[
                                (df_scene_t["TIMESTAMP"].between(t_min, t_c))
                                & (df_scene_t["OBJID"] == objid)
                            ]
                            df_scene_t_f_objid = df_scene_t[
                                (df_scene_t["TIMESTAMP"].between(t_c, t_max))
                                & (df_scene_t["OBJID"] == objid)
                            ]
                            # filter interval steps (the current time data is considered the latest received data,
                            # thus the history dataframe has one more row than h_val_steps)
                            df_scene_t_h_objid = df_scene_t_h_objid.iloc[
                                :: self.h_val_steps, :
                            ]
                            df_scene_t_f_objid = df_scene_t_f_objid.iloc[
                                self.f_val_steps :: self.f_val_steps, :
                            ]
                            # get current time and UTM x, y and angle values which is equivalent to the data of the
                            # latest history timestamp
                            current_x = df_scene_t_h_objid[
                                df_scene_t_h_objid["TIMESTAMP"] == t_c
                            ]["UTM_X"].item()
                            current_y = df_scene_t_h_objid[
                                df_scene_t_h_objid["TIMESTAMP"] == t_c
                            ]["UTM_Y"].item()
                            current_angle = df_scene_t_h_objid[
                                df_scene_t_h_objid["TIMESTAMP"] == t_c
                            ]["UTM_ANGLE"].item()
                            # transform history positions, angles, velocities (=lateral velocity) and lateral and
                            # tangential accelerations to current agent coordinate system and convert them to tensor of
                            # shape [1, :]
                            (
                                x_p_l_dummy,
                                x_p_t_dummy,
                                x_angle_dummy,
                                x_v_l_dummy,
                                x_v_t_dummy,
                                x_a_l_dummy,
                                x_a_t_dummy,
                            ) = ([] for _ in range(7))

                            if self.current_pos_idx_0:
                                iter_input = reversed(df_scene_t_h_objid["UTM_X"].index)
                            else:
                                iter_input = df_scene_t_h_objid["UTM_X"].index

                            for p in iter(iter_input):
                                (
                                    (x_p_l, x_p_t, x_angle),
                                    (x_v_l, x_v_t),
                                    (x_a_l, x_a_t),
                                ) = coord_to_coord(
                                    new_coord=(
                                        torch.tensor(current_x, dtype=torch.float64),
                                        torch.tensor(current_y, dtype=torch.float64),
                                        torch.tensor(
                                            current_angle, dtype=torch.float64
                                        ),
                                    ),
                                    old_coord=(
                                        torch.tensor(
                                            df_scene_t_h_objid["UTM_X"][p],
                                            dtype=torch.float64,
                                        ),
                                        torch.tensor(
                                            df_scene_t_h_objid["UTM_Y"][p],
                                            dtype=torch.float64,
                                        ),
                                        torch.tensor(
                                            df_scene_t_h_objid["UTM_ANGLE"][p],
                                            dtype=torch.float64,
                                        ),
                                    ),
                                    agent_p=(
                                        torch.tensor(0, dtype=torch.float64),
                                        torch.tensor(0, dtype=torch.float64),
                                        torch.tensor(0, dtype=torch.float64),
                                    ),
                                    agent_v=(
                                        torch.tensor(
                                            df_scene_t_h_objid["V"][p],
                                            dtype=torch.float64,
                                        ),
                                        torch.zeros_like(
                                            torch.tensor(
                                                df_scene_t_h_objid["V"][p],
                                                dtype=torch.float64,
                                            )
                                        ),
                                    ),
                                    agent_a=(
                                        torch.tensor(
                                            df_scene_t_h_objid["ACC_LAT"][p],
                                            dtype=torch.float64,
                                        ),
                                        torch.tensor(
                                            df_scene_t_h_objid["ACC_TAN"][p],
                                            dtype=torch.float64,
                                        ),
                                    ),
                                )
                                x_p_l_dummy.append(x_p_l)
                                x_p_t_dummy.append(x_p_t)
                                x_angle_dummy.append(x_angle)
                                x_v_l_dummy.append(x_v_l)
                                x_v_t_dummy.append(x_v_t)
                                x_a_l_dummy.append(x_a_l)
                                x_a_t_dummy.append(x_a_t)
                            x_p_l_list.append(
                                torch.tensor(x_p_l_dummy, dtype=torch.float)
                            )
                            x_p_t_list.append(
                                torch.tensor(x_p_t_dummy, dtype=torch.float)
                            )
                            x_angle_list.append(
                                torch.tensor(x_angle_dummy, dtype=torch.float)
                            )
                            x_v_l_list.append(
                                torch.tensor(x_v_l_dummy, dtype=torch.float)
                            )
                            x_v_t_list.append(
                                torch.tensor(x_v_t_dummy, dtype=torch.float)
                            )
                            x_a_l_list.append(
                                torch.tensor(x_a_l_dummy, dtype=torch.float)
                            )
                            x_a_t_list.append(
                                torch.tensor(x_a_t_dummy, dtype=torch.float)
                            )
                            # get agent class
                            x_class_list.append(
                                torch.tensor(
                                    [
                                        label_to_int_open_dd(
                                            df_scene_t_h_objid["CLASS"].values[0]
                                        )
                                    ],
                                    dtype=torch.float,
                                )
                            )
                            # calculate future positions relative to the current x, y and angle and
                            # convert them to tensor of shape [1, :]
                            y_p_l_dummy, y_p_t_dummy, y_angle_dummy = (
                                [] for _ in range(3)
                            )
                            for p in iter(df_scene_t_f_objid["UTM_X"].index):
                                (y_p_l, y_p_t, y_angle) = coord_to_coord(
                                    new_coord=(
                                        torch.tensor(current_x, dtype=torch.float64),
                                        torch.tensor(current_y, dtype=torch.float64),
                                        torch.tensor(
                                            current_angle, dtype=torch.float64
                                        ),
                                    ),
                                    old_coord=(
                                        torch.tensor(
                                            df_scene_t_f_objid["UTM_X"][p],
                                            dtype=torch.float64,
                                        ),
                                        torch.tensor(
                                            df_scene_t_f_objid["UTM_Y"][p],
                                            dtype=torch.float64,
                                        ),
                                        torch.tensor(
                                            df_scene_t_f_objid["UTM_ANGLE"][p],
                                            dtype=torch.float64,
                                        ),
                                    ),
                                    agent_p=(
                                        torch.tensor(0, dtype=torch.float64),
                                        torch.tensor(0, dtype=torch.float64),
                                        torch.tensor(0, dtype=torch.float64),
                                    ),
                                )
                                y_p_l_dummy.append(y_p_l)
                                y_p_t_dummy.append(y_p_t)
                                y_angle_dummy.append(y_angle)
                            y_p_l_list.append(
                                torch.tensor(y_p_l_dummy, dtype=torch.float)
                            )
                            y_p_t_list.append(
                                torch.tensor(y_p_t_dummy, dtype=torch.float)
                            )
                            y_angle_list.append(
                                torch.tensor(y_angle_dummy, dtype=torch.float)
                            )
                            # Convert additional information, namely the roundabout scene affiliation as
                            # roundabout number and scene number, current time, object id, bounding box
                            # dimension abd current agent geo reference
                            width = df_scene_t_h_objid["WIDTH"].values[0]
                            length = df_scene_t_h_objid["LENGTH"].values[0]
                            ref_list.append(
                                torch.tensor(
                                    [
                                        rdb,
                                        int(rdb_scene[5:]),
                                        t_c,
                                        objid,
                                        width,
                                        length,
                                        current_x,
                                        current_y,
                                        current_angle,
                                    ],
                                    dtype=torch.float64,
                                )
                            )
                        # Convert node feature lists to tensors
                        x_p_l_tensor = torch.stack(x_p_l_list).reshape(
                            len(x_p_l_list), 1, len(x_p_l_list[0])
                        )
                        x_p_t_tensor = torch.stack(x_p_t_list).reshape(
                            len(x_p_t_list), 1, len(x_p_t_list[0])
                        )
                        x_angle_tensor = torch.stack(x_angle_list).reshape(
                            len(x_angle_list), 1, len(x_angle_list[0])
                        )
                        x_v_l_tensor = torch.stack(x_v_l_list).reshape(
                            len(x_v_l_list), 1, len(x_v_l_list[0])
                        )
                        x_v_t_tensor = torch.stack(x_v_t_list).reshape(
                            len(x_v_t_list), 1, len(x_v_t_list[0])
                        )
                        x_a_l_tensor = torch.stack(x_a_l_list).reshape(
                            len(x_a_l_list), 1, len(x_a_l_list[0])
                        )
                        x_a_t_tensor = torch.stack(x_a_t_list).reshape(
                            len(x_a_t_list), 1, len(x_a_t_list[0])
                        )
                        x_class_tensor = torch.stack(x_class_list).reshape(
                            len(x_class_list), 1, len(x_class_list[0])
                        )
                        ref_tensor = torch.stack(ref_list).reshape(
                            len(ref_list), 1, len(ref_list[0])
                        )
                        y_p_l_tensor = torch.stack(y_p_l_list).reshape(
                            len(y_p_l_list), 1, len(y_p_l_list[0])
                        )
                        y_p_t_tensor = torch.stack(y_p_t_list).reshape(
                            len(y_p_t_list), 1, len(y_p_t_list[0])
                        )
                        y_angle_tensor = torch.stack(y_angle_list).reshape(
                            len(y_angle_list), 1, len(y_angle_list[0])
                        )
                        # concatenate Time series data
                        x_time_series = torch.cat(
                            (
                                x_p_l_tensor,
                                x_p_t_tensor,
                                x_angle_tensor,
                                x_v_l_tensor,
                                x_v_t_tensor,
                                x_a_l_tensor,
                                x_a_t_tensor,
                            ),
                            dim=1,
                        )
                        y_time_series = torch.cat(
                            (y_p_l_tensor, y_p_t_tensor, y_angle_tensor), dim=1
                        )
                        # create graph data
                        data_ = Data(
                            x=x_time_series,
                            y=y_time_series,
                            obj_class=x_class_tensor,
                            obj_ref=ref_tensor,
                            edge_index=get_edge_index(len(scene_t_objid))
                            .t()
                            .contiguous(),
                        )
                        return data_

                    else:
                        return None

                """  For all scenes with this roundabout """
                for rdb_scene in tqdm(rdb_scene_list, desc="rdb " + str(rdb)):
                    # roundabout specific sql operations
                    rdb_sql_path = os.path.join(
                        rdb_path, "trajectories_rdb" + str(rdb) + "_v3.sqlite"
                    )
                    conn_rdb = sqlite3.connect(rdb_sql_path)
                    with conn_rdb:
                        # scene specific sql operations
                        sql_scene = "SELECT * FROM " + rdb_scene
                        df_scene = pd.read_sql_query(sql_scene, conn_rdb)
                    # calculate unique timestamps
                    scene_t = (
                        df_scene.drop_duplicates(subset="TIMESTAMP")["TIMESTAMP"]
                        .copy()
                        .to_list()
                    )
                    # calculated list containing all samples from one scene in parallel
                    if self.debug:
                        cpu_jobs = 1
                    else:
                        cpu_jobs = min(os.cpu_count(), 12)
                    scene_data_list = Parallel(n_jobs=cpu_jobs)(
                        delayed(process_scene_sample)(t_c_idx)
                        for t_c_idx in range(
                            self.h_t_steps,
                            len(scene_t) - self.f_t_steps,
                            self.h_t_steps,
                        )
                    )
                    # filter None values
                    scene_data_list = list(filter(None, scene_data_list))

                    if len(scene_data_list) > 0 and self.debug:
                        idx_min = [
                            j
                            for j in range(len(scene_data_list))
                            if sum([sc.x.shape[0] for sc in scene_data_list[:j]]) > 16
                        ][0]
                        data_list.append(scene_data_list[:idx_min])
                        break

                    data_list.append(scene_data_list)

        """ Flatten list of lists containing data samples into one big list """
        data_list = [
            item
            for elem in tqdm(data_list, desc="Flatten scene lists")
            for item in elem
        ]

        """ Store Data """
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    dataset = Dataset_OpenDD(split="r_1", split_type="train", debug=True)
    dataset = Dataset_OpenDD(split="r_1", split_type="val", debug=True)

    # Dataset_OpenDD(split="r_2", split_type="train")
    # Dataset_OpenDD(split="r_2", split_type="val")

    # Dataset_OpenDD(split="r_3", split_type="train")
    # Dataset_OpenDD(split="r_3", split_type="val")

    # Dataset_OpenDD(split="r_A", split_type="test")
    # Dataset_OpenDD(split="r_B", split_type="test")
    # Dataset_OpenDD(split="r_C", split_type="test")
