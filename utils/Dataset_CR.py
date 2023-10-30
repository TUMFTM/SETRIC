import sys
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import random
import datetime

from commonroad.common.file_reader import CommonRoadFileReader

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import numpy as np

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

from utils.processing import get_edge_index
from utils.geometry import coord_to_coord
from utils.map_processing import (
    generate_self_rendered_sc_img,
)


class Dataset_CR(InMemoryDataset):
    def __init__(
        self,
        split="",
        split_type="train",
        past_points=30,
        future_points=50,
        sliwi_size=1,
        min_len_t_hist=30,
        watch_radius=64,
        current_pos_idx_0=False,
        processed_file_folder="data",
        debug=False,
    ):
        # InMemoryDataset class for commonroad data

        """Store Hyperparameters"""
        self.pp = past_points
        self.fp = future_points
        self.sliwi_size = sliwi_size
        self.watch_radius = watch_radius
        self.current_pos_idx_0 = current_pos_idx_0
        self.debug = debug

        self.min_len_t_hist = min_len_t_hist

        """ Check if split and split_type is valid """
        self.split_type = split_type
        self.split = split

        """  Get processed file path """
        self.repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.processed_file_folder = processed_file_folder

        if self.debug:
            self.processed_file_folder += "_debug"
            print("Running on debug data")

        self.processed_file_name = "CR" + "_" + self.split_type + ".pt"

        train_scenario_file, val_scenario_file, test_scenario_file = create_split_files(
            self.processed_file_folder,
        )

        """  Get split definition """
        if self.split_type == "train":
            split_definition_file = train_scenario_file
        elif self.split_type == "val":
            split_definition_file = val_scenario_file
        elif self.split_type == "test":
            split_definition_file = test_scenario_file
        tf = open(split_definition_file, "r")
        self.scenario_list = [line.rstrip() for line in tf]
        tf.close()

        # initialize InMemoryDataset
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
        """Generate a list of samples. Every sample is a graph with one node for each agent.
        Each node contains the following node features (data.x):
            X = Past x positions relative to the current x position
            Y = Past y positions relative to the current y position
            ANGLE = Past bounding box angle relative to the current bounding box angle
            V = Velocity (probably in tangential direction)
            ACC = Acceleration (probably in tangential direction)
            CLASS = Integer representing one class e.g. ['car', 'pedestrian', 'bicycle', 'car'] -> [1, 2, 3, 1]

        The target (data.y) is equals the future trajectory of each agent given as:
            X = Future x positions relative to the current x position
            Y = Future y positions relative to the current y position
            ANGLE = Future bounding box angle relative to the current bounding box angle

        The edges (data.edge_index) are constructed from every agent to every other agent within the scene without self-loops
        Note: scene = scenario = all data within one drone flight
        """
        if self.debug:
            data_list = []
            for file_name in tqdm(self.scenario_list):
                """calculate datalist from all scenarios in scenario_list"""
                scenario_data_list = process_scenario(
                    file_name=file_name,
                    pp=self.pp,
                    fp=self.fp,
                    watch_radius=self.watch_radius,
                    min_len_t_hist=self.min_len_t_hist,
                    sliwi_size=self.sliwi_size,
                )
                if len(scenario_data_list) == 0:
                    continue
                elif len(scenario_data_list) > 0 and self.debug:
                    idx_min = get_debug_idx(scenario_data_list)
                    data_list.append(scenario_data_list[:idx_min])
                    break
                data_list.append(scenario_data_list)
        else:
            n_files = len(self.scenario_list)
            p_iter = iter(
                zip(
                    self.scenario_list,
                    [self.pp for _ in range(n_files)],
                    [self.fp for _ in range(n_files)],
                    [self.watch_radius for _ in range(n_files)],
                    [self.min_len_t_hist for _ in range(n_files)],
                    [self.sliwi_size for _ in range(n_files)],
                )
            )
            cpu_jobs = min(os.cpu_count(), 12)
            data_list = Parallel(n_jobs=cpu_jobs)(
                delayed(process_scenario)(
                    file_name,
                    pp,
                    fp,
                    watch_radius,
                    min_len_t_hist,
                    sliwi_size,
                )
                for (
                    file_name,
                    pp,
                    fp,
                    watch_radius,
                    min_len_t_hist,
                    sliwi_size,
                ) in tqdm(p_iter)
            )
            data_list = [dd for dd in data_list if dd]

        """ Flatten list of lists containing data samples into one big list """
        data_list = [
            item
            for elem in tqdm(data_list, desc="Flatten scene lists")
            for item in elem
        ]

        """ Store Data """
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def get_debug_idx(scenario_data_list):
    return [
        j
        for j in range(len(scenario_data_list))
        if sum([sc.x.shape[0] for sc in scenario_data_list[:j]]) > 16
    ][0]


def add_scenario_id(cr_file_, scenario):
    if "scenario_id" in scenario.__dict__.keys():
        return
    cr_file_._read_header()
    if "scenario_id" in cr_file_.__dict__.keys():
        scenario.scenario_id = cr_file_.scenario_id
    else:
        scenario.scenario_id = cr_file_._benchmark_id


def get_trajectories_list(scenario):
    return [
        [
            scenario.dynamic_obstacles[i].prediction.trajectory.state_list[j].position
            for j in range(
                0,
                len(scenario.dynamic_obstacles[i].prediction.trajectory.state_list),
            )
        ]
        for i in range(0, len(scenario.dynamic_obstacles))
        if "_trajectory" in scenario.dynamic_obstacles[i].prediction.__dict__.keys()
    ]


def process_scenario(
    file_name,
    pp,
    fp,
    watch_radius,
    min_len_t_hist,
    sliwi_size,
):
    """function to process one scenario/commonroad file"""

    # open commonroad file
    cr_file_ = CommonRoadFileReader(file_name)

    # get scenario
    try:
        scenario, _ = cr_file_.open()
    except:
        print("{} not found".format(file_name))
        return []

    # get benchmark id
    _ = add_scenario_id(cr_file_, scenario)

    # get list of all trajectory in scenario
    trajectories_list = get_trajectories_list(scenario)

    if len(trajectories_list) == 0:
        print("{} too short".format(file_name))
        return []

    # get total number of timesteps in scenario
    timesteps_in_scenario = max([len(tj) for tj in trajectories_list])

    # iterate over all possible sliding_windows within the total time_window of the scenario
    scenario_data_list = []
    scene_id = 0

    for current_time_step in range(
        min_len_t_hist - 1,
        timesteps_in_scenario - fp,
        sliwi_size,
    ):
        t_min = max(0, current_time_step - pp + 1)
        scene_id_list = []

        for n_iter, t_start in enumerate(
            range(t_min, current_time_step - min_len_t_hist + 2)
        ):
            len_t_hist = current_time_step - t_start + 1

            # get list of dynamic obstacles that are in the scenario for the entire timeintervall
            scene_t_objid = []

            for i in range(0, len(scenario.dynamic_obstacles)):
                final_time_step = scenario.dynamic_obstacles[
                    i
                ].prediction._final_time_step

                # check if object's trajectory exists for the entire time_window
                if ((final_time_step - 1) >= len_t_hist + fp) and (
                    current_time_step <= (final_time_step - 1) - fp
                ):
                    scene_t_objid.append(scenario.dynamic_obstacles[i])
                    # define scenes_id only during first iteration
                    # all scene_ids are mapped on current timestep + objid
                    if n_iter == 0:
                        scene_id_list.append(
                            (scenario.dynamic_obstacles[i]._obstacle_id, scene_id)
                        )
                        scene_id += 1

            # Initialize lists
            # data.x:
            # x_p_l_list:       position lateral
            # x_p_t_list:       position tangential
            # x_angle_list:     position angle
            # x_class_list:     integer representing car, bus etc.

            # data.y:
            # y_p_l_list:       position lateral
            # y_p_t_list:       position tangential
            # y_angle_list:     position angle

            # ref_list:         9 ref entries [...]

            (
                x_p_l_list,
                x_p_t_list,
                x_angle_list,
                x_class_list,
                y_p_l_list,
                y_p_t_list,
                y_angle_list,
                ref_list,
                sc_img_list,
                scenario_id_list,
            ) = ([] for _ in range(10))

            if len(scene_t_objid) == 0:
                break

            if n_iter == 0:
                id_dict = dict(scene_id_list)

            # iterate over objects
            for objid in scene_t_objid:
                # obtain history and future data of the object within the respective time intervall.
                # Hereby, t_c / current_time_step is viewed as the last received data

                # current global position of object at current_time_step
                current_global_x = objid.prediction.trajectory.state_list[
                    current_time_step
                ].position[0]
                current_global_y = objid.prediction.trajectory.state_list[
                    current_time_step
                ].position[1]
                current_global_angle = objid.prediction.trajectory.state_list[
                    current_time_step
                ].orientation

                sc_img = generate_self_rendered_sc_img(
                    curr_pos=objid.prediction.trajectory.state_list[
                        current_time_step
                    ].position,
                    curr_orient=objid.prediction.trajectory.state_list[
                        current_time_step
                    ].orientation,
                    scenario=scenario,
                    map=None,
                    get_borders=False,
                    watch_radius=watch_radius,
                    res=256,
                    light_lane_dividers=True,
                )

                # Initialize dummy tensors of shape [1, :]
                (
                    x_p_l_dummy,
                    x_p_t_dummy,
                    x_angle_dummy,
                ) = ([0 for _ in range(pp)] for _ in range(3))
                (
                    y_p_l_dummy,
                    y_p_t_dummy,
                    y_angle_dummy,
                ) = ([0 for _ in range(fp)] for _ in range(3))

                # iterate over past timesteps
                for p in reversed(range(0, len_t_hist)):
                    # get gloabal position of object at timestep = current_time_step-p
                    global_x_p_l = objid.prediction.trajectory.state_list[
                        current_time_step - p
                    ].position[0]
                    global_x_p_t = objid.prediction.trajectory.state_list[
                        current_time_step - p
                    ].position[1]
                    global_x_angle = objid.prediction.trajectory.state_list[
                        current_time_step - p
                    ].orientation

                    # transform global cs into relative cs (cs of object at current_time_step)
                    (
                        (x_p_l, x_p_t, x_angle),
                        (_, _),
                        (_, _),
                    ) = coord_to_coord(
                        new_coord=(
                            torch.tensor(current_global_x, dtype=torch.float64),
                            torch.tensor(current_global_y, dtype=torch.float64),
                            torch.tensor(current_global_angle, dtype=torch.float64),
                        ),
                        old_coord=(
                            torch.tensor(global_x_p_l, dtype=torch.float64),
                            torch.tensor(global_x_p_t, dtype=torch.float64),
                            torch.tensor(global_x_angle, dtype=torch.float64),
                        ),
                        agent_p=(
                            torch.tensor(0, dtype=torch.float64),
                            torch.tensor(0, dtype=torch.float64),
                            torch.tensor(0, dtype=torch.float64),
                        ),
                        agent_v=(
                            torch.tensor(0, dtype=torch.float64),
                            torch.tensor(0, dtype=torch.float64),
                        ),
                        agent_a=(
                            torch.tensor(0, dtype=torch.float64),
                            torch.tensor(0, dtype=torch.float64),
                        ),
                    )

                    x_p_l_dummy[pp - 1 - p] = x_p_l
                    x_p_t_dummy[pp - 1 - p] = x_p_t
                    x_angle_dummy[pp - 1 - p] = x_angle

                # iterate over future timesteps
                for p in range(1, fp + 1):
                    # get gloabal position of object at timestep p
                    global_y_p_l = objid.prediction.trajectory.state_list[
                        current_time_step + p
                    ].position[0]
                    global_y_p_t = objid.prediction.trajectory.state_list[
                        current_time_step + p
                    ].position[1]
                    global_y_angle = objid.prediction.trajectory.state_list[
                        current_time_step + p
                    ].orientation

                    # transform global position into relative position (position in cs of object at current_time_step)
                    (y_p_l, y_p_t, y_angle) = coord_to_coord(
                        new_coord=(
                            torch.tensor(current_global_x, dtype=torch.float64),
                            torch.tensor(current_global_y, dtype=torch.float64),
                            torch.tensor(current_global_angle, dtype=torch.float64),
                        ),
                        old_coord=(
                            torch.tensor(global_y_p_l, dtype=torch.float64),
                            torch.tensor(global_y_p_t, dtype=torch.float64),
                            torch.tensor(global_y_angle, dtype=torch.float64),
                        ),
                        agent_p=(
                            torch.tensor(0, dtype=torch.float64),
                            torch.tensor(0, dtype=torch.float64),
                            torch.tensor(0, dtype=torch.float64),
                        ),
                    )

                    y_p_l_dummy[p - 1] = y_p_l
                    y_p_t_dummy[p - 1] = y_p_t
                    y_angle_dummy[p - 1] = y_angle

                # add to lists
                x_p_l_list.append(torch.tensor(x_p_l_dummy, dtype=torch.float))
                x_p_t_list.append(torch.tensor(x_p_t_dummy, dtype=torch.float))
                x_angle_list.append(torch.tensor(x_angle_dummy, dtype=torch.float))
                x_class_list.append(torch.tensor([0], dtype=torch.float))
                y_p_l_list.append(torch.tensor(y_p_l_dummy, dtype=torch.float))
                y_p_t_list.append(torch.tensor(y_p_t_dummy, dtype=torch.float))
                y_angle_list.append(torch.tensor(y_angle_dummy, dtype=torch.float))

                sc_img_list.append(torch.tensor(sc_img, dtype=torch.float))

                # get data for ref_list
                ref_list.append(
                    torch.tensor(
                        [
                            1,
                            id_dict[objid._obstacle_id],
                            current_time_step,
                            objid._obstacle_id,
                            objid.obstacle_shape.width,
                            objid.obstacle_shape.length,
                            current_global_x,
                            current_global_y,
                            current_global_angle,
                        ],
                        dtype=torch.float64,
                    )
                )
                scenario_id_list.append(
                    str(scenario.scenario_id)
                    + "_{:08d}".format(id_dict[objid._obstacle_id])
                )

            # convert node feature lists to tensors
            x_p_l_tensor = torch.stack(x_p_l_list).reshape(
                len(x_p_l_list), 1, len(x_p_l_list[0])
            )
            x_p_t_tensor = torch.stack(x_p_t_list).reshape(
                len(x_p_t_list), 1, len(x_p_t_list[0])
            )
            x_angle_tensor = torch.stack(x_angle_list).reshape(
                len(x_angle_list), 1, len(x_angle_list[0])
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
            sc_img_tensor = torch.stack(sc_img_list).reshape(
                len(sc_img_list),
                1,
                len(sc_img_list[0][0]),
                len(sc_img_list[0][1]),
            )

            # concatenate time series data
            x_time_series = torch.cat(
                (
                    x_p_l_tensor,
                    x_p_t_tensor,
                    x_angle_tensor,
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
                edge_index=get_edge_index(len(scene_t_objid)).t().contiguous(),
                sc_img=sc_img_tensor,
                scenario_id_list=scenario_id_list,
            )

            scenario_data_list.append(data_)

    # filter None values
    scenario_data_list = list(filter(None, scenario_data_list))

    if len(scenario_data_list) == 0:
        print("{} too short".format(file_name))

    return scenario_data_list


def create_split_files(processed_file_folder, train_ratio=0.8, val_ratio=0.1):
    scenario_names_list = []
    scenario_directory = os.path.join(
        processed_file_folder, "raw", "commonroad-scenarios"
    )  # change data folder here
    if not os.path.exists(scenario_directory):
        raise ValueError(
            "scenario directory does not exist {}".format(scenario_directory)
        )
    for path, _, files in os.walk(scenario_directory):
        for name in files:
            scenario_names_list.append(os.path.join(path, name))

    random.seed(42)
    random.shuffle(scenario_names_list)

    train_scenarios_file = os.path.join(
        processed_file_folder,
        "raw",
        "split_definition",
        "split_definition" + "_train.txt",
    )
    val_scenarios_file = os.path.join(
        processed_file_folder,
        "raw",
        "split_definition",
        "split_definition" + "_val.txt",
    )
    test_scenarios_file = os.path.join(
        processed_file_folder,
        "raw",
        "split_definition",
        "split_definition" + "_test.txt",
    )

    if not os.path.exists(train_scenarios_file):
        if not os.path.exists(os.path.dirname(train_scenarios_file)):
            os.mkdir(os.path.dirname(train_scenarios_file))

        split_1 = int(train_ratio * len(scenario_names_list))
        split_2 = int((train_ratio + val_ratio) * len(scenario_names_list))
        train_filenames = scenario_names_list[:split_1]
        val_filenames = scenario_names_list[split_1:split_2]
        test_filenames = scenario_names_list[split_2:]

        # Open the file in write mode
        with open(train_scenarios_file, "w") as file:
            # Iterate over the strings and write them line by line
            for scenario in train_filenames:
                file.write(scenario + "\n")

        with open(val_scenarios_file, "w") as file:
            # Iterate over the strings and write them line by line
            for scenario in val_filenames:
                file.write(scenario + "\n")

        with open(test_scenarios_file, "w") as file:
            # Iterate over the strings and write them line by line
            for scenario in test_filenames:
                file.write(scenario + "\n")

    return train_scenarios_file, val_scenarios_file, test_scenarios_file


def get_scenario(
    xml_file,
    scenario_path,
    with_traj_list=False,
):
    filename = os.path.join(scenario_path, xml_file)
    try:
        scenario, _ = CommonRoadFileReader(filename).open()
    except:
        print("Scenario does not work: {}".format(filename))
        scenario = None
    if with_traj_list:
        if scenario is None:
            return None, None, None
        else:
            try:
                trajectories_list = get_future_states(scenario)
                timesteps_in_scenario = max([len(tj) for tj in trajectories_list])
            except:
                return None, None, None

        return scenario, trajectories_list, timesteps_in_scenario
    return scenario


def get_future_states(scenario):
    return [
        [
            (
                scenario.dynamic_obstacles[i]
                .prediction.trajectory.state_list[j]
                .position,
                scenario.dynamic_obstacles[i]
                .prediction.trajectory.state_list[j]
                .orientation,
            )
            for j in range(
                0,
                len(scenario.dynamic_obstacles[i].prediction.trajectory.state_list),
            )
        ]
        for i in range(0, len(scenario.dynamic_obstacles))
        if "_trajectory" in scenario.dynamic_obstacles[i].prediction.__dict__.keys()
    ]


def count_object_per_scenario(split_type_list, scenario_path):
    strstr = ""
    for split_str in split_type_list:
        split_definition_file = os.path.join(
            os.path.dirname(scenario_path),
            "split_definition",
            "split_definition_{}.txt".format(split_str),
        )
        with open(split_definition_file, "r") as tf:
            scenario_list = [line.rstrip() for line in tf]

        strstr += "############ {} ############\n".format(split_str.upper())

        n_obj_list = []
        n_scenarios = 0

        for xml_item in tqdm(scenario_list):
            xml_file = os.path.basename(xml_item)
            scenario = get_scenario(
                xml_file, scenario_path=scenario_path, with_traj_list=False
            )

            if scenario is None:
                continue

            scenario_data_list = process_scenario(
                file_name=os.path.join(scenario_path, xml_file),
                pp=30,
                fp=30,
                watch_radius=64,
                min_len_t_hist=30,
                sliwi_size=1,
            )
            if not scenario_data_list:
                continue

            n_obj_list += [sc.x.shape[0] for sc in scenario_data_list]
            n_scenarios += len(scenario_data_list)

        iter_vals = np.arange(0.0, 1.0, 0.05)
        for qt in iter_vals:
            strstr += "q{:.02f}: {:.02f} \n".format(qt, np.quantile(n_obj_list, qt))
        strstr += "mean: {:.02f}, median: {:.02f}, num_sc: {:d}\n\n\n".format(
            np.mean(n_obj_list), np.median(n_obj_list), n_scenarios
        )

    with open(os.path.join(repo_path, "data_stats.txt"), "w") as f:
        f.write(strstr)
    print(strstr)


if __name__ == "__main__":
    from torch_geometric.loader import DataLoader
    from utils.scheduling import print_datastats

    debug = False
    split_type_list = ["train", "val", "test"]
    # count_object_per_scenario(split_type_list=split_type_list, scenario_path="data/raw/commonroad-scenarios")

    for split_type in split_type_list:
        print("\nCreating Dataloader = {} ...".format(split_type))
        time_test_dataloader_start = datetime.datetime.now()

        dataset = Dataset_CR(split_type=split_type, debug=debug)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

        print(
            "Completed after (Hours:Minutes:Seconds:Microseconds): "
            + str(datetime.datetime.now() - time_test_dataloader_start)
        )

        print_datastats(dataloader, split_type)
