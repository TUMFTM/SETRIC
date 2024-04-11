import os
import sys
import pickle
import xml.etree.ElementTree as ET
from tqdm import tqdm
import argparse

repo_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(repo_path)

import torch
import numpy as np
from shapely.wkt import loads
import geopandas as gpd

from utils.geometry import abs_to_rel_coord
from utils.scheduling import permute_input


def generate_self_rendered_sc_img(
    curr_pos,
    curr_orient,
    scenario=None,
    map=None,
    get_borders=False,
    watch_radius=64,
    res=256,
    light_lane_dividers=True,
):
    """Render scene image in relative position."""
    # region generate_self_rendered_sc_img()
    # region read inputs
    pixel_dist = 2 * watch_radius / res
    interp_factor = 0.8
    # endregion
    # timer = time.time()

    # region read all lanelet boundarys into a list
    bound_list = []
    type_list = []
    if map is None:  # CommonRoad
        lane_net = scenario.lanelet_network
        for lanelet in lane_net.lanelets:
            bound_list.append(lanelet.left_vertices)
            line_type = "road-boundary" if lanelet.adj_left is None else "lane-marking"
            type_list.append(line_type)
            bound_list.append(lanelet.right_vertices)
            line_type = "road-boundary" if lanelet.adj_right is None else "lane-marking"
            type_list.append(line_type)
    else:  # OpenDD
        if get_borders:
            iter_list = map.borderlines
        else:
            iter_list = map.trafficlanes.values()
        for trafficlane in iter_list:
            bound_list.append(trafficlane["geometry"])
            line_type = "road-boundary"
            type_list.append(line_type)
    # endregion

    # region translate rotate image
    bound_list = [
        abs_to_rel_coord(curr_pos, curr_orient, bound_line) for bound_line in bound_list
    ]

    # endregion
    # print(f"Time for reading points:{time.time() - timer}")
    # timer = time.time()

    # region limit_boundarys to watch_radius
    # region limit_boundary_subfunction
    def limit_boundary(boundary):
        array = np.empty(len(boundary))
        last_point_was_out = None  # This line makes the linter happy
        # Loop over all points
        for index, point in enumerate(boundary):
            # Check index to avoid indexerrors
            if index > 0:
                # check if point is outside of viewing window
                point_is_out = bool(max(abs(point)) > watch_radius)
                # If point is inside
                if point_is_out is False:
                    array[index] = False
                    # Add the neighbour, so that a continous line
                    # to the image border can be rendered
                    if last_point_was_out is True:
                        array[index - 1] = False
                # if point is outside of watch_radius
                else:
                    # Add this point as neighbor if the last point was in
                    if last_point_was_out is False:
                        array[index] = False
                    # Remove point from boundary line
                    else:
                        array[index] = True
            else:
                # Handling of first element
                point_is_out = bool(max(abs(point)) > watch_radius)
                array[index] = point_is_out
            last_point_was_out = point_is_out
        return array

    # endregion
    # Call the function
    limit_bound_list = [
        np.delete(bound, limit_boundary(bound).astype(bool), axis=0)
        for bound in bound_list
    ]
    # endregion

    # timer = time.time()
    # region Interpolate boundary lines

    # region interpolate_boundary() subfunction
    def interpolate_boundary(boundary):
        # region calc curve length of boundary
        curve_length = np.zeros(len(boundary))
        bound_array = np.array(boundary)
        for index, point in enumerate(bound_array[1:], start=1):
            curve_length[index] = curve_length[index - 1] + np.linalg.norm(
                point - boundary[index - 1]
            )
        # endregion
        # region interpolate over curve_length
        if len(curve_length) > 0:
            eval_array = np.arange(0, curve_length[-1], pixel_dist * interp_factor)
            return np.array(
                [
                    np.interp(eval_array, curve_length, bound_array.transpose()[0]),
                    np.interp(eval_array, curve_length, bound_array.transpose()[1]),
                ]
            )
        # if no point is left return None
        return None
        # endregion

    # endregion

    # region call subfunction and add concat pixel values
    interp_bound_list = []
    for bound_line, line_type in zip(limit_bound_list, type_list):
        interp_line = interpolate_boundary(bound_line)
        if interp_line is not None:
            if line_type == "road-boundary":
                value = 255
            elif line_type == "lane-marking":
                value = 127
            value_vec = np.ones((1, interp_line.shape[1])) * value
            interp_bound_list.append(np.concatenate([interp_line, value_vec], axis=0))
        else:
            continue
    # endregion
    # endregion
    # print(f"Time for creating interpolation points:{time.time() - timer}")
    # timer = time.time()

    # region create image indexes
    interp_bound_arr = np.concatenate(interp_bound_list, axis=1)
    pixel_indexes = np.concatenate(
        [
            interp_bound_arr[0:2] // pixel_dist + res / 2,
            interp_bound_arr[2].reshape(1, interp_bound_arr.shape[1]),
        ],
        axis=0,
    )

    # endregion

    # region limit index indices to resolution
    pixel_indexes = np.delete(
        pixel_indexes,
        np.logical_or(
            np.amax(pixel_indexes[0:2], axis=0) > res - 1,
            np.amin(pixel_indexes[0:2], axis=0) < 0,
        ),
        axis=1,
    )

    # endregion

    # print(f"Time for creating index-set:{time.time() - timer}")
    # timer = time.time()

    # region build full-size image
    # create empty black image
    img = 0 * np.ones((res, res))
    pixel_values = pixel_indexes[2] if light_lane_dividers else 0
    # add values to image
    img[pixel_indexes[1].astype(int), pixel_indexes[0].astype(int)] = pixel_values
    # endregion
    # print(f"Time for building image:{time.time() - timer}")

    # saving the full size image needs less space than the pixel_index_data
    # there must be any kind of optimisation for saving pickling large tensors
    # in the background
    # pylint: disable=not-callable
    return img
    # pylint: enable=not-callable
    # endregion


def get_xy_geometry_from_linestring(linestring):
    return np.array(loads(linestring.text).coords.xy).T


def get_successors(traffic_lane_element):
    if len(traffic_lane_element) < 4:
        return []
    return [sucessor_lane.text for sucessor_lane in traffic_lane_element[3]]


def get_color(borderline_str):
    if borderline_str == "GRAY":
        return "0.5"
    if borderline_str == "WHITE":
        return "1"


def get_rdp_map_by_shp(rdb, data_path):
    rdb_str = "rdb" + str(rdb)

    # Read the trafficlanes
    shp_tl = load_shp(rdb_str=rdb_str, data_path=data_path, type_str="trafficlanes")
    trafficlanes = {
        str(shp_tl.identifier[tl_idx]): {
            "geometry": np.array(shp_tl.geometry[tl_idx].coords.xy).T,
            "successors": (
                shp_tl.successors[tl_idx].split(",")
                if isinstance(shp_tl.successors[tl_idx], str)
                else []
            ),
            "type": shp_tl.type[tl_idx],
        }
        for tl_idx in shp_tl.index
    }

    # Read the boderlines
    shp_bl = load_shp(rdb_str=rdb_str, data_path=data_path, type_str="borderlines")
    borderlines = [
        {
            "geometry": np.array(shp_bl.geometry[bl_idx].coords.xy).T,
            "markingType": shp_bl.markType[bl_idx],
            "colorType": get_color(shp_bl.colorType[bl_idx]),
            "materialType": shp_bl.matType[bl_idx],
        }
        for bl_idx in shp_bl.index
        if bool(shp_bl.geometry[bl_idx])
    ]
    shp_ar = load_shp(rdb_str=rdb_str, data_path=data_path, type_str="drivableareas")
    areas = [{} for _ in shp_ar.index]

    return trafficlanes, borderlines, areas


def load_shp(rdb_str, data_path, type_str):
    # type_str: traffilanes, borderlins, drivableareas
    shp_path_tl = os.path.join(
        data_path,
        "raw",
        rdb_str,
        "map_" + rdb_str,
        "shapefiles_" + type_str,
        "map_" + rdb_str + "_UTM32N_" + type_str + ".shp",
    )
    return gpd.read_file(shp_path_tl)


def get_rdb_map_by_xml(rdb: int, data_path=os.path.join(repo_path, "data")):
    rdb_tags_def = ["trafficLanes", "borderLines", "areas"]
    rdb_str = "rdb" + str(rdb)
    map_path = os.path.join(
        data_path, "raw", rdb_str, "map_" + rdb_str, "map_" + rdb_str + "_UTM32N.xml"
    )
    root_map = ET.parse(map_path).getroot()
    rdb_tags = [rm.tag for rm in root_map]

    assert min([rr in rdb_tags for rr in rdb_tags_def])

    idx = rdb_tags.index("trafficLanes")
    trafficlanes = {
        traffic_lane_element[0].text: {
            "geometry": get_xy_geometry_from_linestring(traffic_lane_element[1]),
            "successors": get_successors(traffic_lane_element),
            "type": traffic_lane_element[2].text,
        }
        for traffic_lane_element in root_map[idx]
    }

    idx = rdb_tags.index("borderLines")
    borderlines = [
        {
            "geometry": get_xy_geometry_from_linestring(border_line_element[0]),
            "markingType": border_line_element[1].text,
            "colorType": get_color(border_line_element[2].text),
            "materialType": border_line_element[3].text,
        }
        for border_line_element in root_map[idx]
        if "geometry" in [bll.tag for bll in border_line_element]
    ]

    idx = rdb_tags.index("areas")
    areas = [{} for area in root_map[idx]]

    return trafficlanes, borderlines, areas


def get_rdp_map(rdb: int, data_path=os.path.join(repo_path, "data")):
    if rdb in [4, 6]:
        trafficlanes, borderlines, areas = get_rdp_map_by_shp(rdb, data_path)
    else:
        trafficlanes, borderlines, areas = get_rdb_map_by_xml(rdb, data_path)

    return MapClass(
        rdb_num=rdb,
        trafficlanes=trafficlanes,
        borderlines=borderlines,
        areas=areas,
    )


class MapClass:
    def __init__(self, rdb_num, trafficlanes, borderlines, areas):
        self.rdb_num = rdb_num
        self.trafficlanes = trafficlanes
        self.borderlines = borderlines
        self.areas = areas


def plot_map(map_cl):
    trafficlanes = map_cl.trafficlanes
    borderlines = map_cl.borderlines
    _, ax = plt.subplots(1, 1)
    ax.set_facecolor("0.8")
    for tl in trafficlanes.values():
        ax.plot(tl["geometry"][:, 0], tl["geometry"][:, 1], "k")
    for bl in borderlines:
        ax.plot(bl["geometry"][:, 0], bl["geometry"][:, 1], color=bl["colorType"])
    plt.show()


def load_sc_img_pkl(sc_img_file_path):
    with open(sc_img_file_path, "rb") as f:
        sc_img = pickle.load(f)
    return sc_img


def get_sc_img_batch(batch_obj_ref, rdp_map_dict, data_path, mod, cfg, res=256):
    sc_img = torch.zeros([batch_obj_ref.shape[0], 1, res, res], dtype=torch.float32)
    sc_img_path = os.path.join(data_path, "sc_img")

    postfix = get_postfix(cfg)

    for obj_idx in range(batch_obj_ref.shape[0]):
        sc_img_name = "rdb_{}_sc_{}_id_{}_{}.pkl".format(
            int(batch_obj_ref[obj_idx][0][0]),
            int(batch_obj_ref[obj_idx][0][1]),
            int(batch_obj_ref[obj_idx][0][3]),
            mod + postfix,
        )
        sc_img_file_path = os.path.join(sc_img_path, sc_img_name)
        if False and os.path.exists(sc_img_file_path):
            sc_img[obj_idx] = torch.FloatTensor(
                load_sc_img_pkl(sc_img_file_path)
            ).squeeze(0)
        else:
            rdb_num = int(batch_obj_ref[obj_idx][0][0])
            coord_x_utm = batch_obj_ref[obj_idx, 0, 6]
            coord_y_utm = batch_obj_ref[obj_idx, 0, 7]
            coord_angle_utm = batch_obj_ref[obj_idx, 0, 8]
            sc_img[obj_idx] = torch.FloatTensor(
                generate_self_rendered_sc_img(
                    curr_pos=np.array([coord_x_utm, coord_y_utm]),
                    curr_orient=coord_angle_utm,
                    scenario=None,
                    map=rdp_map_dict[rdb_num],
                    get_borders=cfg["get_borders"],
                    watch_radius=64,
                    res=res,
                    light_lane_dividers=True,
                )
            ).unsqueeze(0)
    return sc_img


def get_postfix(cfg):
    if cfg["get_borders"]:
        postfix = "_bord"
    else:
        postfix = ""
    return postfix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--get_borders", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()

    from torch_geometric.loader import DataLoader
    from utils.Dataset_OpenDD import Dataset_OpenDD
    import matplotlib.pyplot as plt
    import matplotlib.image

    args.debug = True

    train_dataset = Dataset_OpenDD(split="r_1", split_type="train", debug=args.debug)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    val_dataset = Dataset_OpenDD(split="r_1", split_type="val", debug=args.debug)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    iter_list = [(train_dataloader, "train"), (val_dataloader, "val")]

    data_path = train_dataset.processed_file_folder
    sc_img_path = os.path.join(data_path, "sc_img")
    if not os.path.exists(sc_img_path):
        os.makedirs(sc_img_path)

    valid_rdbs = [1, 2, 3, 4, 5, 6, 7]
    rdp_map_dict = {
        rdb_int: get_rdp_map(rdb_int, data_path=data_path) for rdb_int in valid_rdbs
    }

    cfg_train = {"get_borders": args.get_bordersl}

    for data_loader, mod in iter_list:
        for batch in tqdm(data_loader):
            permute_input(data_batch=data_loader)

            postfix = get_postfix(cfg_train)

            for obj_idx in range(batch.x.shape[0]):
                rdb_num = int(batch.obj_ref[obj_idx][0][0])
                coord_x_utm = batch.obj_ref[obj_idx, 0, 6]
                coord_y_utm = batch.obj_ref[obj_idx, 0, 7]
                coord_angle_utm = batch.obj_ref[obj_idx, 0, 8]

                sc_img_name = "rdb_{}_sc_{}_id_{}_{}.png".format(
                    rdb_num,
                    int(batch.obj_ref[obj_idx][0][1]),
                    int(batch.obj_ref[obj_idx][0][3]),
                    mod + postfix,
                )
                save_path = os.path.join(sc_img_path, sc_img_name)

                if os.path.exists(save_path):
                    continue

                sc_img = generate_self_rendered_sc_img(
                    curr_pos=np.array([coord_x_utm, coord_y_utm]),
                    curr_orient=coord_angle_utm,
                    scenario=None,
                    map=rdp_map_dict[rdb_num],
                    watch_radius=64,
                    res=256,
                    light_lane_dividers=True,
                )

                # save array to pkl
                with open(save_path.replace(".png", ".pkl"), "wb") as f:
                    pickle.dump(sc_img, f)

                # save array to png
                matplotlib.image.imsave(os.path.join(sc_img_path, sc_img_name), sc_img)
