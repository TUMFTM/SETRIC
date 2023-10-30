import os
import sys
import argparse
import json
import datetime
from importlib.metadata import version

repo_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(repo_path)

import cv2
import imageio.v2 as imageio

import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm


from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.scenario.obstacle import DynamicObstacle

if int(version("commonroad-io").split(".")[0]) < 2023:
    raise ValueError("Stale Commonroad-version, update to at least to 2023.2")

from models.Fusion_Model import Fusion_Model
from utils.Dataset_CR import process_scenario, get_scenario

TUM_BLUE = (0 / 255, 101 / 255, 189 / 255)
TUM_ORAN = (227 / 255, 114 / 255, 34 / 255)
TUM_GREEN = (162 / 255, 173 / 255, 0 / 255)
TUM_BLACK = (0, 0, 0)
GRAY = (104 / 255, 104 / 255, 104 / 255)
BOUND_COL = (204 / 255, 204 / 255, 204 / 255)

SCENARIO_PATH = "data"

SC_RENDER_PATH = os.path.join(
    os.path.dirname(SCENARIO_PATH), "commonroad-scenarios-rendered"
)

if not os.path.exists(SC_RENDER_PATH):
    os.mkdir(SC_RENDER_PATH)


def transform_back(pred_traj, pos, orien):
    """Transform back from local to global."""
    rotation = -orien
    translation = -pos
    rot_mat = np.array(
        [[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]]
    )
    pred_traj[:, :2] = np.matmul(pred_traj[:, :2], rot_mat)
    pred_traj[:, :2] = pred_traj[:, :2] - translation

    return pred_traj[:, :2]


def get_selection_sample(batch, pred, idx_list, pred1, pred2, pred3):
    default_tuple = None, 0, ""
    if batch.y.shape[1] < pred.shape[1]:
        return default_tuple

    # error preds to gt
    err = torch.pow(pred - batch.y[idx_list, :, :2], 2.0).sum(dim=2).sum(dim=1)
    err1 = torch.pow(
        torch.pow(pred1 - batch.y[:, :, :2], 2.0).sum(dim=2).sum(dim=1) / pred.shape[1],
        0.5,
    )
    err2 = torch.pow(
        torch.pow(pred2 - batch.y[:, :, :2], 2.0).sum(dim=2).sum(dim=1) / pred.shape[1],
        0.5,
    )
    err3 = torch.pow(
        torch.pow(pred3 - batch.y[:, :, :2], 2.0).sum(dim=2).sum(dim=1) / pred.shape[1],
        0.5,
    )
    err_stack = torch.stack([err1, err2, err3])

    min_err_val, _ = err_stack.min(dim=0)
    max_err_val, _ = err_stack.max(dim=0)

    cor_selection = min_err_val[idx_list] == err

    if cor_selection.shape[0] == 0 or not max(cor_selection):
        return default_tuple

    # biggest deviation of prediction
    max_dev_idx = int((max_err_val - min_err_val).argmax())
    max_dev = (max_err_val - min_err_val).max()

    if min_err_val[max_dev_idx] < 0.02:
        return default_tuple

    # deviation too small
    if max_dev < 1.0:
        return default_tuple

    # correct selection
    if max_dev_idx not in idx_list:
        return default_tuple

    if cor_selection[max_dev_idx]:
        strstr = "_right_sel"
    else:
        strstr = "_wrong_sel"

    # low error to gt of correct selection
    rmse_obj = torch.pow(err[idx_list.index(max_dev_idx)] / pred.shape[1], 0.5)

    if rmse_obj < 1.0:
        return True, max_dev_idx, str(int(max_dev)).zfill(4) + "_rmse_m" + strstr

    return default_tuple


def render_full_scenario(scenario, args, model=None):
    scenario, trajectories_list, timesteps_in_scenario = get_scenario(
        xml_file, scenario_path=SCENARIO_PATH, with_traj_list=True
    )
    if scenario is None:
        return None

    pp = 30
    fp = 50
    lw = 0.3

    scenario_data_list = process_scenario(
        file_name=os.path.join(SCENARIO_PATH, xml_file),
        pp=pp,
        fp=fp,
        watch_radius=64,
        min_len_t_hist=30,
        sliwi_size=1,
    )
    if not scenario_data_list:
        return None

    full_scene_path = os.path.join(SC_RENDER_PATH, xml_file.replace(".xml", ""))
    if args.all_pred:
        full_scene_path += "_all_pred"
    if not os.path.exists(full_scene_path):
        os.mkdir(full_scene_path)

    selection_sample = None
    iter_scen_d = iter(scenario_data_list)

    for tj in range(pp - 1, timesteps_in_scenario - fp):
        rnd = MPRenderer()
        scenario.lanelet_network.draw(rnd)

        # fig, ax = plt.subplots(1, 1)
        # bound_list = []
        # type_list = []
        # lane_net = scenario.lanelet_network
        # for lanelet in lane_net.lanelets:
        # bound_list.append(lanelet.left_vertices)
        # line_type = "road-boundary" if lanelet.adj_left is None else "lane-marking"
        # type_list.append(line_type)
        # bound_list.append(lanelet.right_vertices)
        # line_type = "road-boundary" if lanelet.adj_right is None else "lane-marking"
        # type_list.append(line_type)
        # plt.plot(lanelet.right_vertices[:, 0], lanelet.right_vertices[:, 1])
        # plt.plot(lanelet.left_vertices[:, 0], lanelet.left_vertices[:, 1])

        if model is not None:
            # get net input
            batch = next(iter_scen_d)
            batch.x = batch.x.permute(0, 2, 1)
            batch.y = batch.y.permute(0, 2, 1)
            # Add object class to the time series data
            obj_class = batch.obj_class.repeat(
                1, batch.x.shape[1], 1
            )  # obj_class shape: (N, seq_length, 1)
            batch.x = torch.cat((batch.x, obj_class), dim=2)
            batch = batch.to(device)

            with torch.no_grad():
                pred, idx_list = model(batch)

            if args.all_pred:
                sc_img_enc = model.fn_sc_img_encoder(batch=batch)
                pred1 = model.pred_cv(batch, sc_img_enc)
                pred2 = model.pred_l_lstm(batch, sc_img_enc)
                pred3 = model.pred_dg_lstm(batch, sc_img_enc)

                selection_sample, max_dev_idx, add_str = get_selection_sample(
                    batch, pred, idx_list, pred1, pred2, pred3
                )

                pred1 = pred1.detach().cpu().numpy()
                pred2 = pred2.detach().cpu().numpy()
                pred3 = pred3.detach().cpu().numpy()

                batch.y = batch.y.detach().cpu().numpy()
            else:
                selection_sample = None

            pred = pred.detach().cpu().numpy()

        for n_ob, dyn_ob in enumerate(scenario.obstacles):
            # remove prediction
            if tj == 0:
                dyn_ob.prediction = None

            # continue if object not in scene anymore
            if tj >= len(trajectories_list[n_ob]):
                continue

            # update state of object
            pos, orien = trajectories_list[n_ob][tj]
            dyn_ob.initial_state.position = pos
            dyn_ob.initial_state.orientation = orien
            plt_gca = plt.gca()
            plt_gca.plot(pos[0], pos[1], "X", markersize=20)

            # create updated object
            obj_temp = DynamicObstacle(
                obstacle_id=dyn_ob.obstacle_id,
                obstacle_type=dyn_ob.obstacle_type,
                obstacle_shape=dyn_ob.obstacle_shape,
                initial_state=dyn_ob.initial_state,
                prediction=dyn_ob.prediction,
                initial_center_lanelet_ids=dyn_ob.initial_center_lanelet_ids,
                initial_shape_lanelet_ids=dyn_ob.initial_shape_lanelet_ids,
                initial_signal_state=dyn_ob.initial_signal_state,
                signal_series=dyn_ob.signal_series,
                initial_meta_information_state=dyn_ob.initial_meta_information_state,
                meta_information_series=dyn_ob.meta_information_series,
                external_dataset_id=dyn_ob.external_dataset_id,
            )

            if n_ob in idx_list:
                pred_idx = idx_list.index(n_ob)
                _ = transform_back(pred[pred_idx], pos, orien)

                if args.all_pred:
                    _ = transform_back(pred1[n_ob], pos, orien)
                    _ = transform_back(pred2[n_ob], pos, orien)
                    _ = transform_back(pred3[n_ob], pos, orien)
                    _ = transform_back(batch.y[n_ob, :, :2], pos, orien)

            obj_temp.draw(rnd)

        # render before prediction plots
        rnd.render()

        if args.get_pred_samples and selection_sample:
            lw = 1.0
            gt_ = batch.y[idx_list.index(max_dev_idx)][:, :2]
            add_patch_to_gca(gt_, col=TUM_BLUE, lw=lw)

            x_lim = (min(gt_[:, 0]) - 20, max(gt_[:, 0]) + 20)
            y_lim = (min(gt_[:, 1]) - 20, max(gt_[:, 1]) + 20)

            iter_obj = [idx_list.index(max_dev_idx)]
        else:
            iter_obj = range(pred.shape[0])

        # plot predictions
        for n_pred_ob in iter_obj:
            if args.all_pred:
                pred_input1 = pred1[idx_list[n_pred_ob]]
                add_patch_to_gca(pred_input1, col=TUM_ORAN, lw=lw)

                pred_input2 = pred2[idx_list[n_pred_ob]]
                add_patch_to_gca(pred_input2, col=TUM_GREEN, lw=lw)

                pred_input3 = pred3[idx_list[n_pred_ob]]
                add_patch_to_gca(pred_input3, col=GRAY, lw=lw)

            pred_input = pred[n_pred_ob]
            add_patch_to_gca(pred_input, col=TUM_BLACK, ls="dashed", lw=lw / 1.3)

        if args.get_pred_samples and selection_sample:
            plt_gca = plt.gca()
            plt_gca.set_xlim(x_lim)
            plt_gca.set_ylim(y_lim)

            if "wrong_sel" in add_str:
                dir = "wrong_sel"
            else:
                dir = "right_sel"
            if not os.path.exists(dir):
                os.mkdir(dir)

            # debug scenario gt
            # we = trajectories_list[max_dev_idx]
            # x_gt = [kk[0] for kk in np.array(we)[:, 0]]
            # y_gt = [kk[1] for kk in np.array(we)[:, 0]]
            # gt_input = np.array([x_gt, y_gt]).T
            # add_patch_to_gca(gt_input, col=TUM_GREEN, lw=lw)

            save_str = (
                dir
                + "/sample_"
                + os.path.basename(full_scene_path)
                + "_"
                + str(tj)
                + "_"
                + add_str
                + ".png"
            )
            plt.savefig(save_str, format="png")
            plt.savefig(save_str.replace(".png", ".svg"), format="svg")

        plt.savefig(os.path.join(full_scene_path, str(tj) + ".png"), format="png")
        plt.savefig(os.path.join(full_scene_path, str(tj) + ".svg"), format="svg")

    return full_scene_path


def add_patch_to_gca(pred_input, col="k", lw=0.3, ls="solid"):
    plt_gca = plt.gca()
    cc = [mpl.path.Path.LINETO for _ in range(pred_input.shape[0])]
    cc[0] = mpl.path.Path.MOVETO
    cc[-1] = mpl.path.Path.STOP

    path = mpl.path.Path(pred_input, codes=cc, closed=False)
    pp = mpl.patches.PathPatch(path, color=col, lw=lw, ls=ls, zorder=100, fill=False)
    plt_gca.add_patch(pp)


def create_video(full_scene_path, freq: float = 10.0, video_name: str = None):
    """Create .mp4-video from pngs.

    Args:
        freq (float, optional): Video frequency in Hz. Defaults to 10.0.
        video_name (str, optional): Name of video. Defaults to None.
    """
    if video_name is None:
        video_name = datetime.datetime.now().__format__("%Y-%m-%d-%H-%M-%S") + ".mp4"

    images = [img for img in os.listdir(full_scene_path) if img.endswith(".png")]

    images_sort = [i.zfill(8) + i for i in images]
    images_sort.sort()
    images = [i.split("g")[1] + "g" for i in images_sort]
    frame = cv2.imread(os.path.join(full_scene_path, images[1]))
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_name, fourcc, freq, (width, height))

    i = 0
    print("Creating video ..")
    for image in tqdm(images):
        i += 1
        img = cv2.imread(os.path.join(full_scene_path, image))
        hostname = os.uname()[1]
        if hostname != "gpu-vm":
            cv2.imshow("Creating video...", img)
            cv2.waitKey(1)
        video.write(img)

    cv2.destroyAllWindows()
    video.release()

    print("MP4 saved to {}".format(video_name))


def create_gif(
    full_scene_path,
    freq: float = 10.0,
    gif_duration_s: float = 5.0,
    video_name: str = None,
):
    """Create gif from mp4 video.

    Args:
        freq (float): Frequency of the video in Hz.
        gif_duration_s (int, optional): Duration of gif in seconds. Defaults to 5.
        video_name (_type_, optional): name of output gif, has to end with '.gif'. Defaults to None.

    Raises:
        ValueError: Raised if gif name has wrong ending.
    """
    if video_name is None:
        video_name = datetime.datetime.now().__format__("%Y-%m-%d-%H-%M-%S") + ".gif"
    elif not video_name.endswith(".gif"):
        raise ValueError(
            "Invalide gif name {}, has to end with '.gif'".format(video_name)
        )

    filenames = [img for img in os.listdir(full_scene_path) if img.endswith(".png")]
    _ = filenames.sort()

    max_images = int(gif_duration_s * freq)
    chunk_lists = [
        filenames[x : x + max_images] for x in range(0, len(filenames), max_images)
    ]

    kargs = {"duration": 1 / freq}
    for j, chunk_list in tqdm(enumerate(chunk_lists)):
        with imageio.get_writer(
            video_name.replace(".gif", "_{}.gif".format(j)), mode="I", **kargs
        ) as writer:
            for filename in chunk_list:
                image = imageio.imread(os.path.join(full_scene_path, filename))
                writer.append_data(image)

    print("GIF saved to {}".format(video_name))


def render_initial_scenario(xml_file):
    scenario = get_scenario(xml_file)
    if scenario is None:
        return
    rnd = MPRenderer()

    scenario.lanelet_network.draw(rnd)
    for dyn_ob in scenario.obstacles:
        dyn_ob.prediction = None
        dyn_ob.draw(rnd)

    rnd.render()

    plt.savefig(
        os.path.join(SC_RENDER_PATH, xml_file.replace(".xml", ".svg")), format="svg"
    )
    plt.savefig(
        os.path.join(SC_RENDER_PATH, xml_file.replace(".xml", ".pdf")), format="pdf"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="results/cr_fusion_08",
        help="path to stored training",
    )
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--all_pred", default=False, action="store_true")
    parser.add_argument("--get_pred_samples", default=False, action="store_true")
    args = parser.parse_args()

    for ts_ in ["train", "val", "test"]:
        split_definition_file = os.path.join(
            os.path.dirname(SCENARIO_PATH),
            "split_definition",
            "split_definition_{}.txt".format(ts_),
        )
        with open(split_definition_file, "r") as tf:
            scenario_list = [line.rstrip() for line in tf]

        if args.get_pred_samples:
            args.all_pred = True

        base_path = args.path

        with open(os.path.join(base_path, "train_config.json"), "r") as f:
            cfg_train = json.load(f)
        with open(os.path.join(base_path, "net_config.json"), "r") as f:
            net_config = json.load(f)

        cfg_train["device"] = "cuda"
        device = "cuda"

        model = Fusion_Model(cfg=cfg_train, net_config=net_config, is_inference=True)
        model.update_model_tag(n_iter=0, model_tag="g_sel")
        model_load_path = os.path.join(
            args.path,
            cfg_train["split"],
            model.tag,
            str(cfg_train["seed"]),
            "model_parameters_{}.pth.tar".format(cfg_train["seed"]),
        )
        checkpoint = torch.load(model_load_path, map_location=cfg_train["device"])
        model_stat_dict = checkpoint["model_state_dict"]
        model.load_state_dict(model_stat_dict)
        model.to(cfg_train["device"])

        scenario_list = scenario_list[155:]

        for xml_item in tqdm(scenario_list):
            xml_file = os.path.basename(xml_item)

            full_scene_path = render_full_scenario(
                xml_file,
                args=args,
                model=model,
            )
            if full_scene_path is None or args.get_pred_samples:
                continue
            create_gif(
                full_scene_path,
                video_name=os.path.join(
                    os.path.dirname(full_scene_path), xml_file.replace(".xml", ".gif")
                ),
            )
            create_video(
                full_scene_path,
                video_name=os.path.join(
                    os.path.dirname(full_scene_path), xml_file.replace(".xml", ".mp4")
                ),
            )
