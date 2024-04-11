import os
import sys
import copy
import datetime
from tqdm import tqdm

import torch
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorch_model_summary import summary

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

from utils.Dataset_OpenDD import Dataset_OpenDD
from utils.Dataset_CR import Dataset_CR
from utils.map_processing import get_rdp_map, get_sc_img_batch
from utils.scheduling import (
    print_overview,
    print_datastats,
    print_quantiles,
    permute_input,
)
from utils.metrics import get_mse_over_horizon, get_rmse_over_samples


def log_epoch(
    current_model,
    n_iter,
    iter_loss_sum,
    rmse_over_horizon,
    opt_rmse_over_horizon,
    rmse_over_samples,
    num_correct_selections,
    is_val,
    log_string,
    epoch_time_start,
):
    strstr = print_overview(
        current_model,
        n_iter,
        iter_loss_sum,
        rmse_over_horizon.mean(),
        opt_rmse_over_horizon.mean(),
        num_correct_selections,
        end_str="\n",
        string_only=True,
    )

    if is_val:
        strstr = "Validation"
    else:
        strstr = "Training"
    log_string(
        "{}, mean loss: {:.02f}, RMSE = {:.02f} m".format(
            strstr, iter_loss_sum, rmse_over_horizon.mean()
        )
    )
    print_quantiles(rmse_over_samples, log_string=log_string)

    log_string(
        "Epoch iteration time: " + str(datetime.datetime.now() - epoch_time_start)
    )


def create_log_path(log_root_path, run_tag, printer, model, device, cfg):
    """Check input validity"""
    if "open" in cfg["data"] and cfg["split"] not in ["r_1", "r_2", "r_3"]:
        raise ValueError(
            "invalid split - split can either be 'r_1', 'r_2', 'r_3', got: "
            + cfg["split"]
        )
    if not hasattr(model, "tag"):
        raise ValueError("invalid model - model has to have the attribute 'tag'")

    """ Save Model parameter inputs as dict """
    input_arg_dict = {}
    for key, val in model.__dict__.items():
        if key in ["model_fns", "best_model_params"] or key[0] == "_":
            continue
        input_arg_dict[key] = val

    seed = cfg.get("seed", 42)
    if run_tag is None:
        run_tag = str(seed)

    """Create paths"""
    if log_root_path is not None and run_tag is not None:
        log_folder_path = os.path.join(log_root_path, run_tag)
        log_file_path = os.path.join(log_folder_path, "train_" + run_tag + ".txt")
        tensorboard_path = os.path.join(log_root_path, "tensorboard", run_tag)
        model_save_path = os.path.join(
            log_folder_path, "model_parameters_" + run_tag + ".pth.tar"
        )

        """Create log file"""
        if os.path.exists(log_file_path):
            os.remove(log_file_path)
        if not os.path.exists(log_folder_path):
            os.makedirs(log_folder_path)
        log_fout = open(log_file_path, "w")

        def log_string(out_str, printer_=printer):
            # function for writing in the txt-logfile and print log
            log_fout.write(out_str + "\n")
            if printer_:
                print(out_str)

        save_results = True

    else:
        print(
            "'run_tag' or 'log_file_path' appears to be None - No log file, checkpoint or tensorboard entry is "
            "created!"
        )
        save_results = False

        def log_string(out_str, printer_=printer):
            # function to print log
            if printer_:
                print(out_str)

    """Write log header"""
    log_string(
        "\n\n"
        + "\tLog File - Single Training with Validation\t".center(80, "#")
        + "\n\n"
        + "Date:\t"
        + datetime.datetime.now().strftime("%Y-%m-%d")
        + "\n"
        + "Time:\t"
        + datetime.datetime.now().strftime("%H:%M:%S")
        + "\n"
        + "\n"
        + "model_tag:".ljust(20)
        + model.tag
        + "\n"
        + "device:".ljust(20)
        + device
        + "\n"
        + "seed:".ljust(20)
        + str(seed)
        + "\n"
    )

    for p in cfg:
        log_string((p + ":").ljust(20) + str(cfg[p]))

    for p in input_arg_dict:
        log_string((p + ":").ljust(20) + str(input_arg_dict[p]), printer_=False)

    return (
        log_fout,
        seed,
        run_tag,
        input_arg_dict,
        log_string,
        tensorboard_path,
        model_save_path,
        save_results,
    )


def get_data(cfg, log_string, debug=False, train_dataloader=None):
    """Create dataloader for training and validation data"""
    # choose dataset
    if "open" in cfg["data"]:
        dataset = Dataset_OpenDD
    else:
        dataset = Dataset_CR

    # check for debug
    if debug:
        is_shuffle = False
    else:
        is_shuffle = True

    processed_file_folder = None

    # Training Dataloader
    log_string("\n" + "-" * 100)
    if train_dataloader is None:
        log_string("Creating Training Dataloader (data = {})...".format(cfg["data"]))
        time_train_dataloader_start = datetime.datetime.now()
        train_dataset = dataset(
            split=cfg["split"],
            split_type="train",
            debug=debug,
        )
        processed_file_folder = train_dataset.processed_file_folder
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg["batch_size"],
            shuffle=is_shuffle,
            num_workers=cfg["num_workers"],
            pin_memory=cfg["pin_memory"],
        )
        log_string(
            "Completed after (Hours:Minutes:Seconds:Microseconds): "
            + str(datetime.datetime.now() - time_train_dataloader_start)
            + "\n"
        )
        train_str = "train"

    # Validation Dataloader
    log_string("Creating Validation Dataloader ...")
    time_val_dataloader_start = datetime.datetime.now()
    if debug:
        # use train data set for debug
        val_dataset = dataset(
            split=cfg["split"],
            split_type="train",
            debug=debug,
        )
    else:
        val_dataset = dataset(split=cfg["split"], split_type="val")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=is_shuffle,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
    )
    log_string(
        "Completed after (Hours:Minutes:Seconds:Microseconds): "
        + str(datetime.datetime.now() - time_val_dataloader_start)
        + "\n"
    )

    # get map dict
    if cfg["sc_img"] and "open" in cfg["data"]:
        valid_rdbs = [1, 2, 3, 4, 5, 6, 7]
        rdp_map_dict = {
            rdb_int: get_rdp_map(rdb_int, data_path=train_dataset.processed_file_folder)
            for rdb_int in valid_rdbs
        }
    else:
        rdp_map_dict = None

    print_datastats(train_dataloader, train_str)
    print_datastats(val_dataloader, "val")

    return (
        train_dataloader,
        val_dataloader,
        rdp_map_dict,
        processed_file_folder,
    )


def init_optimizer(model, cfg):
    # init optimizer
    grad_params = [pp for pp in model.parameters() if pp.requires_grad]
    if "cv" in model.tag:
        return None, None

    if cfg["optim"] == "sgd":
        optimizer = optim.SGD(grad_params, lr=cfg["base_lr"], momentum=0.9)
    else:
        optimizer = optim.Adam(grad_params, lr=cfg["base_lr"])

    # learning rate decay schedule
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, cfg["step_size"], gamma=cfg["gamma"]
    )

    return optimizer, scheduler


def fn_save_model(
    save_model,
    checkpoint,
    log_string,
    save_results,
    run_tag,
    model,
    seed,
    device,
    cfg,
    epoch,
    train_time,
    input_arg_dict,
    model_save_path,
    best_val_loss,
    train_sample,
):
    log_string("\n" + "-" * 100)
    if save_results:
        model_data = get_model_data(train_sample, cfg, device)

        if save_model == "last":
            log_string("Saving last model state ...")
            checkpoint = {
                "run_tag": run_tag,
                "model_tag": model.tag,
                "seed": seed,
                "device": device,
                "cfg": cfg,
                "epoch": epoch,
                "train_time": train_time,
                "model_class_name": model.__class__,
                "model_input_arg": input_arg_dict,
                "model_state_dict": model.state_dict(),
                "model_summary": summary(
                    model,
                    model_data,
                    show_input=True,
                    show_hierarchical=True,
                ),
            }
        else:
            log_string(
                "Saving model state after Epoch "
                + str(checkpoint["epoch"])
                + " ... (best_val_loss: {:.02f}, path: {})".format(
                    best_val_loss, model_save_path
                )
            )


def write_summary(
    tensorboard_path,
    run_tag,
    train_loss,
    epoch,
    val_loss,
    train_mean_rmse_over_horizon,
    val_mean_rmse_over_horizon,
    best_val_loss,
    train_mean_opt_rmse_over_horizon,
    val_mean_opt_rmse_over_horizon,
    train_selections_valid,
    val_samples_valid,
    train_samples,
    val_samples,
    train_correct_selections,
    val_correct_selections,
    best_val_correct_selections,
    cfg,
):
    t_writer_ = SummaryWriter(log_dir=tensorboard_path, comment=run_tag)

    if "g_sel" in tensorboard_path:
        strstr = cfg["loss_fn"].upper()
        is_g_sel = True
    else:
        strstr = "MSE SUM"
        is_g_sel = False

    # write loss
    t_writer_.add_scalar(
        "00 LOSS ({}) over epoch/a_train".format(strstr), train_loss, epoch
    )
    t_writer_.add_scalar(
        "00 LOSS ({}) over epoch/b_val".format(strstr), val_loss, epoch
    )
    t_writer_.add_scalar(
        "00 LOSS ({}) over epoch/c_train_-_val".format(strstr),
        train_loss - val_loss,
        epoch,
    )

    # write rmse
    t_writer_.add_scalar(
        "01 RMSE over epoch/a_train", train_mean_rmse_over_horizon, epoch
    )
    t_writer_.add_scalar("01 RMSE over epoch/b_val", val_mean_rmse_over_horizon, epoch)
    if is_g_sel:
        t_writer_.add_scalar(
            "01 RMSE over epoch/c_train_opt", train_mean_opt_rmse_over_horizon, epoch
        )
        t_writer_.add_scalar(
            "01 RMSE over epoch/d_val_opt", val_mean_opt_rmse_over_horizon, epoch
        )
        t_writer_.add_scalar(
            "01 RMSE over epoch/e_train_-_train_opt",
            train_mean_rmse_over_horizon - train_mean_opt_rmse_over_horizon,
            epoch,
        )
        t_writer_.add_scalar(
            "01 RMSE over epoch/f_val_-_val_opt",
            val_mean_rmse_over_horizon - val_mean_opt_rmse_over_horizon,
            epoch,
        )

        t_writer_.add_scalar(
            "02 Correct selection in percent over epoch/a_train",
            train_correct_selections * 100,
            epoch,
        )
        t_writer_.add_scalar(
            "02 Correct selection in percent over epoch/b_val",
            val_correct_selections * 100,
            epoch,
        )

        t_writer_.add_scalar(
            "03 Classification 'valid' in percent over epoch/a_train",
            train_selections_valid / train_samples * 100,
            epoch,
        )
        t_writer_.add_scalar(
            "03 Classification 'valid' in percent over epoch/b_val",
            val_samples_valid / val_samples * 100,
            epoch,
        )

    if epoch == cfg["epochs"]:
        hp_dict = {
            key: (
                val if type(val) in [int, float, str, bool, torch.Tensor] else str(val)
            )
            for key, val in cfg.items()
        }
        metric_dict = {
            "train_loss": train_loss,
            "train_rmse": train_mean_rmse_over_horizon,
            "val_loss": val_loss,
            "val_rmse": val_mean_rmse_over_horizon,
            "best_val_loss": best_val_loss,
        }
        if is_g_sel:
            metric_dict.update(
                {
                    "train_opt_rmse": train_mean_opt_rmse_over_horizon,
                    "val_opt_rmse": val_mean_opt_rmse_over_horizon,
                    "train_correct_selections_rel": train_correct_selections,
                    "val_correct_selections_rel": val_correct_selections,
                    "best_val_correct_selections_rel": best_val_correct_selections,
                    "train_selections_valid_rel": train_selections_valid
                    / train_samples,
                    "val_samples_valid_rel": val_samples_valid / train_samples,
                }
            )

        t_writer_.add_hparams(
            hparam_dict=hp_dict,
            metric_dict=metric_dict,
        )
    t_writer_.close()


def get_model_data(train_sample, cfg, device):
    _, hist_features, hist_len = train_sample.x.shape
    _, fut_features, fut_len = train_sample.y.shape

    if cfg["sc_img"]:
        if "open" in cfg["data"]:
            sc_img_dims = [256, 256]
        else:
            sc_img_dims = train_sample.sc_img.shape[2:]
        model_data = Data(
            x=torch.zeros(1, hist_len, hist_features + 1, dtype=torch.float),
            y=torch.zeros(1, fut_len, fut_features, dtype=torch.float),
            obj_class=torch.zeros(1, 1, 1, dtype=torch.float),
            obj_ref=torch.zeros(1, 1, 9, dtype=torch.float64),
            edge_index=torch.tensor([[], []], dtype=torch.long),
            sc_img=torch.zeros(1, 1, sc_img_dims[0], sc_img_dims[1], dtype=torch.float),
        ).to(device)
    else:
        model_data = Data(
            x=torch.zeros(1, hist_len, hist_features + 1, dtype=torch.float),
            y=torch.zeros(1, fut_len, fut_features, dtype=torch.float),
            obj_class=torch.zeros(1, 1, 1, dtype=torch.float),
            obj_ref=torch.zeros(1, 1, 9, dtype=torch.float64),
            edge_index=torch.tensor([[], []], dtype=torch.long),
        ).to(device)
    return model_data


def create_checkpoint(
    cfg,
    device,
    run_tag,
    model,
    seed,
    epoch,
    train_time,
    input_arg_dict,
    log_string,
    model_save_path,
    train_sample,
):
    model_data = get_model_data(train_sample, cfg, device)
    if "cv" in model.model_tag_list:
        model_summary = "cv_model"
    else:
        model_summary = summary(
            model,
            model_data,
            show_input=True,
            show_hierarchical=True,
        )
    checkpoint = {
        "run_tag": run_tag,
        "model_tag": model.tag,
        "seed": seed,
        "device": device,
        "cfg": cfg,
        "epoch": epoch,
        "train_time": train_time,
        "model_class_name": model.__class__,
        "model_input_arg": input_arg_dict,
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "model_summary": model_summary,
    }

    log_string(
        "Saving model state after Epoch " + str(checkpoint["epoch"]) + " ...",
        printer_=True,
    )

    torch.save(checkpoint, model_save_path)
    log_string("Model saved in file: " + model_save_path, printer_=True)

    return checkpoint


def train(
    model,
    cfg,
    run_tag=None,
    log_root_path=None,
    tensorboard=True,
    printer=True,
    save_model="best",
    train_dataloader=None,
):
    """
    Input
    -------
    MODEL: Model object (has to have attributes loss and tag)
    CFG: Configuration dict containing all training specific information:
        - SPLIT: Train and validation split (r_1, r_2 or r_3)
        - EPOCHS: Total number of training epochs
        - BATCH_SIZE
        - BASE_LR: Learning rate at the trainings begin
        - GAMMA: Learning rate decay factor
        - STEP_SIZE: Number of epochs after which the learning rate decay is applied
    RUN_TAG: Name for the run which is used as folder name within the log_rooth_path
    LOG_ROOT_PATH: Path that is specific for model and split that includes runs with different parameters or seeds
    TENSORBOARD: If True, make tensorboard entry in the log directory with train and val loss
    PRINTER: If True, print log messages in addition to writing them in the log file
    SAVE_MODEL: Model state can either be saved after the last epoch ('last'), or with the best results on the
                validation data ('best')

    Output
    -------
    BEST_VAL_LOSS: Best average Loss on Agent trajectory of Model on the Validation data
    BEST_VAL_SELECTIONS: Best average selection rate on Agent trajectory of Model on the Validation data
    """
    """Assign device"""
    device = cfg["device"]

    (
        log_fout,
        seed,
        run_tag,
        input_arg_dict,
        log_string,
        tensorboard_path,
        model_save_path,
        save_results,
    ) = create_log_path(log_root_path, run_tag, printer, model, device, cfg)

    """Cast seed"""
    torch.manual_seed(seed)
    if device != "cpu":
        torch.cuda.manual_seed(seed)

    """Get Data"""
    train_dataloader, val_dataloader, rdp_map_dict, processed_file_folder = get_data(
        cfg, log_string, debug=cfg["debug"], train_dataloader=train_dataloader
    )

    # send model to device and set it in training mode
    if "g_fusion" in model.tag:
        model.load_best_model()
        model.freeze_params(log_string=log_string)
        model.to(device)
    else:
        model.train()

    """Training and Validation"""
    optimizer, scheduler = init_optimizer(model, cfg)

    """Define function for training one epoch"""

    def iter_one_epoch(iter_dataloader, is_val=False):
        """
        Returns
        -------
        float
            loss.
        """
        epoch_time_start = datetime.datetime.now()
        if is_val:
            log_string("\n---- EPOCH %03d VALIDATION ----" % epoch)
            sc_img_mod = "val"
        else:
            log_string("\n---- EPOCH %03d TRAINING ----" % epoch)
            sc_img_mod = "train"
        log_string(str(epoch_time_start))

        if not is_val and "cv" not in model.tag:
            log_string("Learning rate: " + str(scheduler.get_last_lr()[0]))

        # initialize losses
        if device == "cpu":
            mse_over_horizon = torch.zeros(model.output_length)
            opt_mse_over_horizon = torch.zeros(model.output_length)
        else:
            mse_over_horizon = torch.zeros(model.output_length, device="cuda:0")
            opt_mse_over_horizon = torch.zeros(model.output_length, device="cuda:0")

        rmse_over_samples = []

        num_correct_selections = 0.0
        num_selections_valid = 0

        iter_loss_sum = 0.0
        num_samples = 0

        for iter_batch in tqdm(iter_dataloader, desc="Epoch " + str(epoch)):
            permute_input(data_batch=iter_batch)

            # Get sc images
            if cfg["sc_img"] and "open" in cfg["data"]:
                iter_batch.sc_img = get_sc_img_batch(
                    batch_obj_ref=iter_batch.obj_ref,
                    rdp_map_dict=rdp_map_dict,
                    data_path=processed_file_folder,
                    cfg=cfg,
                    mod=sc_img_mod,
                )

            # send batch to device
            iter_batch = iter_batch.to(device)
            if is_val or "cv" in model.tag:
                with torch.no_grad():
                    pred = model(iter_batch)
            else:
                # Zero the parameter gradients
                optimizer.zero_grad()
                pred = model(iter_batch)

            # Compute loss
            loss_output = model.loss(pred, iter_batch.y[:, :, :2])

            if "g_fusion" in model.tag:
                (
                    iter_loss_batch,
                    sel_model_mse,
                    selections,
                    opt_model_mse,
                    opt_selections,
                ) = loss_output
                valid_selection_list = selections < len(model.model_tag_list)
                num_selections_valid += int(sum(valid_selection_list))
                num_correct_selections += torch.sum(selections == opt_selections)
            else:
                iter_loss_batch = loss_output
                sel_model_mse = torch.pow(pred - iter_batch.y[:, :, :2], 2.0)
                selections = torch.zeros(
                    [iter_batch.y.shape[0]], dtype=int, device=device
                )
                opt_model_mse = sel_model_mse
                opt_selections = selections
                num_selections_valid += len(selections)
                num_correct_selections += len(selections)

            if not is_val and "cv" not in model.tag:
                # backpropagation
                iter_loss_batch.backward()

                # Gradient Clipping
                if cfg["clip"]:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.get("clip_grad", 10)
                    )

                # optimization step
                optimizer.step()

            # (R)mse
            mse_over_horizon += get_mse_over_horizon(sel_model_mse)
            opt_mse_over_horizon += get_mse_over_horizon(opt_model_mse)
            rmse_over_samples += get_rmse_over_samples(sel_model_mse).tolist()

            # Sum over loss
            iter_loss_sum += iter_loss_batch

            # num samples
            num_samples += iter_batch.x.shape[0]

        if num_selections_valid == 0:
            log_string("\n############ All selections invalid ############\n")
            rmse_over_horizon = torch.tensor(-1).cpu().detach().numpy()
            opt_rmse_over_horizon = torch.tensor(-1).cpu().detach().numpy()
        else:
            # get rmse values
            rmse_over_horizon = (
                torch.pow(mse_over_horizon / num_selections_valid, 0.5)
                .cpu()
                .detach()
                .numpy()
            )
            opt_rmse_over_horizon = (
                torch.pow(opt_mse_over_horizon / num_selections_valid, 0.5)
                .cpu()
                .detach()
                .numpy()
            )

        if num_samples == 0:
            iter_loss_sum = 999.9
            num_correct_selections = 0
        else:
            # Overall model loss
            iter_loss_sum /= num_samples

            # Overall correct selections
            num_correct_selections /= num_samples

        # print epoch summary
        _ = log_epoch(
            model.current_model,
            len(iter_dataloader) + 1,
            iter_loss_sum,
            rmse_over_horizon,
            opt_rmse_over_horizon,
            rmse_over_samples,
            num_correct_selections,
            is_val,
            log_string,
            epoch_time_start,
        )

        return (
            iter_loss_sum,
            rmse_over_horizon.mean(),
            opt_rmse_over_horizon.mean(),
            num_correct_selections,
            num_selections_valid,
            num_samples,
        )

    # Logging
    log_string("\n" + "-" * 100)
    time_start = datetime.datetime.now()
    log_string("\n**** start time: " + str(time_start) + " ****")
    best_val_loss = 100000.0
    best_rmse_val = 100000.0
    best_val_correct_selections = -1.0

    for epoch in range(1, cfg["epochs"] + 1):
        log_string("\n\n**** EPOCH %03d ****" % epoch)
        sys.stdout.flush()
        """Train one epoch"""
        store_net = False
        (
            train_loss,
            train_mean_rmse_over_horizon,
            train_mean_opt_rmse_over_horizon,
            train_correct_selections,
            train_selections_valid,
            train_samples,
        ) = iter_one_epoch(train_dataloader, is_val=False)
        """Val one epoch"""
        (
            val_loss,
            val_mean_rmse_over_horizon,
            val_mean_opt_rmse_over_horizon,
            val_correct_selections,
            val_samples_valid,
            val_samples,
        ) = iter_one_epoch(val_dataloader, is_val=True)
        train_time = datetime.datetime.now() - time_start
        """Create Checkpoint"""
        if (
            val_mean_rmse_over_horizon < best_rmse_val
            and (epoch + 1) >= cfg["epochs"] * 0.5
        ):
            best_rmse_val = val_mean_rmse_over_horizon
        if "g_sel" in model.tag and "sel" in cfg["best_val"]:
            store_net = val_correct_selections >= best_val_correct_selections
        else:
            store_net = val_loss < best_val_loss
        if save_results and epoch > 0 and save_model == "best" and store_net:
            checkpoint = create_checkpoint(
                cfg,
                device,
                run_tag,
                model,
                seed,
                epoch,
                train_time,
                input_arg_dict,
                log_string,
                model_save_path,
                train_sample=next(iter(train_dataloader)),
            )
            # Update last validation loss
            best_val_loss = val_loss
            best_val_correct_selections = val_correct_selections
            model.best_model_params = checkpoint["model_state_dict"]

        """Loss on Training and Validation data to Tensorboard"""
        if save_results and tensorboard is not None:
            write_summary(
                tensorboard_path,
                run_tag,
                train_loss,
                epoch,
                val_loss,
                train_mean_rmse_over_horizon,
                val_mean_rmse_over_horizon,
                best_val_loss,
                train_mean_opt_rmse_over_horizon,
                val_mean_opt_rmse_over_horizon,
                train_selections_valid,
                val_samples_valid,
                train_samples,
                val_samples,
                train_correct_selections,
                val_correct_selections,
                best_val_correct_selections,
                cfg,
            )

        # learning rate decay step
        if "cv" not in model.tag:
            scheduler.step()

        train_time = datetime.datetime.now() - time_start
        log_string("\nelapsed time: " + str(train_time))

        """Abort Training"""
        if "cv" in model.tag:
            break
        if train_mean_rmse_over_horizon < 0.0:
            break

    # save last learning rate
    if scheduler is not None:
        model.last_lr[model.current_model] = scheduler.get_last_lr()[0]

    log_string("\n**** end time: " + str(datetime.datetime.now()) + " ****")

    """Save trained Model"""
    fn_save_model(
        save_model,
        checkpoint,
        log_string,
        save_results,
        run_tag,
        model,
        seed,
        device,
        cfg,
        epoch,
        train_time,
        input_arg_dict,
        model_save_path,
        best_val_loss,
        train_sample=next(iter(train_dataloader)),
    )

    """Close Log"""
    log_fout.close()

    return best_rmse_val, best_val_correct_selections
