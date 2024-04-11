"""Main class of fusion model."""

import os
import sys
import copy

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

import torch
import torch.nn.functional as F
import torch.nn as nn

from utils.metrics import get_rmse_over_samples, get_mis_pred_over_samples
from models.model_utils import CustomModel
from models.CV_Model import CV_Model
from models.modules.LSTM_Encoder_Modules import LSTM_Encoder
from models.modules.SCIMG_Encoder import SCIMG_Encoder
from models.modules.LSTM_Decoder_Modules import Indy_Decoder
from models.modules.Embedding_Modules import (
    ML_Linear_Embedding,
    GNN_Embedding_CustomConv,
)
from models.modules.Link_Pred_Modules import Distance_Edge_Module


class Fusion_Model(CustomModel):
    """Main class of fusion model."""

    def __init__(
        self,
        cfg,
        net_config,
        is_inference=False,
    ):
        """Initialize model.

        Model description: This model selects its prediction
        from the outputs of multiple models within a list of sub models.
        """
        super(Fusion_Model, self).__init__()

        self.__dict__.update(net_config)
        self.__dict__.update(cfg)

        self.tag = "g_fusion"
        self.base_tag = "g_fusion"
        self.is_inference = is_inference
        self.current_model = self.model_tag_list[0]

        self.nllloss = cfg["loss_fn"] == "nll"

        self.trajectory_number = (
            len(self.model_tag_list) + 1
        )  # additional selection of invalid

        # cv model
        self.cv_model = CV_Model(
            input_features=self.input_features,
            dt_step_s=self.dt_step_s,
            output_length=self.output_length,
            output_features=self.output_features,
            device=cfg["device"],
        )

        # Linear Embedding
        self.l_lstm_encoder_embedding = ML_Linear_Embedding(
            input_features=self.input_features,
            num_hidden_layers=self.linear_hidden_layers,
            hidden_size=self.linear_hidden_size,
            embedding_size=self.input_embedding_size,
        )

        # GNN Embedding
        self.link_pred = Distance_Edge_Module(distance=self.gnn_distance)
        self.dg_lstm_encoder_emb_gnn = GNN_Embedding_CustomConv(
            input_features=self.input_features,
            message_size=self.gnn_message_size,
            aggr=self.gnn_aggr,
            embedding_size=self.gnn_embedding_size,
            num_hidden_layers=self.gnn_num_hidden_layers,
            hidden_size=self.gnn_hidden_size,
        )

        # Conversion Layer D_GNN
        self.dg_lstm_encoder_emb_linear = nn.Linear(
            self.gnn_embedding_size, self.input_embedding_size
        )

        # Encoder objects
        self.encoder = LSTM_Encoder(
            embedding_size=self.input_embedding_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
        )

        if self.dg_encoder:
            # Encoder objects
            self.dg_lstm_encoder = LSTM_Encoder(
                embedding_size=self.input_embedding_size,
                hidden_size=self.lstm_hidden_size,
                num_layers=self.lstm_num_layers,
            )

        # Encoder scene images
        if self.sc_img:
            self.scimg_encoder = SCIMG_Encoder(
                num_img_filters=self.num_img_filters,
                dec_img_size=self.dec_img_size,
            )

        # Decoder L_LSTM
        # dynamic embedding
        self.l_lstm_dyn_emb = torch.nn.Linear(
            self.lstm_hidden_size, self.dyn_embedding_size
        )
        # LSTM decoder
        self.l_lstm_decoder = Indy_Decoder(
            decoder_size=self.decoder_size,
            dyn_embedding_size=self.dyn_embedding_size
            + self.dec_img_size * int(self.sc_img),
            output_length=net_config["output_length"],
        )

        # Dencoder DGNN
        # dynamic embedding
        self.dg_lstm_dyn_emb = torch.nn.Linear(
            self.lstm_hidden_size, self.dyn_embedding_size
        )
        # LSTM decoder
        self.dg_lstm_decoder = Indy_Decoder(
            decoder_size=self.decoder_size,
            dyn_embedding_size=self.dyn_embedding_size
            + self.dec_img_size * int(self.sc_img),
            output_length=net_config["output_length"],
        )

        # Decoder selector
        if self.g_sel_dyn_emb == "no_lstm":
            self.g_sel_dyn_emb_dg_lstm = torch.nn.Linear(
                self.lstm_hidden_size, self.dyn_embedding_size
            )
            self.g_sel_dyn_emb_l_lstm = torch.nn.Linear(
                self.lstm_hidden_size, self.dyn_embedding_size
            )
            # Linear output selector
            self.g_sel_output_linear_1 = nn.Linear(
                self.dyn_embedding_size * 2 + self.dec_img_size * int(self.sc_img),
                self.g_sel_output_linear_size * 2,
            )
            self.g_sel_output_linear_2 = nn.Linear(
                self.g_sel_output_linear_size * 2, self.g_sel_output_linear_size
            )
            self.g_sel_output_linear_3 = nn.Linear(
                self.g_sel_output_linear_size, self.trajectory_number
            )
        else:
            if self.g_sel_dyn_emb:
                self.g_sel_dyn_emb_state = torch.nn.Linear(
                    self.input_features, self.dyn_embedding_size
                )
                g_sel_decoder_input_size = (
                    self.dyn_embedding_size + self.dec_img_size * int(self.sc_img)
                )
            else:
                g_sel_decoder_input_size = (
                    self.input_features + self.dec_img_size * int(self.sc_img)
                )
            self.g_sel_decoder = nn.LSTM(
                input_size=g_sel_decoder_input_size,
                hidden_size=self.lstm_hidden_size,
                num_layers=self.lstm_num_layers,
                bias=True,
                batch_first=True,
            )

            # Linear output selector
            self.g_sel_output_linear_1 = nn.Linear(
                self.lstm_hidden_size, self.g_sel_output_linear_size * 4
            )
            self.g_sel_output_linear_2 = nn.Linear(
                self.g_sel_output_linear_size * 4, self.g_sel_output_linear_size * 2
            )
            self.g_sel_output_linear_3 = nn.Linear(
                self.g_sel_output_linear_size * 2, self.trajectory_number
            )

        # Costum activation function
        self.leaky_relu = torch.nn.LeakyReLU(0.1)

        # Selector loss settings
        weight_t = torch.ones(
            len(self.model_tag_list) + 1
        )  # additional selection of invalid
        weight_t[0] = 1.0
        weight_t[-1] = self.weight
        if self.nllloss:
            self.process_target = torch.nn.LogSoftmax(dim=1)
            self.CELoss = torch.nn.NLLLoss(weight=weight_t)
        else:
            self.CELoss = torch.nn.CrossEntropyLoss(weight=weight_t)

        # Init flags
        self.best_model_params = None
        self.encoder_trained = False
        self.sc_img_trained = False

        # save last learning rate
        self.last_lr = {}

        # Reset all net branches
        self.reset_all_net_branches()

        # Selector submodels
        _ = self.get_sub_model_fns()

        # Output net size
        _ = self.output_tot_params()

    def get_tot_params(self):
        """Get total number of params in model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def output_tot_params(self):
        """Output number of trainable parameters of the net."""
        tot_params = self.get_tot_params()
        print(
            "\n"
            + f"{'Total number of parameters' : <30}{str(tot_params) : >7}{'100.00 %' : >10}"
        )
        for key, mod in self._modules.items():
            try:
                mod_params = sum(p.numel() for p in mod.parameters() if p.requires_grad)
                if mod_params > 0:
                    i2 = "{:.02f} %".format(mod_params / tot_params * 100)
                    print(f"{key : <30}{str(mod_params) : >7}{i2 : >10}")
            except Exception as er:
                print("No trainable params in {}, Erro: {}".format(key, er))

    def get_sub_model_fns(self):
        """Get sub model prediction functions."""
        self.model_fns = []
        for model in self.model_tag_list:
            if "cv" in model:
                self.model_fns.append(self.pred_cv)
            if "l_lstm" in model:
                self.model_fns.append(self.pred_l_lstm)
            if "dg_lstm" in model:
                self.model_fns.append(self.pred_dg_lstm)

    def update_model_tag(self, model_tag):
        """Update the model tag to current model."""
        self.tag = self.base_tag + "_" + model_tag
        self.current_model = model_tag

    def fn_sc_img_encoder(self, batch):
        """Call scene image encoder."""
        if self.sc_img_enc is not None or not self.sc_img:
            return

        self.sc_img_enc = self.scimg_encoder(batch.sc_img)

    def pred_cv(self, batch):
        """Predict with CV-model."""
        return self.cv_model(batch)

    def pred_l_lstm(self, batch):
        """Predict with L_LSTM-model."""
        self.fn_encoder_l_lstm(batch)
        pred = self.fn_decoder_l_lstm()
        # N, output_length, output_features
        return pred

    def pred_dg_lstm(self, batch):
        """Predict with DG-LSTM-model."""
        self.fn_encoder_dg_lstm(batch)
        pred = self.fn_decoder_dg_lstm()
        # N, output_length, output_features
        return pred

    def fw_selector(self, batch):
        """Forward function of G_sel."""
        selection_weights = self.pred_selector(batch)
        all_trajectory_pred = self.get_all_trajectory_pred(batch)

        return selection_weights, all_trajectory_pred

    def get_common_latent_encoding(self, batch):
        """Get common latent enconding."""
        self.fn_encoder_l_lstm(batch)
        self.fn_encoder_dg_lstm(batch)

        dyn_emb_l_lstm = self.g_sel_dyn_emb_l_lstm
        dyn_emb_dg_lstm = self.g_sel_dyn_emb_dg_lstm

        enc_l = self.leaky_relu(
            dyn_emb_l_lstm(
                self.hidden_l.view(self.hidden_l.shape[1], self.hidden_l.shape[2])
            )
        )
        enc_dg = self.leaky_relu(
            dyn_emb_dg_lstm(
                self.hidden_dg.view(self.hidden_dg.shape[1], self.hidden_dg.shape[2])
            )
        )

        enc = self.concat_with_sc_img_enc(enc_dg, enc_l)
        return enc

    def concat_with_sc_img_enc(self, enc, enc_2=None):
        """Concatenate latent space with scene image."""
        if self.sc_img_enc is not None:
            if enc_2 is not None:
                return torch.cat((enc, enc_2, self.sc_img_enc), 1)
            else:
                return torch.cat((enc, self.sc_img_enc), 1)

        if enc_2 is not None:
            return torch.cat((enc, enc_2), 1)
        return enc

    def pred_selector(self, batch):
        """Predicts with G_sel."""
        if self.g_sel_dyn_emb == "no_lstm":
            enc = self.get_common_latent_encoding(batch)
            # Predict class probabilities / weights
            x = self.leaky_relu(self.g_sel_output_linear_1(enc))
            x = self.leaky_relu(self.g_sel_output_linear_2(x))  # selection weights
            x = self.g_sel_output_linear_3(x)  # selection weights
            # torch.Size([12, 3])
            return x

        self.fn_encoder_dg_lstm(batch)
        if self.g_sel_dyn_emb:
            # enc: N, embedding_size
            enc = self.leaky_relu(self.g_sel_dyn_emb_state(batch.x[:, -2, :]))
        else:
            # enc: N, input_features
            enc = batch.x[:, -1, :]
        enc = self.concat_with_sc_img_enc(enc)
        x, (_, _) = self.g_sel_decoder(enc.unsqueeze(1), (self.hidden_dg, self.cell_dg))
        x = x.squeeze(1)

        # Predict class probabilities / weights
        x = self.leaky_relu(self.g_sel_output_linear_1(x))
        x = self.leaky_relu(self.g_sel_output_linear_2(x))  # selection weights
        x = self.g_sel_output_linear_3(x)  # selection weights
        # torch.Size([12, 3])

        return x

    def fn_encoder_l_lstm(self, batch):
        """Call L_LSTM encoding step (embedding + encoder)."""
        if self.hidden_l is not None:
            return

        x_emb = self.l_lstm_encoder_embedding(batch.x)
        # x_emb: N, hist_len, embedding_size

        # LSTM Encoder
        self.hidden_l, _ = self.fn_l_enc_lstm(x_emb)

    def fn_encoder_dg_lstm(self, batch):
        """Call DG_LSTM encoding step (embedding + encoder)."""
        if self.hidden_dg is not None:
            return

        # Link Prediction
        if not hasattr(
            batch, "ptr"
        ):  # In case no dataloader with batches is used, ptr might not be available
            batch.ptr = torch.tensor([0, batch.x.shape[0]])
        edge_index_pred = self.link_pred(
            x_time_series=batch.x, ptr=batch.ptr, obj_ref=batch.obj_ref
        )
        # edge_index_pred: 2, N

        # GNN Embedding
        x_e = self.dg_lstm_encoder_emb_gnn(batch.x, edge_index_pred, batch.obj_ref)
        # x_e: N, hist_len, g_sel_output_linear_size * 4

        # embedding to enc
        x_emb = self.leaky_relu(self.dg_lstm_encoder_emb_linear(x_e))
        # x_emb: N, hist_len, self.input_embedding_size

        if self.dg_encoder:
            self.hidden_dg, self.cell_dg = self.fn_dg_enc_lstm(x_emb)
        else:
            self.hidden_dg, self.cell_dg = self.fn_l_enc_lstm(x_emb)

    def fn_dg_enc_lstm(self, x_emb):
        """Call DG_LSTM encoder."""
        # LSTM Encoder
        hidden, cell = self.dg_lstm_encoder(x_emb)
        # hidden, cell: num_layers_enc, N, lstm_hidden_size
        return hidden, cell

    def fn_l_enc_lstm(self, x_emb):
        """Call L_LSTM encoder."""
        # LSTM Encoder
        hidden, cell = self.encoder(x_emb)
        # hidden, cell: num_layers_enc, N, lstm_hidden_size
        return hidden, cell

    def fn_decoder_l_lstm(self):
        """Call L_LSTM decoder."""
        # LSTM Decoder
        enc = self.leaky_relu(
            self.l_lstm_dyn_emb(
                self.hidden_l[-1].view(self.hidden_l.shape[1], self.hidden_l.shape[2])
            )
        )
        # enc: N, dyn_embedding_size
        enc = self.concat_with_sc_img_enc(enc)
        # enc: N, dyn_embedding_size + self.dec_img_size
        return self.l_lstm_decoder(enc)

    def fn_decoder_dg_lstm(self):
        """Call DG_LSTM decoder."""
        # LSTM Decoder
        enc = self.leaky_relu(
            self.dg_lstm_dyn_emb(
                self.hidden_dg[-1].view(
                    self.hidden_dg.shape[1], self.hidden_dg.shape[2]
                )
            )
        )
        enc = self.concat_with_sc_img_enc(enc)
        return self.dg_lstm_decoder(enc)

    def get_all_trajectory_pred(self, batch):
        """Get predictions from all submodels."""
        all_trajectory_pred = torch.zeros(
            [
                len(self.model_tag_list),
                batch.x.shape[0],
                self.output_length,
                self.output_features,
            ],
            device=batch.x.device.type,
        )
        for sub_model in range(len(self.model_tag_list)):
            all_trajectory_pred[sub_model] = self.model_fns[sub_model](batch)
        return all_trajectory_pred

    def forward(self, batch):
        """Forward function during training."""
        # Reset all net branches
        self.reset_all_net_branches()

        # Image Encoder
        self.fn_sc_img_encoder(batch=batch)

        # Selector Model
        if self.current_model == "g_sel":
            pred = self.fw_selector(batch)
        # Prediction Models
        elif self.current_model == "cv":
            pred = self.pred_cv(batch)
        elif self.current_model == "l_lstm":
            pred = self.pred_l_lstm(batch)
        elif self.current_model == "dg_lstm":
            pred = self.pred_dg_lstm(batch)
        else:
            raise NotImplementedError(
                "Invalid model choice '{}'".format(self.current_model)
            )

        # Unified output in case of inference
        if self.is_inference:
            if self.current_model == "g_sel":
                pred, valid_selections = self.get_selections(pred)
                idx_list = [
                    j for j in range(len(valid_selections)) if valid_selections[j]
                ]
            else:
                idx_list = [j for j in range(pred.shape[0])]

            return pred, idx_list

        return pred

    def reset_all_net_branches(self):
        """Reset all net branches."""
        self.sc_img_enc = None
        self.hidden_l = None
        self.hidden_dg = None
        self.cell_dg = None

    def zero_loss(self, device):
        """Get zero loss."""
        zero_mse = torch.zeros((0, 50, 2), device=device)
        zero_sel = torch.zeros((0), device=device)
        print("   NO SAMPLES TO CALCULATE LOSS   ".center(80, "#"))
        return (0.0, zero_mse, zero_sel, zero_mse, zero_sel)

    def loss(self, pred, y):
        """Calculate the loss."""
        if y.shape[0] == 0:
            return self.zero_loss(device=y.device)

        if self.current_model == "g_sel":
            return self.loss_sel(pred, y)

        model_loss = self.loss_single_model(pred, y)
        model_mse = torch.pow(pred - y, 2.0)
        selections = torch.zeros([y.shape[0]], dtype=int, device=y.device.type)
        return (model_loss, model_mse, selections, model_mse, selections)

    @staticmethod
    def loss_single_model(pred, y):
        """Calculate the loss of a single model."""
        return F.mse_loss(input=pred, target=y, reduction="sum")

    def get_selections(self, pred, y=None, valids_only=False):
        """Get selections of G_sel."""
        selection_weights, all_trajectory_pred = pred
        selections = selection_weights.max(dim=1).indices
        valid_selections = selections < len(self.model_tag_list)
        if valids_only:
            return valid_selections
        sel_trajectories = all_trajectory_pred[
            selections[valid_selections], valid_selections
        ]

        # return only selections
        if y is None:
            return sel_trajectories, valid_selections

        # get loss of selected
        sel_model_mse = F.mse_loss(
            input=sel_trajectories, target=y[valid_selections], reduction="none"
        )

        (
            opt_selections_non_filtered,
            non_filtered_opt_model_mse,
        ) = self.get_opt_selection_non_filtered(all_trajectory_pred, y, with_mse=True)

        return (
            sel_model_mse,
            selections,
            selection_weights,
            non_filtered_opt_model_mse,
            opt_selections_non_filtered,
        )

    @staticmethod
    def get_opt_selection_non_filtered(
        all_trajectory_pred, y, with_mse=False, with_traj=False
    ):
        """Get optimal selection unfiltered."""
        # get loss of best selection
        all_trajectory_loss = torch.pow(all_trajectory_pred - y, 2.0)
        # same as F.mse_loss(input=all_trajectory_pred, target=y, reduction="none")
        _, opt_selections_non_filtered = all_trajectory_loss.sum((2, 3)).min(dim=0)

        if with_mse:
            non_filtered_opt_model_mse = all_trajectory_loss[
                opt_selections_non_filtered, opt_selections_non_filtered > -1
            ]

        if with_traj:
            non_filtered_traj_pred = all_trajectory_pred[
                opt_selections_non_filtered, opt_selections_non_filtered > -1
            ]

        if with_mse and with_traj:
            return (
                opt_selections_non_filtered,
                non_filtered_opt_model_mse,
                non_filtered_traj_pred,
            )

        if with_mse:
            return opt_selections_non_filtered, non_filtered_opt_model_mse

        if with_traj:
            return opt_selections_non_filtered, non_filtered_traj_pred

        return opt_selections_non_filtered

    def get_invalid_pred(self, model_mse):
        """Get invalid prediction."""
        if self.__dict__.get("invalid_def", "rmse") == "rmse":
            opt_model_rmse_over_samples = get_rmse_over_samples(model_mse)
            # select error by rmse, either abs (err_rmse > 0) or quantile
            if self.err_rmse > 0:
                selection_invalid = opt_model_rmse_over_samples > self.err_rmse
            else:
                quantile = torch.quantile(
                    opt_model_rmse_over_samples, self.err_quantile
                )
                selection_invalid = opt_model_rmse_over_samples > quantile
        else:
            # select error by missrate
            selection_invalid = get_mis_pred_over_samples(model_mse)

        return selection_invalid

    def get_error_vals(self, pred, y, get_invalids_only=False):
        """Get error values."""
        (
            sel_model_mse,
            selections,
            selection_weights,
            non_filtered_opt_model_mse,
            opt_selections,
        ) = self.get_selections(pred, y)

        # filter invalids
        selection_invalid = self.get_invalid_pred(non_filtered_opt_model_mse)

        if not max(selection_invalid):
            print("ALL VALID")
        opt_selections[selection_invalid] = self.trajectory_number - 1
        opt_model_mse = non_filtered_opt_model_mse[selection_invalid is False]

        return (
            sel_model_mse,
            selections,
            selection_weights,
            opt_model_mse,
            opt_selections,
        )

    def loss_sel(self, pred, y):
        """Get loss of G_sel model."""
        (
            sel_model_mse,
            selections,
            selection_weights,
            opt_model_mse,
            opt_selections,
        ) = self.get_error_vals(pred, y)

        # calc loss
        if self.nllloss:
            loss_input = self.process_target(selection_weights)
        CELoss = self.CELoss(input=loss_input, target=opt_selections)

        return (CELoss, sel_model_mse, selections, opt_model_mse, opt_selections)

    def load_best_model(self):
        """Load best model from previous model training."""
        if self.best_model_params is not None:
            self.load_state_dict(self.best_model_params)

    def freeze_params(self, log_string=None):
        """Freeze parameters that should not be trained."""
        self.unfreeze()
        self.freeze()

        # keep all params frozen if current model is "cv"
        if self.current_model == "cv":
            self.freeze_stats(log_string=log_string)
            return

        # sc_img is always trained with the first model
        if not self.sc_img_trained and self.sc_img:
            self.scimg_encoder.unfreeze()
            self.sc_img_trained = True

        # unfreeze l_lstm net
        if self.current_model == "l_lstm":
            if not self.encoder_trained:
                self.encoder.unfreeze()
                self.encoder_trained = True

            # linear embedding
            self.l_lstm_encoder_embedding.unfreeze()

            # decoder (in case of indy additional embedding)
            self.l_lstm_dyn_emb.weight.requires_grad = True
            self.l_lstm_dyn_emb.bias.requires_grad = True
            self.l_lstm_decoder.unfreeze()

        # unfreeze g_lstm and dg_lstm
        elif self.current_model == "dg_lstm":
            # gnn embedding
            self.link_pred.unfreeze()
            self.dg_lstm_encoder_emb_gnn.unfreeze()
            self.dg_lstm_encoder_emb_linear.requires_grad_(True)

            # dg encoder
            if self.dg_encoder:
                self.dg_lstm_encoder.unfreeze()
            elif not self.encoder_trained:
                self.encoder.unfreeze()
                self.encoder_trained = True

            # decoder (in case of indy additional embedding)
            self.dg_lstm_dyn_emb.requires_grad_(True)
            self.dg_lstm_decoder.unfreeze()
        # unfreeze g_sel
        elif self.current_model == "g_sel":
            self.unfreeze_g_sel()

        self.freeze_stats(log_string=log_string)

    def freeze_stats(self, log_string):
        """Output freeze stats."""
        unfrozen_params = len([p for p in self.parameters() if p.requires_grad])
        if unfrozen_params > 0:
            self.frozen = False

        strstr = "Unfrozen: {} of {} layers, model = {}".format(
            unfrozen_params,
            len([p for p in self.parameters()]),
            self.current_model,
        )
        if log_string is None:
            print(strstr)
        else:
            log_string(strstr)

    def unfreeze_g_sel(self):
        """Unfreeze selector head."""
        # both gnn's not trained: train gnn_embedding
        if "dg_lstm" not in self.model_tag_list:
            # gnn embedding
            self.link_pred.unfreeze()
            self.dg_lstm_encoder_emb_gnn.unfreeze()
            self.g_sel_dyn_emb_dg_lstm.requires_grad_(True)

        # train l_lstm for selector model
        if not self.encoder_trained and self.g_sel_dyn_emb == "no_lstm":
            self.encoder.unfreeze()
            self.encoder_trained = True

        # decoder time series incl. embedding
        if self.g_sel_dyn_emb == "no_lstm":
            self.g_sel_dyn_emb_dg_lstm.requires_grad_(True)
            self.g_sel_dyn_emb_l_lstm.requires_grad_(True)
        else:
            for ww in self.g_sel_decoder._flat_weights:
                ww.requires_grad = True
            if self.g_sel_dyn_emb:
                self.g_sel_dyn_emb_state.requires_grad_(True)

        # fullies to selector
        self.g_sel_output_linear_1.requires_grad_(True)
        self.g_sel_output_linear_2.requires_grad_(True)
        self.g_sel_output_linear_3.requires_grad_(True)

    def output_rmse_over_horizon(self, pred, y, n_samples):
        """Output rmse over horizon."""
        model_mse = torch.pow(pred - y, 2.0)
        rmse_over_horizon = torch.pow(get_mse_over_horizon(model_mse) / n_samples, 0.5)
        return rmse_over_horizon.mean()

    def eval_model(self, batch, model_tag):
        """Evaluate all submodels."""
        old_tag = copy.deepcopy(self.current_model)
        self.current_model = model_tag
        pred = self.forward(batch)
        assert batch.y.shape[1] == self.output_length
        gt = batch.y[:, :, :2]
        if "g_sel" in model_tag:
            pred, valid_selections = model.get_selections(pred)
            gt = gt[valid_selections]

        self.current_model = old_tag
        return self.output_rmse_over_horizon(pred, gt, batch.x.shape[0])


if __name__ == "__main__":
    # Debugging of the model
    import torch.optim as optim
    from utils.processing import get_debug_data
    from utils.scheduling import print_overview
    from utils.metrics import get_mse_over_horizon

    batch, cfg_train, net_config = get_debug_data(data="cr")

    device = cfg_train["device"]

    model = Fusion_Model(cfg=cfg_train, net_config=net_config)

    for model_tag in model.model_tag_list + ["g_sel"]:
        print("\n\n########## Training ################")
        print("Training of {} ...".format(model_tag))
        model.update_model_tag(model_tag)

        # send batch to device
        model.load_best_model()
        model.freeze_params()
        model.to(device)
        grad_params = [pp for pp in model.parameters() if pp.requires_grad]

        if "cv" not in model.tag:
            optimizer = optim.Adam(grad_params, lr=cfg_train["base_lr"])
        batch = batch.to(device)

        min_loss = 1000000.0
        for j in range(cfg_train["epochs"]):
            # Zero the parameter gradients
            if "cv" not in model.tag:
                optimizer.zero_grad()

            # Compute prediction
            pred = model(batch)

            # Compute loss
            (
                iter_loss_batch,
                sel_model_mse,
                selections,
                opt_model_mse,
                opt_selections,
            ) = model.loss(pred, batch.y[:, :, :2])

            if "cv" not in model.tag:
                # Backpropagation
                iter_loss_batch.backward()

                # Gradient Clipping
                if cfg_train["clip"]:
                    torch.nn.utils.clip_grad_norm_(grad_params, 10)

                # optimization step
                optimizer.step()

            # Metrics
            valid_selection_list = selections < len(model.model_tag_list)
            num_samples_valid = int(sum(valid_selection_list))

            mse_over_horizon = get_mse_over_horizon(sel_model_mse)
            opt_mse_over_horizon = get_mse_over_horizon(opt_model_mse)
            rmse_over_horizon = (
                torch.pow(mse_over_horizon / num_samples_valid, 0.5)
                .cpu()
                .detach()
                .numpy()
            )
            opt_rmse_over_horizon = (
                torch.pow(opt_mse_over_horizon / num_samples_valid, 0.5)
                .cpu()
                .detach()
                .numpy()
            )

            num_correct_selections = torch.sum(selections == opt_selections)

            iter_loss_sum = iter_loss_batch / batch.x.shape[0]
            num_correct_selections = num_correct_selections / batch.x.shape[0]

            if iter_loss_sum < min_loss:
                min_loss = iter_loss_sum
                model.best_model_params = copy.deepcopy(model.state_dict())
                best_model_j = j

            print_overview(
                model.current_model,
                j + 1,
                iter_loss_sum,
                rmse_over_horizon.mean(),
                opt_rmse_over_horizon.mean(),
                num_correct_selections,
                end_str="\r",
            )

            if "cv" in model.tag:
                break

        print_overview(
            model.current_model,
            j + 1,
            iter_loss_sum,
            rmse_over_horizon.mean(),
            opt_rmse_over_horizon.mean(),
            num_correct_selections,
            end_str="\n",
        )

        # eval models
        print("######### Evaluation ###############")
        model.load_best_model()
        res_list = []
        for cm in model.model_tag_list + ["g_sel"]:
            mean_rmse_over_horizon = model.eval_model(batch, model_tag=cm)
            res_list.append(
                "{}: {:.02f} cm".format(cm.upper(), 100 * mean_rmse_over_horizon)
            )
        print(" | ".join(res_list))
