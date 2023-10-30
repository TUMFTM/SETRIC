import os
import sys
import torch

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

from models.model_utils import CustomModel


class CV_Model(CustomModel):
    def __init__(
        self,
        input_features=8,
        dt_step_s=0.1,
        output_length=30,
        output_features=2,
        hist_steps=3,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Module description:
        Constant Velocity (CV) model for trajectory prediction.
        """

        super(CV_Model, self).__init__()
        self.tag = "cv_model"

        self.output_length = output_length
        self.hist_steps = hist_steps
        self.dt_step_s = dt_step_s
        self.input_features = input_features
        self.output_features = output_features
        self.device = device

        self.t_end = output_length * dt_step_s + dt_step_s
        self.t_array = torch.arange(
            start=self.dt_step_s, end=self.t_end, step=self.dt_step_s, device=device
        )

    def forward(self, batch):
        x = batch.x  # x shape: (N, seq_length, input_features)
        pred = torch.zeros(
            [batch.x.shape[0], self.output_length, self.output_features],
            device=self.device,
        )

        if self.input_features > 4:
            # x y heading v_x v_lat a_x a_y
            x_out = x[:, -1, 3:4].repeat(1, self.output_length) * self.t_array
            y_out = x[:, -1, 4:5].repeat(1, self.output_length) * self.t_array
            pred[:, :, 0] = x_out
            pred[:, :, 1] = y_out
        else:
            ds = (
                (x[:, -self.hist_steps :, :2] - x[:, -self.hist_steps - 1 : -1, :2])
                .pow(2)
                .sum(2)
                .sqrt()
                .sum(1)
            )
            dv = ds / (self.hist_steps * self.dt_step_s)
            pred[:, :, 0] = self.t_array * dv.unsqueeze(1).repeat(1, self.output_length)

        return pred


if __name__ == "__main__":
    # Debugging of the model
    from utils.processing import get_debug_data

    batch, cfg_train, net_config = get_debug_data(data="cr")

    model = CV_Model(
        input_features=net_config["input_features"],
        output_length=net_config["output_length"],
        dt_step_s=net_config["dt_step_s"],
    )
    model.to(cfg_train["device"])

    iter_loss = -1.0
    # send batch to device
    batch = batch.to(cfg_train["device"])
    for j in range(1):
        # Compute prediction
        pred = model(batch)
        # Compute loss
        train_loss_batch = model.loss(pred, batch.y[:, :, :2])
        iter_loss = train_loss_batch / batch.x.shape[0]

    print("Iter: {:03d}, Loss: {:.02f}".format(j + 1, iter_loss))
