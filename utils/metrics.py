import torch


def get_mse_over_horizon(model_mse):
    return model_mse.sum(2).sum(0)


def get_rmse_over_samples(model_mse):
    return torch.pow(model_mse.sum(2).sum(1) / model_mse.shape[1], 0.5)


def MSE(y_pred, y_gt, batch_first=False):
    """MSE Loss for single outputs.

    Arguments:
        y_pred {[type]} -- [description]
        y_gt {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    if batch_first:
        y_pred = y_pred.permute(1, 0, 2)
        y_gt = y_gt.permute(1, 0, 2)

    # If GT has not enough timesteps, shrink y_pred
    if y_gt.shape[0] < y_pred.shape[0]:
        y_pred = y_pred[: y_gt.shape[0], :, :]

    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    mse_det = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    count = torch.sum(torch.ones(mse_det.shape))
    mse = torch.sum(mse_det) / count
    return mse, mse_det
