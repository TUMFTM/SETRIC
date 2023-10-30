import torch.nn.functional as F
from models.modules.module_utils import CustomModule


class CustomModel(CustomModule):
    def __init__(self):
        super(CustomModel, self).__init__()

    def loss(self, pred, target):
        return F.mse_loss(input=pred, target=target, reduction="sum")
