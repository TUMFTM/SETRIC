import os
import sys

repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(repo_path)

import torch
from models.modules.module_utils import CustomModule


class SCIMG_Encoder(CustomModule):
    def __init__(self, num_img_filters=32, dec_img_size=32):
        super(SCIMG_Encoder, self).__init__()

        """
        Module description: 
        The CNN Encoder encodes the Scene Image
        """
        self.num_img_filters = num_img_filters
        self.dec_img_size = dec_img_size

        # Convolutional processing of scene representation
        self.sc_conv1 = torch.nn.Conv2d(
            1, self.num_img_filters, kernel_size=3, stride=1, padding=1
        )
        self.sc_conv2 = torch.nn.Conv2d(
            self.num_img_filters,
            self.num_img_filters,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.sc_conv3 = torch.nn.Conv2d(
            self.num_img_filters,
            self.num_img_filters,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.sc_conv4 = torch.nn.Conv2d(
            self.num_img_filters,
            self.num_img_filters,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.sc_conv5 = torch.nn.Conv2d(
            self.num_img_filters,
            self.dec_img_size,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.sc_conv6 = torch.nn.Conv2d(
            self.dec_img_size, self.dec_img_size, kernel_size=3, stride=1, padding=1
        )
        self.sc_conv7 = torch.nn.Conv2d(
            self.dec_img_size, self.dec_img_size, kernel_size=3, stride=1, padding=1
        )
        self.sc_conv8 = torch.nn.Conv2d(
            self.dec_img_size, self.dec_img_size, kernel_size=3, stride=1, padding=1
        )
        self.sc_maxpool = torch.nn.MaxPool2d((2, 2), padding=(0, 0))

    def forward(self, sc_img):
        # in torch.Size([N, 1, 256, 256])
        # Forward pass sc_img
        sc_img = self.sc_maxpool(self.sc_conv1(sc_img))
        # 256 --> 128
        sc_img = self.sc_maxpool(self.sc_conv2(sc_img))
        # 128 --> 64
        sc_img = self.sc_maxpool(self.sc_conv3(sc_img))
        # 64 --> 32
        sc_img = self.sc_maxpool(self.sc_conv4(sc_img))
        # 32 --> 16

        sc_img = self.sc_maxpool(self.sc_conv5(sc_img))
        # 16 --> 8
        sc_img = self.sc_maxpool(self.sc_conv6(sc_img))
        #  8 --> 4
        sc_img = self.sc_maxpool(self.sc_conv7(sc_img))
        #  4 --> 2
        sc_img = self.sc_maxpool(self.sc_conv8(sc_img))
        #  2 --> 1
        # torch.Size([N, 32, 1, 1])

        sc_img = torch.squeeze(sc_img, 2)
        sc_img = torch.squeeze(sc_img, 2)
        # torch.Size([N, 32])

        return sc_img
