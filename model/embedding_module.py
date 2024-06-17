
import torch
from torch import nn


class EmbeddingModule(torch):
    def __init__(self, feature_map_ch, cloud_ch, intermediate_channels, out_shape, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        projection = nn.Conv2d(feature_map_ch+cloud_ch, intermediate_channels)

        # point cloud features input
        fc_1 = nn.Linear()
        fc_2 = nn.Linear()

        # point cloud features output
        fc_3 = nn.Linear()
        fc_4 = nn.Linear()

    def adain(self, features, cloud):
        pass

    def forward(self, features, cloud):
        pass