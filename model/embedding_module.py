
import torch
from torch import nn


class EmbeddingModule(nn.Module):
    def __init__(self, feature_map_sh, n_points, points_ch, pcf_shape, out_shape, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.proj_conv = nn.Conv2d(feature_map_sh[2]+points_ch, out_shape[1], 3)
        self.proj_fc_1 = nn.Linear(feature_map_sh[0]*feature_map_sh[1]*out_shape[1], n_points*out_shape[1])
        self.proj_fc_2 = nn.Linear(n_points*out_shape[1], n_points*out_shape[1])

        # point cloud encoding
        self.pc_enc = nn.Linear(3*n_points, feature_map_sh[0]*feature_map_sh[1]*points_ch)

        # point cloud features input
        self.fc_1 = nn.Linear(pcf_shape[0]*pcf_shape[1], n_points*out_shape[1])
        self.fc_2 = nn.Linear(n_points*out_shape[1], n_points*out_shape[1])

        # point cloud features output
        self.fc_3 = nn.Linear(n_points*out_shape[1]*2, n_points*out_shape[1])
        self.fc_4 = nn.Linear(n_points*out_shape[1], n_points*out_shape[1])

        self.activation = nn.Sigmoid()

    def adain(self, features, cloud_features):
        mean_x = torch.mean(features)
        dev_x  = torch.std(features)
        mean_y = torch.mean(cloud_features)
        dev_y  = torch.std(cloud_features)

        return dev_x * (cloud_features - mean_y) / dev_y + mean_x

    def forward(self, features, cloud, cloud_features):
        cloud = self.pc_enc(torch.flatten(cloud))

        pc_out = torch.cat((cloud, features), dim=2)
        pc_out = self.proj_conv(pc_out)
        pc_out = self.activation(pc_out)
        pc_out = self.proj_fc_1(pc_out)
        pc_out = self.activation(pc_out)
        pc_out = self.proj_fc_2(pc_out)

        pc_feat = self.fc_1(cloud_features)
        pc_feat = self.activation(pc_feat)
        pc_feat = self.fc_2(pc_feat)

        pc_feat = self.adain(features, pc_feat)

        output = torch.cat((pc_out, pc_feat))

        output = self.fc_3(output)
        output = self.activation(output)
        output = self.fc_4(output)

        return output



if __name__ == "__main__":
    model = EmbeddingModule([112, 112, 64], 128, 64, [128, 64], [128, 128])
    model(torch.rand((64, 112, 112), torch.rand((128, 3), torch.rand((128, 64)))))