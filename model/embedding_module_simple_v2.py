
import torch
from torch import nn
from torchvision.transforms import v2

class EmbeddingModule(nn.Module):
    def __init__(self, feature_map_sh, out_shape, inner_dims = (256, 256, 512), *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.proj_conv_1 = nn.Conv2d(feature_map_sh[2]+out_shape[2], inner_dims[0], 3, padding=1)
        self.proj_conv_2 = nn.Conv2d(inner_dims[0], inner_dims[1], 3, padding=1)
        self.proj_conv_3 = nn.Conv2d(inner_dims[1], inner_dims[2], 3, padding=1)
        self.fusion_conv = nn.Conv2d(inner_dims[-1], out_shape[2], 1)

        self.resize = v2.Resize(size=(out_shape[0], out_shape[1]))

        self.activation = nn.Sigmoid()

    def forward(self, features, cloud):
        features = self.resize(features)

        pc_out = torch.cat((cloud, features), dim=-3)
        pc_out = self.proj_conv_1(pc_out)
        pc_out = self.proj_conv_2(pc_out)
        pc_out = self.proj_conv_3(pc_out)
        pc_out = self.fusion_conv(pc_out)
        pc_out = self.activation(pc_out)

        return pc_out



if __name__ == "__main__":
    model = EmbeddingModule([112, 112, 64], [128, 128, 128])
    print(model(torch.rand((64, 112, 112)), torch.rand((128, 128, 128))).shape)