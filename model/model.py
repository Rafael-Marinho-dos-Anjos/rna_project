
import torch
from torch import nn
from torchvision.models import vgg16, VGG16_Weights

from model.embedding_module_simple import EmbeddingModule


transforms = VGG16_Weights.IMAGENET1K_FEATURES.transforms()

class Rec3D(nn.Module):
    def __init__(self, output_size=1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.transforms = VGG16_Weights.IMAGENET1K_FEATURES.transforms
        self.vgg_pretrained = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES)

        # ENCODER LAYERS (Vgg-16)

        self.enc_conv_0 = self.vgg_pretrained.features[0]
        self.enc_conv_1 = self.vgg_pretrained.features[2]
        # max_pooling -> skip_connection
        self.enc_conv_2 = self.vgg_pretrained.features[5]
        self.enc_conv_3 = self.vgg_pretrained.features[7]
        # max_pooling -> skip_connection
        self.enc_conv_4 = self.vgg_pretrained.features[10]
        self.enc_conv_5 = self.vgg_pretrained.features[12]
        self.enc_conv_6 = self.vgg_pretrained.features[14]
        # max_pooling -> skip_connection
        self.enc_conv_7 = self.vgg_pretrained.features[17]
        self.enc_conv_8 = self.vgg_pretrained.features[19]
        self.enc_conv_9 = self.vgg_pretrained.features[21]
        # max_pooling -> skip_connection
        self.enc_conv_10 = self.vgg_pretrained.features[24]
        self.enc_conv_11 = self.vgg_pretrained.features[26]
        self.enc_conv_12 = self.vgg_pretrained.features[28]
        # max_pooling


        # DECODER LAYERS

        # upsampling
        self.dec_conv_0 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.dec_conv_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.dec_conv_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # concatenation -> upsampling
        self.dec_conv_3 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.dec_conv_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.dec_conv_5 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        # concatenation -> upsampling
        self.dec_conv_6 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.dec_conv_7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.dec_conv_8 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        # concatenation -> upsampling
        self.dec_conv_9 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec_conv_10 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        # concatenation -> upsampling
        self.dec_conv_11 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec_conv_12 = nn.Conv2d(64, output_size, kernel_size=3, padding=1)

        # embedding modules
        self.emb_mod_0 = EmbeddingModule([112, 112, 64], [128, 128, 128])
        self.emb_mod_1 = EmbeddingModule([56, 56, 128], [128, 128, 128])
        self.emb_mod_2 = EmbeddingModule([28, 28, 256], [128, 128, 128])
        self.emb_mod_3 = EmbeddingModule([14, 14, 512], [128, 128, 128])
        self.emb_mod_4 = EmbeddingModule([7, 7, 512], [128, 128, 128])

        # OTHER LAYERS
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.batch_norm32 = nn.BatchNorm2d(32)
        self.batch_norm64 = nn.BatchNorm2d(64)
        self.batch_norm128 = nn.BatchNorm2d(128)
        self.batch_norm256 = nn.BatchNorm2d(256)
        self.batch_norm512 = nn.BatchNorm2d(512)


    def freeze_encoder_block(self):
        self.enc_conv_0.requires_grad = False
        self.enc_conv_1.requires_grad = False
        self.enc_conv_2.requires_grad = False
        self.enc_conv_3.requires_grad = False
        self.enc_conv_4.requires_grad = False
        self.enc_conv_5.requires_grad = False
        self.enc_conv_6.requires_grad = False
        self.enc_conv_7.requires_grad = False
        self.enc_conv_8.requires_grad = False
        self.enc_conv_9.requires_grad = False
        self.enc_conv_10.requires_grad = False
        self.enc_conv_11.requires_grad = False
        self.enc_conv_12.requires_grad = False

    def unfreeze_encoder_block(self):
        self.enc_conv_0.requires_grad = True
        self.enc_conv_1.requires_grad = True
        self.enc_conv_2.requires_grad = True
        self.enc_conv_3.requires_grad = True
        self.enc_conv_4.requires_grad = True
        self.enc_conv_5.requires_grad = True
        self.enc_conv_6.requires_grad = True
        self.enc_conv_7.requires_grad = True
        self.enc_conv_8.requires_grad = True
        self.enc_conv_9.requires_grad = True
        self.enc_conv_10.requires_grad = True
        self.enc_conv_11.requires_grad = True
        self.enc_conv_12.requires_grad = True

    def conv_block(self, x, *args):
        """
            Does sucessives convolutions with relu activation and batch normalization
            for every convolution in arguments
        """
        for conv_layer in args:
            x = conv_layer(x)
            x = self.relu(x)
            if x.shape[1] == 32:
                x = self.batch_norm32(x)
            elif x.shape[1] == 64:
                x = self.batch_norm64(x)
            elif x.shape[1] == 128:
                x = self.batch_norm128(x)
            elif x.shape[1] == 256:
                x = self.batch_norm256(x)
            elif x.shape[1] == 512:
                x = self.batch_norm512(x)
        
        return x

    def forward(self, x, pc):
        
        # ENCODING
        skip_0 = self.conv_block(x, self.enc_conv_0, self.enc_conv_1)
        skip_0 = self.pool(skip_0)

        skip_1 = self.conv_block(skip_0, self.enc_conv_2, self.enc_conv_3)
        skip_1 = self.pool(skip_1)

        skip_2 = self.conv_block(skip_1, self.enc_conv_4, self.enc_conv_5, self.enc_conv_6)
        skip_2 = self.pool(skip_2)
        
        skip_3 = self.conv_block(skip_2, self.enc_conv_7, self.enc_conv_8, self.enc_conv_9)
        skip_3 = self.pool(skip_3)
        
        bottleneck = self.conv_block(skip_3, self.enc_conv_10, self.enc_conv_11, self.enc_conv_12)
        bottleneck = self.pool(bottleneck)

        # DECODING
        upsampled = self.upsampling(bottleneck)
        upsampled = self.conv_block(upsampled, self.dec_conv_0, self.dec_conv_1, self.dec_conv_2)

        upsampled = torch.cat((upsampled, skip_3), dim=1)
        upsampled = self.upsampling(upsampled)
        upsampled = self.conv_block(upsampled, self.dec_conv_3, self.dec_conv_4, self.dec_conv_5)

        upsampled = torch.cat((upsampled, skip_2), dim=1)
        upsampled = self.upsampling(upsampled)
        upsampled = self.conv_block(upsampled, self.dec_conv_6, self.dec_conv_7, self.dec_conv_8)

        upsampled = torch.cat((upsampled, skip_1), dim=1)
        upsampled = self.upsampling(upsampled)
        upsampled = self.conv_block(upsampled, self.dec_conv_9, self.dec_conv_10)

        upsampled = torch.cat((upsampled, skip_0), dim=1)
        upsampled = self.upsampling(upsampled)
        upsampled = self.conv_block(upsampled, self.dec_conv_11, self.dec_conv_12)

        cloud_0 = self.emb_mod_4(bottleneck, pc)
        cloud_1 = self.emb_mod_3(skip_3, cloud_0)
        cloud_2 = self.emb_mod_2(skip_2, cloud_1)
        cloud_3 = self.emb_mod_1(skip_1, cloud_2)
        cloud = self.emb_mod_0(skip_0, cloud_3)

        return self.sig(upsampled), cloud


if __name__ == "__main__":
    Rec3D()(torch.rand((1, 3, 224, 224)))