import torch
import torch.nn as nn


def double_conv_block(in_channels, out_channels):
    # Conv2D then ReLU activation
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels)
    )


def downsample_block(in_channels, out_channels):
    f = double_conv_block(in_channels, out_channels)
    p = nn.MaxPool2d(kernel_size=2)
    return f, p


def upsample_block(in_channels, conv_features, out_channels):
    # upsample
    upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    x = upsample(in_channels)
    x = nn.Conv2d(x.shape[1] + conv_features.shape[1], out_channels, kernel_size=3, padding=1)(x)
    x = nn.BatchNorm2d(out_channels)(x)
    x = double_conv_block(out_channels, out_channels)(x)
    return x


class UNet1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet1, self).__init__()
        self.encoder = nn.ModuleList([downsample_block(in_channels, 64), downsample_block(64, 128),
                                      downsample_block(128, 256), downsample_block(256, 512)])
        self.bottleneck = double_conv_block(512, 1024)
        self.decoder = nn.ModuleList([upsample_block(1024, 512), upsample_block(512, 256),
                                      upsample_block(256, 128), upsample_block(128, 64)])
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        features = []
        for f, p in self.encoder:
            x = f(x)
            features.append(x)
            x = p(x)
        x = self.bottleneck(x)
        for conv_features in self.decoder:
            x = conv_features(x, features.pop())
        x = self.final_conv(x)
        x = x.view(x.size(0), -1)
        x = self.softmax(x)
        return x


class UNet2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet2, self).__init__()
        self.encoder = nn.ModuleList([downsample_block(in_channels, 64), downsample_block(64, 128),
                                      downsample_block(128, 256), downsample_block(256, 512)])
        self.bottleneck = double_conv_block(512, 1024)
        self.decoder = nn.ModuleList([upsample_block(1024, 512), upsample_block(512, 256),
                                      upsample_block(256, 128), upsample_block(128, 64)])
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        features = []
        for f, p in self.encoder:
            x = f(x)
            features.append(x)
            x = p(x)
        x = self.bottleneck(x)
        for conv_features in self.decoder:
            x = conv_features(x, features.pop())
        x = self.final_conv(x)
        x = x.view(x.size(0), -1)
        x = self.softmax(x)
        return x


class UNet3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3, self).__init__()
        self.t1_embed_input = nn.Linear(64, 128)
        self.t2_embed_input = nn.Linear(64, 128)
        self.t1_activation = nn.SiLU()
        self.t2_activation = nn.SiLU()
        self.t_emb = nn.Linear(256, 512)
        self.encoder = nn.ModuleList([downsample_block(in_channels, 32), downsample_block(32, 64),
                                      downsample_block(64, 128), downsample_block(128, 256)])
        self.bottleneck = double_conv_block(256, 512)
        self.decoder = nn.ModuleList([upsample_block(512, 256), upsample_block(256, 128),
                                      upsample_block(128, 64), upsample_block(64, 32)])
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, t1_embed_input, t2_embed_input):
        t1_emb = self.t1_embed_input(t1_embed_input)
        t1_emb = self.t1_activation(t1_emb)
        t2_emb = self.t2_embed_input(t2_embed_input)
        t2_emb = self.t2_activation(t2_emb)
        t_emb = torch.cat((t1_emb, t2_emb), dim=1)
        t_emb = self.t_emb(t_emb)

        features = []
        for f, p in self.encoder:
            x = f(x)
            features.append(x)
            x = p(x)
        x = self.bottleneck(x + t_emb[:, None, None])
        for conv_features in self.decoder:
            x = conv_features(x, features.pop())
        x = self.final_conv(x)
        x = x.view(x.size(0), -1)
        x = self.softmax(x)
        return x
