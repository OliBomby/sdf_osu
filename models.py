import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.conv(x)


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSampleBlock, self).__init__()
        self.double_conv = DoubleConvBlock(in_channels, out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        f = self.double_conv(x)
        p = self.maxpool(f)
        return f, p


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampleBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv = DoubleConvBlock(in_channels + out_channels, out_channels)

    def forward(self, x, conv_features):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, conv_features], dim=1)
        x = self.conv(x)
        return x


class EmbedBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(EmbedBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.act = nn.SiLU()  # Swish activation

    def forward(self, x):
        return self.act(self.linear(x))


class UNet3(nn.Module):
    def __init__(self, img_channels, embed_size):
        super(UNet3, self).__init__()

        self.down_blocks = nn.ModuleList([
            DownSampleBlock(img_channels, 32),
            DownSampleBlock(32, 64),
            DownSampleBlock(64, 128),
            DownSampleBlock(128, 256)
        ])

        self.embed_t1 = EmbedBlock(embed_size, 128)
        self.embed_t2 = EmbedBlock(embed_size, 128)
        self.linear_t = nn.Linear(256, 512)

        self.up_blocks = nn.ModuleList([
            UpSampleBlock(512, 256),
            UpSampleBlock(256, 128),
            UpSampleBlock(128, 64),
            UpSampleBlock(64, 32)
        ])

        self.conv_output = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, t1_embed_input, t2_embed_input):
        t1_emb = self.embed_t1(t1_embed_input)
        t2_emb = self.embed_t2(t2_embed_input)
        t_emb = torch.cat([t1_emb, t2_emb], dim=1)
        t_emb = self.linear_t(t_emb)

        outputs = []
        for down_block in self.down_blocks:
            f, x = down_block(x)
            outputs.append(f)

        x = self.down_blocks[-1].double_conv(x)
        x = x + t_emb.unsqueeze(-1).unsqueeze(-1)

        for up_block in self.up_blocks:
            x = up_block(x, outputs.pop())

        x = self.conv_output(x)
        x = self.flatten(x)
        x = self.softmax(x)

        return x
