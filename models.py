import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif class_name.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)


class G(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(G, self).__init__()
        channels = input_shape[0]
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        ]

        # Down Sampling (Encoding)
        in_features = out_features

        for _ in range(2):
            out_features = out_features * 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features

        #  Residual Block
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        #  Up Sampling (Decode)
        for _ in range(2):
            out_features //= 2  # int
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features

        # Output layer
        model += [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(out_features,channels,7),
            nn.Tanh()
        ]
        self.main = nn.Sequential(*model)

    def forward(self,x):
        return self.main(x)


class D(nn.Module):
    def __init__(self, input_shape):
        super(D, self).__init__()
        channels, height, width = input_shape

        # PatchGAN
        self.output_shape = (1, height // 2**4, width // 2**4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2,inplace=True))
            return layers

        self.main = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),  # list to items
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1,0,1,0)),
            nn.Conv2d(512,1,4,padding=1)
        )

    def forward(self,x):
        return self.main(x)

