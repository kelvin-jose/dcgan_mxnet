import mxnet.gluon.nn as nn


def generator_block(in_channels, out_channels, kernel_size, stride, padding=0, last_layer=False):
    net = nn.Sequential()
    net.add(nn.Conv2DTranspose(out_channels, kernel_size=kernel_size, strides=stride, in_channels=in_channels,
                               padding=padding, use_bias=False))
    if last_layer:
        net.add(nn.Activation(activation='tanh'))
    else:
        net.add((nn.BatchNorm()))
        net.add(nn.Activation(activation='relu'))
    return net


def discriminator_block(in_channels, out_channels, kernel_size, stride, padding=0, last_layer=False):
    net = nn.Sequential()
    net.add(nn.Conv2D(in_channels=in_channels, channels=out_channels, kernel_size=kernel_size, strides=stride,
                      padding=padding))
    if last_layer:
        net.add(nn.Activation(activation='sigmoid'))
    else:
        net.add(nn.BatchNorm())
        net.add(nn.LeakyReLU(alpha=0.2))
    return net


class Generator(nn.Block):
    def __init__(self):
        super().__init__()
        self.project_reshape = generator_block(100, 1024, 4, 1)
        self.conv_1 = generator_block(1024, 512, 4, 2, 1)
        self.conv_2 = generator_block(512, 256, 4, 2, 1)
        self.conv_3 = generator_block(256, 128, 4, 2, 1)
        self.conv_4 = generator_block(128, 3, 4, 2, 1, True)

    def forward(self, input):
        x = self.project_reshape(input)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        return x


class Discriminator(nn.Block):
    def __init__(self):
        super().__init__()
        self.conv_1 = discriminator_block(3, 128, 4, 2, 1)
        self.conv_2 = discriminator_block(128, 256, 4, 2, 1)
        self.conv_3 = discriminator_block(256, 512, 4, 2, 1)
        self.conv_4 = discriminator_block(512, 1024, 4, 2, 1)
        self.final_layer = discriminator_block(1024, 1, 4, 1, 0, True)

    def forward(self, input):
        x = self.conv_1(input)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.final_layer(x)
        return x
