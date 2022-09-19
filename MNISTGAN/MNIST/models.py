from torch.nn import Module, Conv2d, Linear
from torch.nn import ConvTranspose2d, BatchNorm2d
from torch.nn import functional as F
import torch

class Generator(Module):
    def __init__(self):

        # Superclass
        super(Generator, self).__init__()

        # Transposed Convolutionals
        # Have I gone too far? .... perhaps
        self.tconv1 = ConvTranspose2d(in_channels=48, out_channels=24, kernel_size=4, stride = 1)
        self.batchnorm1 = BatchNorm2d(num_features=24)
        self.tconv2 = ConvTranspose2d(in_channels=24, out_channels=12, kernel_size=5, stride=1)
        self.batchnorm2 = BatchNorm2d(num_features=12)
        self.tconv3 = ConvTranspose2d(in_channels=12, out_channels=6, kernel_size=5, stride=1)
        self.batchnorm3 = BatchNorm2d(num_features=6)
        self.tconv4 = ConvTranspose2d(in_channels=6, out_channels=1, kernel_size=6, stride=2)
        self.batchnorm4 = BatchNorm2d(num_features=1)

    pass


    def forward(self, x):
        x = self.tconv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)

        x = self.tconv2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)

        x = self.tconv3(x)
        x = self.batchnorm3(x)
        x = F.relu(x)

        x = self.tconv4(x)
        x = self.batchnorm4(x)
        x = torch.sigmoid(x)

        # Color scala
        out = x * 256
        return out


class Discriminator(Module):
    def __init__(self):

        # Superclass
        super(Discriminator, self).__init__()

        # Conv layers
        self.conv1 = Conv2d(in_channels=1, out_channels=12,
                            kernel_size=5)
        self.conv2 = Conv2d(in_channels=12, out_channels=24,
                            kernel_size=5)

        # Classification layers
        self.fc1 = Linear(in_features=384, out_features=192)
        self.fc2 = Linear(in_features=192, out_features=96)
        self.fc3 = Linear(in_features=96, out_features=1)

    pass

    def forward(self, x):

        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=(2, 2))
        x = F.leaky_relu(x, negative_slope=0.01)

        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=(2, 2))
        x = F.leaky_relu(x, negative_slope=0.01)

        x = x.view(-1, 24 * 4 * 4)

        x = self.fc1(x)
        x = F.leaky_relu(x, negative_slope=0.01)

        x = self.fc2(x)
        x = F.leaky_relu(x, negative_slope=0.01)

        x = self.fc3(x)
        out = torch.sigmoid(x)

        return out
