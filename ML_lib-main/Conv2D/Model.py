from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import Softmax
from torch.nn import LeakyReLU


class CnnModel(Module):
    def __init__(self):
        super(CnnModel, self).__init__()
        # Primer capa convolucional:
        self.conv1 = Conv2d(in_channels=3, out_channels=6,
                            kernel_size=(3, 3), padding="same")
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2))
        self.lrelu1 = LeakyReLU(0.01)

        # Segunda capa convolucional
        self.conv2 = Conv2d(in_channels=6, out_channels=12,
                            kernel_size=(3, 3), padding="same")
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2))
        self.lrelu2 = LeakyReLU(0.01)

        # Tercera capa convolucional
        self.conv3 = Conv2d(in_channels=12, out_channels=24,
                            kernel_size=(3, 3), padding="same")
        self.maxpool3 = MaxPool2d(kernel_size=(2, 2))
        self.lrelu3 = LeakyReLU(0.01)

        # Capa de aprendizaje
        self.fc1 = Linear(in_features=384, out_features=192)
        self.lrelu4 = LeakyReLU(0.01)

        # Capa de clasificación
        self.fc2 = Linear(in_features=192, out_features=10)
        self.softmax = Softmax(dim=0)

    def forward(self, x):
        # Capa 1
        x = self.lrelu1(self.conv1(x))
        x = self.maxpool1(x)

        # Capa 2
        x = self.lrelu2(self.conv2(x))
        x = self.maxpool2(x)

        # Capa 3
        x = self.lrelu3(self.conv3(x))
        x = self.maxpool3(x)

        # Flattening
        x = x.view(-1, 24 * 4 * 4)

        # FC1
        x = self.lrelu4(self.fc1(x))

        output = self.softmax(self.fc2(x))

        return output
