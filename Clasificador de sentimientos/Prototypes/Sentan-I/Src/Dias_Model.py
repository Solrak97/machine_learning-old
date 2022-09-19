from torch.nn import Module, Conv1d, Linear, ReLU, Softmax, MaxPool1d, BatchNorm1d, Dropout


class Dias_Model(Module):

    def __init__(self):
        super(Dias_Model, self).__init__()

        # Capa 1
        self.conv1 = Conv1d(in_channels=1, out_channels=256,
                            kernel_size=5, stride=1)
        self.batchnorm1 = BatchNorm1d(num_features=256)
        self.relu1 = ReLU()

        # Capa 2
        self.conv2 = Conv1d(in_channels=256, out_channels=128,
                            kernel_size=5, stride=1)
        self.relu2 = ReLU()
        self.droput1 = Dropout(p=0.1)
        self.batchnorm2 = BatchNorm1d(num_features=128)

        # Maxpool
        self.maxpool = MaxPool1d(kernel_size=8)

        # Intermedias
        self.intermedia_conv1 = Conv1d(
            in_channels=128, out_channels=128, kernel_size=5, stride=1)
        self.intermeida_relu1 = ReLU()

        self.intermedia_conv2 = Conv1d(
            in_channels=128, out_channels=128, kernel_size=5, stride=1)
        self.intermeida_relu2 = ReLU()

        self.intermedia_conv3 = Conv1d(
            in_channels=128, out_channels=128, kernel_size=5, stride=1)
        self.intermedia_batchnorm = BatchNorm1d(num_features=128)
        self.intermeida_relu3 = ReLU()
        self.intermedia_dropout = Dropout(p=0.2)

        # Final
        self.final_conv = Conv1d(
            in_channels=128, out_channels=128, kernel_size=5, stride=1)
        self.final_dropout = Dropout(p=0.2)

        # Clasificador
        self.final_clasificador = Linear(in_features=896, out_features=8)
        self.final_batchnorm = BatchNorm1d(num_features=8)
        self.final_softmax = Softmax(dim=-1)

    def forward(self, x):
        # Capa 1
        x = self.relu1(self.batchnorm1(self.conv1(x)))

        # Capa 2
        x = self.batchnorm2(self.droput1(self.relu2(self.conv2(x))))

        # Maxpool
        x = self.maxpool(x)

        # Intermedias 1
        x = self.intermeida_relu1(self.intermedia_conv1(x))

        # Intermedias 2
        x = self.intermeida_relu2(self.intermedia_conv2(x))

        # Intermedias 3
        x = self.intermedia_dropout(self.intermeida_relu3(
            self.intermedia_batchnorm(self.intermedia_conv3(x))))
        # Final

        x = self.final_conv(x)
        x = self.final_dropout(x.view(-1, 128*7))

        # Clasificador
        x = self.final_batchnorm(self.final_clasificador(x))
        output = self.final_softmax(x)

        return output
