from torch.nn import Module, Conv1d, Linear, ReLU, Softmax, MaxPool1d, BatchNorm1d, Dropout

# Modelo simplificado
class Sentan_simple(Module):

    def __init__(self):
        super(Sentan_simple, self).__init__()

        self.conv1 = Conv1d(in_channels=1, out_channels=25,
                            kernel_size=5, stride=1)
        

        self.conv2 = Conv1d(in_channels=25, out_channels=50,
                            kernel_size=5, stride=1)


        # Clasificador
        self.fc1 = Linear(in_features=50*185, out_features=8)
        
        # Intra layer Functions
        self.batchnorm1 = BatchNorm1d(num_features=25)
        self.batchnorm2 = BatchNorm1d(num_features=50)
        self.relu = ReLU()
        self.dropout = Dropout(p=0.1)
        self.softmax = Softmax(dim=-1)


    def forward(self, x):
        # Capa 1
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.relu(self.batchnorm2(self.conv2(x)))
        
        x = x.view(-1, 50*185)

        # Clasificador
        x = self.fc1(x)
        output = self.softmax(x)

        return output