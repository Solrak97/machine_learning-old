from torch import nn
from torch.optim import Adam
import pandas as pd
import torch

from pre_processing import load_data
from model_training import train_kfold
from model_evaluation import confussion_matrix, summary


from Dias_Model import Dias_Model
from Sentan_Model import Sentan_Model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_path = 'data.pkl'

features, labels, classes = load_data(data_path)

model = Sentan_Model
optimizer = Adam
lossFn = nn.CrossEntropyLoss()

train_kfold(model, features, labels,  optimizer, lossFn, device, epochs=500)

