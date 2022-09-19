from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np



# Version propia para convertir una columna de tipo vector en un unico vector numpy
def to_numpy(x):
    data = np.array([])
    for v in x:
        data = np.concatenate((data, v), axis=0)
    data = data.reshape(-1, 193)
    return data


# Carga los datos en formato Numpy como X, y, tags
def load_data(path):
    data = pd.read_pickle(path)
    features = to_numpy(data['Composite_Vector'])
    ordinal_labels = data['Ordinal_Emotion']
    label_encoding = data.groupby('Ordinal_Emotion')['Emotion'].unique()
    ordinal_labels = ordinal_labels.to_numpy().astype(np.int64)
    return features, ordinal_labels, label_encoding


# Crea la división de los datos en k folds
# El formato es una lista de listas
# Se puede ver como [[Train], [Test]]
def split(x, y, splits = 5, random_state = 0):
    skf = StratifiedKFold(n_splits=splits, random_state = random_state, shuffle=True)
    return skf.split(x, y)


# Codifica los valores de y en one hot encoding y transforma el vector
# En un tensor apto para usarse en Torch
def to_tensor(x, y):
    x = torch.unsqueeze(torch.tensor(x), dim=1).type(torch.float32)
    y = torch.from_numpy(y)
    return x, y


def to_labels(preds, labels):
    return [labels[i.item()][0] for i in preds]
