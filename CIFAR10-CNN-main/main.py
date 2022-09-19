from Model import CnnModel
from utils import transform, load_data
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np

# Custom subdirectory to find images
DIRECTORY = "images"


# Data transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes, x_train, y_train, x_test, y_test = load_data(DIRECTORY)
x_train, x_test, y_train, y_test = transform(x_train, x_test, y_train, y_test)

# Hiperparametros
INIT_LR = 1e-3
EPOCHS = 250
VAL_SIZE = 10000
TRAIN_SIZE = 50000

# Modelo
model = CnnModel()


# Optimizer y funcion de perdida
opt = Adam(model.parameters(), lr=INIT_LR)
lossFn = nn.CrossEntropyLoss()

# Historial de entrenamiento
loss_hist = []
train_acc_hist = []
val_acc_hist = []
loss = 0
trainable = False

if trainable:
    # Entrenamiento del modelo
    model.train()
    for epoch in range(0, EPOCHS):
        # Training

        x_train.to(device)
        y_train.to(device)

        opt.zero_grad()

        pred = model(x_train)

        _loss = lossFn(pred, y_train)
        loss_dif = loss - _loss
        loss = _loss
        loss.backward()
        opt.step()

        train_correct = (torch.argmax(pred, dim=1) == torch.argmax(
            y_train, 1)).type(torch.float).sum().item()

        # Validation
        with torch.no_grad():

            x_test.to(device)
            y_test.to(device)
            pred = model(x_test)

            val_correct = (torch.argmax(pred, dim=1) == torch.argmax(
                y_test, 1)).type(torch.float).sum().item()

            train_acc_hist.append(train_correct / TRAIN_SIZE)
            val_acc_hist.append(val_correct / VAL_SIZE)

        # Report

        print(f'''

        Epoch #{epoch}
        Loss                {loss}
        Loss Dif:           {loss_dif}
        Train Correct:      {train_correct}
        Train Acc:          {train_acc_hist[-1]}
        Val Correct         {val_correct}
        Val Acc:            {val_acc_hist[-1]}

        ''')

        if epoch % 100 == 0:
            torch.save(model.state_dict(), f'Model_it{epoch}.pt')


    torch.save(model.state_dict(), f'Model_final.pt')
    plt.plot(range(EPOCHS), val_acc_hist, label="Validation")
    plt.plot(range(EPOCHS), train_acc_hist,  label="Training")
    plt.title('Accuracy')
    plt.legend()
    plt.show()


else:
    model = CnnModel()
    model.load_state_dict(torch.load('Model_final.pt'))



y_pred = model(x_test)

y_pred = np.argmax(y_pred.detach().numpy(), axis=1)
y_true = np.argmax(y_test.detach().numpy(), axis=1)

cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 10, index=[i for i in classes],
                     columns=[i for i in classes])
plt.figure(figsize=(12, 7))
sn.heatmap(df_cm, annot=True)
plt.show()
