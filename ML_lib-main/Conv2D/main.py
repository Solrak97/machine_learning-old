from Model import CnnModel
from utils import transform, load_data
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import torch

# Data transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
names, x_train, y_train, x_test, y_test = load_data()
x_train, x_test, y_train, y_test = transform(x_train, x_test, y_train, y_test)

# Hiperparametros
INIT_LR = 1e-3
EPOCHS = 250
VAL_SIZE = 1000
TRAIN_SIZE = 1000

# Modelo
model = CnnModel()


# Optimizer y funcion de perdida
opt = Adam(model.parameters(), lr=INIT_LR)
lossFn = nn.CrossEntropyLoss()

# Historial de entrenamiento
loss_hist = []
train_acc_hist = []
val_acc_hist = []


# Entrenamiento del modelo
model.train()
for epoch in range(0, EPOCHS):
    # Training
    opt.zero_grad()

    pred = model(x_train)

    loss = lossFn(pred, y_train)
    loss.backward()
    opt.step()

    
    train_correct = (torch.argmax(pred, dim=1) == torch.argmax(
        y_train, 1)).type(torch.float).sum().item()

    # Validation
    with torch.no_grad():
        pred = model(x_test)

        val_correct = (torch.argmax(pred, dim=1) == torch.argmax(
            y_test, 1)).type(torch.float).sum().item()

        train_acc_hist.append(train_correct / TRAIN_SIZE)
        val_acc_hist.append(val_correct / VAL_SIZE)

    # Report

    print(f'''
    
    Epoch #{epoch}
    Loss                {loss}
    Train Correct:      {train_correct}
    Train Acc:          {train_acc_hist[-1]}
    Val Correct         {val_correct}
    Val Acc:            {val_acc_hist[-1]}

    ''')


plt.plot(range(EPOCHS), val_acc_hist, label="Validation")
plt.plot(range(EPOCHS), train_acc_hist,  label="Training")
plt.title('Accuracy')
plt.legend()
plt.show()
