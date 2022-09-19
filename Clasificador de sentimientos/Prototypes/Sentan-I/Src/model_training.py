import matplotlib.pyplot as plt
import torch
import numpy as np
import wandb

# Esto cubre el notebook y correr desde el main
try:
    from pre_processing import split, to_tensor

except:
    from .pre_processing import split, to_tensor


def train(model, X, y, device, optimizer, lossFn, epochs):
    wandb.init(project="Pruebas W&B")

    wandb.config = {
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 128
    }

    X, y = to_tensor(X, y)
    X.to(device)
    y.to(device)

    # Entrenamiento del modelo
    model.train()
    for epoch in range(0, epochs):

        optimizer.zero_grad()

        pred = model(X)

        loss = lossFn(pred, y)
        wandb.log({"loss1": loss})
        wandb.log({"loss2": loss})
        loss.backward()
        optimizer.step()


def train_kfold(model_builder, X, y,  optimizer_builder, lossFn, device='cpu', epochs=100):
    folds = 1

    # Kfold Historial
    total_loss_hist = []
    total_train_acc_hist = []
    total_val_acc_hist = []

    # K-folds
    for train_idx, test_idx in split(X, y):

        model = model_builder().to(device)
        optimizer = optimizer_builder(
            model.parameters(), lr=1e-3, weight_decay=1e-5)

        # Fold Data
        x_train = X[train_idx]
        y_train = y[train_idx]

        x_test = X[test_idx]
        y_test = y[test_idx]

        x_train, y_train = to_tensor(x_train, y_train)
        x_test, y_test = to_tensor(x_test, y_test)

        # Historial de entrenamiento
        loss_hist = []
        train_acc_hist = []
        val_acc_hist = []
        loss = 0

        # Train loop
        model.train()
        for epoch in range(0, epochs):

            # Training
            x_train = x_train.to(device)
            y_train = y_train.to(device)

            optimizer.zero_grad()

            pred = model(x_train)

            _loss = lossFn(pred, y_train)
            _loss.backward()
            optimizer.step()

            loss = _loss
            loss_hist.append(loss)

            train_correct = (torch.argmax(pred, dim=1) ==
                             y_train).type(torch.float).sum().item()

        # Validation
            with torch.no_grad():

                x_test = x_test.to(device)
                y_test = y_test.to(device)
                pred = model(x_test)

                val_correct = (torch.argmax(pred, dim=1) ==
                               y_test).type(torch.float).sum().item()

                train_acc_hist.append(train_correct / len(x_train))
                val_acc_hist.append(val_correct / len(x_test))

        # Reporte
        total_loss_hist.append(loss_hist)
        total_train_acc_hist.append(train_acc_hist)
        total_val_acc_hist.append(val_acc_hist)

        print(f'''
        Fold: {folds}
        Training Accuracy:       {train_acc_hist[-1]}
        Validation Accuracy:     {val_acc_hist[-1]}
        ''')

        folds += 1

    #   kfold means
    #mean_loss = np.array(total_loss_hist)
    mean_train_accuracy = np.array(total_train_acc_hist)
    mean_val_accuracy = np.array(total_val_acc_hist)

    #mean_loss = mean_loss.mean()
    mean_train_accuracy = np.mean(mean_train_accuracy, axis=0)
    mean_val_accuracy = np.mean(mean_val_accuracy, axis=0)

    # Training vs validaton plot
    plt.plot(range(epochs), mean_val_accuracy,
             label=f"Mean Validation")
    plt.plot(range(epochs), mean_train_accuracy,
             label=f"Mean Training")
    plt.title('K-fold Mean Validation Vs Training')
    plt.legend()
    plt.show()
