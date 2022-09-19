import pandas as pd
from neuralnet import DenseNN

# Laboratorio 4: Redes Neuronales
# Integrantes:
# - Luis Carlos Quesada Rodriguez / B65580
# - Gianfranco Bagnarello Hernandez / B70866

def train(net, epochs, x, y):
    for epoch in range(1, epochs + 1):
        error = 0
        for _x, _y in zip(x, y):
            error += net.backpropagation(_x, _y)
            net.step()

        print(f'Epoch #{epoch}\nError: {error}')


base_dataset = pd.read_csv("Data/titanic.csv")

titanic_data = base_dataset.drop(["PassengerId", "Name", "Cabin", "Ticket"], axis=1)
titanic_data.dropna(axis=0, how="any", inplace=True)

cnames = ["Pclass", "Sex", "Embarked"]
for cname in cnames:
    dummies = pd.get_dummies(titanic_data[cname], prefix=cname)
    titanic_data = titanic_data.drop(cname, axis=1)
    titanic_data = titanic_data.join(dummies)




y = titanic_data['Survived'].to_numpy()
X = titanic_data.drop(columns='Survived').to_numpy()
y = y.reshape(-1 , 1)

layers = [12,7,5,1]
activation = ['l','l','l','s']

nn = DenseNN(layers, activation, seed=0)
nn.train(lr=1e-3, momentum=1e-3, decay=1e-5)

train(nn, 100, X, y)

pred = nn.predict(X[1, :])
#print(pred.shape)