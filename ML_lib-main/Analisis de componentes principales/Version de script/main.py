from sklearn.decomposition import PCA
from myPCA import My_PCA
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

np.set_printoptions(precision=6, suppress=True)

base_dataset = pd.read_csv("titanic.csv")

#print(base_dataset.head())

percent_missing = base_dataset.isnull().sum() * 100 / len(base_dataset)
missing_value_df = pd.DataFrame({'column_name': base_dataset.columns,
                                 'percent_missing': percent_missing})

#print(missing_value_df)

#   El ID de un pasajero no influye en los datos, de hecho podría producir complicaciones
#   Al ser un numero ascendente

#   El nombre de un pasajero no influirá en si se salva

#   La columna de cabina tiene un 77% de datos faltantes

#   El ticket se puede eliminar por la misma razon que el nombre

titanic_data = base_dataset.drop(["PassengerId", "Name", "Cabin", "Ticket"], axis=1)
titanic_data.dropna(axis=0, how="any", inplace=True)


#   Estos datos son clases, por lo que se les aplica one hot encoding
cnames = ["Survived", "Pclass", "Sex", "Embarked"]
for cname in cnames:
    dummies = pd.get_dummies(titanic_data[cname], prefix=cname)
    titanic_data = titanic_data.drop(cname, axis=1)
    titanic_data = titanic_data.join(dummies)


#   Conversion de datos a numpy matrix
titanic_as_np = titanic_data.to_numpy()


#   Uso del metodo PCA implementado
mypca = My_PCA(n_components=4)
mypca.fit(titanic_as_np)


#   Uso del metodo PCA de la biblioteca sklearn
pca = PCA(n_components=4)

titanic_prep = mypca.center_and_scale(titanic_as_np)
pca.fit(titanic_prep)


#   Transformacion de los datos usando los 2 metodos diferentes
my_matrix = mypca.transform(titanic_prep)
sk_matrix = pca.transform(titanic_prep)




#   Plotting de los datos usando PCA
transformed_sk_data = pd.DataFrame({'F1' : sk_matrix[:, 0], 'F2': sk_matrix[:, 1], 'Survival': titanic_data['Survived_0']})
transformed_my_data = pd.DataFrame({'F1' : my_matrix[:, 0], 'F2': my_matrix[:, 1], 'Survival': titanic_data['Survived_0']})
sns.set()

fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10,5))
fig.suptitle('Comparación de puntos con PCA')

sns.scatterplot(ax=axes[0], x="F1", y="F2", hue='Survival', data=transformed_sk_data)
axes[0].set_title('Datos aplicando PCA de sklearn')

sns.countplot(ax=axes[1], x="F1", y="F2", hue='Survival', data=transformed_my_data)
axes[0].set_title('Datos aplicando mi PCA')

plt.show()

#   Aparecen 3 grupos de datos, 2 claramente separados y uno medio complicado
#   Los datos se separan bastante luego de aplicar el PCA por aumentar la varianza
#   La matriz de correlación indica que tan relacionadas están 2 variables

x, y = mypca.points

(fig, ax) = plt.subplots(figsize=(12, 12))
for i in range(0, len(titanic_data.columns)):
    ax.arrow(0, 0,
             x[i],y[i],
             head_width=0.1,head_length=0.1)
    plt.text(x[i],y[i],titanic_data.columns[i])
 
an = np.linspace(0, 2 * np.pi, 100)  # Add a unit circle for scale
plt.plot(np.cos(an), np.sin(an))
plt.axis('equal')
ax.set_title('Variable factor map')
plt.show()

#   A partir del sexo, clase y pago de pasaje se pueden obtener muchos datos, igualmente de dinde embarcó pero especificamente de un lugar.

