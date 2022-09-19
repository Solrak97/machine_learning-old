import tree as dtree
from tree import DecisionTree
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


np.set_printoptions(precision=6, suppress=True)

base_dataset = pd.read_csv("Data/titanic.csv")

percent_missing = base_dataset.isnull().sum() * 100 / len(base_dataset)
missing_value_df = pd.DataFrame({'column_name': base_dataset.columns,
                                 'percent_missing': percent_missing})

titanic_data = base_dataset.drop(["PassengerId", "Name", "Cabin", "Ticket"], axis=1)
titanic_data.dropna(axis=0, how="any", inplace=True)


#print(titanic_data.head())

Y = titanic_data["Survived"]
X = titanic_data.drop(columns=["Survived"])

X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.75, test_size=0.25, random_state=21)

model = DecisionTree()
model.fit(X_train, y_train, 10)

predict = model.predict(X_test)

matrix = dtree.calculate_confusion_matrix(predict, y_test)

print("==============================")
print("            TITANIC")
for entry in matrix:
    print (entry, matrix[entry])
print("\n\n\n\n")

mushrooms_data = pd.read_csv("Data/mushrooms.csv")
#print(mushrooms_data.head()) 

Y = mushrooms_data["class"]
X = mushrooms_data.drop(columns=["class"])


X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.75, test_size=0.25, random_state=21)

tree = DecisionTree()
tree.fit(X_train, y_train, 10)
predict = tree.predict(X_test)
matrix = dtree.calculate_confusion_matrix(predict, y_test)


print("==============================")
print("            MUSHROOMS")
for entry in matrix:
    print (entry, matrix[entry])


d = tree.to_dict()

print(d)