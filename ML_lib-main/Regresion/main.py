from matplotlib.pyplot import title
import linreg
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
fish = pd.read_csv('Data/fish_perch.csv')

y = fish['Weight']
X = fish.drop(columns= 'Weight')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=21)

regressor = linreg.LinearRegression()
e = regressor.fit(X_train, y_train, max_epochs=100000, threshold=1e-7, learning_rate=1e-7,
        momentum=0, decay=0, error='mse', regularization='none', _lambda=0)

prediction = regressor.predict(X_test)


mse = linreg.MSE(y_test, prediction)
# Grid search... eventualmente
print(mse)
sns.lineplot(y = e, x = [i for i in range(len(e))])
plt.show()