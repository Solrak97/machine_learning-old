import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.metrics as metrics

# Graficación de matríz de confusión
def confussion_matrix(y_true, y_pred, classes):

    cf_matrix = metrics.confusion_matrix(y_true, y_pred)

    df_cm = pd.DataFrame(cf_matrix, index=[i[0] for i in classes],
                         columns=[i[0] for i in classes])

    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt='g', cmap='YlOrBr')

    plt.show()


def accuracy(y_true, y_pred):
    return etrics.accuracy_score(y_true, y_pred)
    


def summary(y_true, y_pred):
    print(metrics.classification_report(y_true, y_pred, digits=3))
    





