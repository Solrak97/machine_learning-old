import pandas as pd
import numpy as np
from utils import *


def make_splits(X: pd.DataFrame):
    splits = []
    for feature in X:
        type = get_type(X[feature])
        if type == FeatureType.NUMERICAL:
            splits.append((feature, numerical_split(X, feature), "numerical"))
        elif type == FeatureType.CATEGORICAL:
            splits.append((feature, categorical_split(X, feature), "categorical"))
    return splits


def select_childs(X: pd.DataFrame, Y:pd.Series):
    feature_splits = make_splits(X)
    gini = Gini(Y)
    split_feature, split_value = None, None
    L, R  = None, None
    best_infoscore = None
    split_type = None
    #   Esta parte devuelve una tupla(feature, splits)
    for fsplit in feature_splits:
        feature, splits, type = fsplit
    
    #   Los splits estan compuestos por una tupla de tipo (left split, right split, value)
    #   BTW los left and right splits son indices, hay que jalarlos con un loc cuando parte
        for split in splits:
            l, r, v = split

            split_score = gini - Gini_splits([Y.loc[l], Y.loc[r]])

            if(best_infoscore is None or best_infoscore <= split_score):
                best_infoscore = split_score
                L = l
                R = r
                split_feature = feature
                split_value = v
                split_type = type

    #   Si la indentacion de python no me traiciona, a este punto debería saber los indices para l, r, valor y feature
    return((L, R, split_feature, split_value, split_type, gini, best_infoscore))


def Gini(y: pd.Series) -> float:    
    #   Yo pensaba que con el type hint era suiciente, pls type enforcement
    p = y.value_counts()/y.shape[0]
    return (1 - np.sum(p**2))


def Gini_splits(ys: list[pd.Series]) -> float:
    gini_scores = [Gini(y) for y in ys]
    n_i = [y.shape[0] for y in ys]
    n = np.sum(n_i)
    funny_number = n_i / n
    funny_scores = gini_scores * funny_number
    gs = np.sum(funny_scores)
    return gs


class Node:
    def __init__(self, X, Y, depth: int): 
        self.type = None
        self.gini = None
        self.count = Y.shape[0]
        self.split_type = None
        self.split_column = None
        self.split_value = None
        self.child_left = None
        self.child_right = None
        self._class = None

        # Caso base, solo una clase
        if Y.unique().shape[0] == 1:
            self.type = "leaf"
            self._class = Y.unique()[0]
        
        else:
            L, R, split_feature, split_value, split_type, gini, best_infoscore = select_childs(X, Y)
            
            # Caso no en max-deepht pero clase no determinada
            if best_infoscore == 0 or depth <= 0:
                self.type = "leaf"
                self._class = Y.mode()[0]

            # Caso donde profundidad aun no es alcanzada
            else:   
                self.type = "split"
                self.split_column = split_feature
                self.split_value = split_value
                self.split_type = split_type
                self.gini = gini
                self.child_left = Node(X.loc[L], Y.loc[L], depth - 1)
                self.child_right = Node(X.loc[R], Y.loc[R], depth - 1)
        

class DecisionTree:

    def __init__(self):
        self.tree = None
        pass


    def fit(self, X, Y, max_depth = None):
        self.tree = Node(X, Y, max_depth)
        pass


# Indice!!! por que???
    def get_class(self, root, row):
        while (root.type != "leaf"):
            if(root.split_type == "categorical"):
                if(row[root.split_column] == root.split_value):
                    root = root.child_left 
                else:
                    root = root.child_right

            elif(root.split_type == "numerical"):
                if(row[root.split_column] <= root.split_value):
                    root = root.child_left 
                else:
                    root = root.child_right
        
        return root._class


    def predict(self, X: pd.DataFrame):
        predicts = []
        for index, row in X.iterrows():
            _class = self.get_class(self.tree, row)
            predicts.append(_class)
        return pd.Series(predicts)


    def recursive_dict(self, node):
        
        node_data = {}
        
        node_data["type"] = node.type
        node_data["count"] = node.count

        if node.type != "leaf":
            node_data["gini"] = node.gini
            node_data["split_type"] = node.split_type
            node_data["split_column"] = node.split_column
            node_data["split_value"] = node.split_value
            node_data["child_left"] = self.recursive_dict(node.child_left)
            node_data["child_right"] = self.recursive_dict(node.child_right) 
        
        else:
            node_data["class"] = node._class
        
        return node_data
        
        


    def to_dict(self):
        return self.recursive_dict(self.tree)
        


def calculate_confusion_matrix(predict, real):
    pclass = set(real.unique()).union(predict.unique())
    matrix = {}

    for r in pclass:
        val = {}
        for p in pclass:
            val[p] = 0
        matrix[r] = val
          
    for p, r in zip(predict, real):
        matrix[r][p] = matrix[r][p] + 1
    return matrix