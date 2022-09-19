from utils import manhattan_distance, euclidean_distance
from matplotlib import pyplot as plt
import numpy as np
import random


def nearest_centroid(point, centroids, distance):
    dst_lst = [distance(point, x) for x in centroids]
    idx = np.argmin(np.array(dst_lst))
    return (idx, dst_lst[idx])



def lloyd(data, k, iters, _type, distance):
    
    if distance == "euclidean":
        distance = euclidean_distance
    elif distance == "manhattan":
        distance = manhattan_distance

    centroid_idx = random.sample(range(len(data)), k)
    centroids = [data[idx] for idx in centroid_idx]
    error = 0

    for it in range(iters):
        n_centroids = [[] for _ in range(k)]
        n_error = 0
        # Sort de los puntos en clusters
        for point in data:
            idx, dst = nearest_centroid(point, centroids, distance)
            n_centroids[idx].append(point)

        # Calculo del error y set centroid
        for idx, centroid in enumerate(n_centroids):
            if _type == "means":

                # Se calcula el promedio
                centroids[idx] = np.array(centroid).mean(axis=0)
                centroid_dst = 0

                # Se calcula el error
                for point in centroid:
                    centroid_dst += distance(point, centroids[idx])

                n_error += centroid_dst

            if _type == "medioids":

                # Se calcula el error inicial
                centroid_dst = np.sum(np.array(errors[idx]))

                # Se selecciona un random
                n_centroid = random.choice(centroid)
                new_dst = 0

                # Se calcula el error con el random
                for point in centroid:
                    new_dst += distance(point, n_centroid)

                # Si el nuevo error es menor, el random se acepta como centroide
                if new_dst < centroid_dst:
                    centroids[idx] = n_centroid
                    centroid_dst = new_dst

                n_error += centroid_dst

        error = n_error

    return {"Centroids":centroids, "Error": error}