from PIL import Image
import numpy as np

def load_image(filename, resize):
    img = np.array(Image.open(filename).resize(resize)).astype(np.float32)
    return img.reshape((img.shape[0] * img.shape[1], img.shape[2])), img


def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)


def manhattan_distance(p1, p2):
    return np.sum(np.abs(p1 - p2))