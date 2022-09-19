from utils import load_image
from lloyd import lloyd
from matplotlib import pyplot as plt

def find_palette(img, type, distance):
    k_res = lloyd(img, 5, 200, type, distance)
    return k_res
