from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import imageio

data = np.load("data.npy")

print("Data shape: ", data.shape)