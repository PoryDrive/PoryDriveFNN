# James William Fletcher - May 2022
# https://github.com/mrbid/porydrivekeras
import sys
import os
import numpy as np
from time import time_ns
from os import mkdir
from os.path import isdir

# print everything / no truncations
np.set_printoptions(threshold=sys.maxsize)

# training set size
inputsize = 6
outputsize = 2
tss = int(os.stat("dataset_y.dat").st_size / 8)
print("Dataset Size:", "{:,}".format(tss))

# helpers (https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison)
def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

##########################################
#   SHUFFLE
##########################################
st = time_ns()

# load training data
train_x = []
train_y = []

print("Loading & Reshaping...")
with open("dataset_x.dat", 'rb') as f:
    data = np.fromfile(f, dtype=np.float32)
    train_x = np.reshape(data, [tss, inputsize])

with open("dataset_y.dat", 'rb') as f:
    data = np.fromfile(f, dtype=np.float32)
    train_y = np.reshape(data, [tss, outputsize])

print("Shuffling...")
shuffle_in_unison(train_x, train_y)

print("Saving...")
np.save("numpy_x", train_x)
np.save("numpy_y", train_y)

timetaken = (time_ns()-st)/1e+9
print("Time Taken:", "{:.2f}".format(timetaken), "seconds")
