# James William Fletcher - April 2022
# https://github.com/mrbid/porydrivekeras
import sys
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from random import seed
from time import time_ns
from sys import exit
from os.path import isfile
from os import mkdir
from os.path import isdir

# import tensorflow as tf
# from tensorflow.python.client import device_lib
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# if tf.test.gpu_device_name():
#     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
# else:
#     print("Please install GPU version of TF")
# print(device_lib.list_local_devices())
# print(tf.config.list_physical_devices())
# exit();

# print everything / no truncations
np.set_printoptions(threshold=sys.maxsize)

# hyperparameters
seed(74035)
model_name = 'keras_model'
optimiser = 'adam'
inputsize = 6
outputsize = 2
training_iterations = 1
activator = 'tanh'
layers = 4
layer_units = 384
batches = 32
# layer_units = 1024
# batches = 64

# load options
argc = len(sys.argv)
print(argc)
if argc >= 2:
    layers = int(sys.argv[1])
    print("layers:", layers)
if argc >= 3:
    layer_units = int(sys.argv[2])
    print("layer_units:", layer_units)
if argc >= 4:
    batches = int(sys.argv[3])
    print("batches:", batches)
if argc >= 5:
    optimiser = sys.argv[4]
    print("optimiser:", optimiser)
if argc >= 6 and sys.argv[5] == '1':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print("CPU_ONLY: 1")

if not isdir('models'): mkdir('models')
model_name = 'models/' + optimiser + '_' + sys.argv[1] + '_' + sys.argv[2] + '_' + sys.argv[3]
print("model_name:", model_name)

# training set size
tss = int(os.stat("dataset_y.dat").st_size / 8)
print("Dataset Size:", "{:,}".format(tss))

# helpers (https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison)
def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

##########################################
#   LOAD DATA
##########################################
st = time_ns()

# load training data
train_x = []
train_y = []

# with open("dataset_x.dat", 'rb') as f:
#     data = np.fromfile(f, dtype=np.float32)
#     train_x = np.reshape(data, [tss, inputsize])

# with open("dataset_y.dat", 'rb') as f:
#     data = np.fromfile(f, dtype=np.float32)
#     train_y = np.reshape(data, [tss, outputsize])

# shuffle_in_unison(train_x, train_y)

if isfile("numpy_x"):
    train_x = np.load("numpy_x")
    train_y = np.load("numpy_y")
    print("Loaded shuffled numpy arrays")
else:
    with open("dataset_x.dat", 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
        train_x = np.reshape(data, [tss, inputsize])

    with open("dataset_y.dat", 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
        train_y = np.reshape(data, [tss, outputsize])

    print("Loaded regular arrays; no shuffle")

    # shuffle_in_unison(train_x, train_y)
    
    # np.save("numpy_x", train_x)
    # np.save("numpy_y", train_y)

# print(train_x.shape)
# print(train_x)
# print(train_y.shape)
# print(train_y)
# exit()

timetaken = (time_ns()-st)/1e+9
print("Time Taken:", "{:.2f}".format(timetaken), "seconds")

##########################################
#   TRAIN
##########################################

# construct neural network
model = Sequential()

model.add(Dense(layer_units, activation=activator, input_dim=inputsize))
if layers > 0: model.add(Dense(layer_units/2, activation=activator))
if layers > 1: model.add(Dense(layer_units/4, activation=activator))
if layers > 2: model.add(Dense(layer_units/8, activation=activator))
if layers > 3: model.add(Dense(layer_units/16, activation=activator))
# model.add(Dropout(.2))
model.add(Dense(outputsize, activation='tanh'))

# output summary
model.summary()

if optimiser == 'adam':
    optim = keras.optimizers.Adam(lr=0.001)
elif optimiser == 'sgd':
    optim = keras.optimizers.SGD(lr=0.01, momentum=0., decay=0., nesterov=False)
elif optimiser == 'momentum':
    optim = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0., nesterov=False)
elif optimiser == 'nesterov':
    optim = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0., nesterov=True)
elif optimiser == 'nadam':
    optim = keras.optimizers.Nadam(learning_rate=0.001)
elif optimiser == 'adagrad':
    optim = keras.optimizers.Adagrad(learning_rate=0.001)
elif optimiser == 'rmsprop':
    optim = keras.optimizers.RMSprop(learning_rate=0.001)
elif optimiser == 'adadelta':
    optim = keras.optimizers.Adadelta(learning_rate=0.001)
elif optimiser == 'adamax':
    optim = keras.optimizers.Adamax(learning_rate=0.001)
elif optimiser == 'ftrl':
    optim = keras.optimizers.Ftrl(learning_rate=0.001)
model.compile(optimizer=optim, loss='mean_squared_error')

# train network
model.fit(train_x, train_y, epochs=training_iterations, batch_size=batches)
timetaken = (time_ns()-st)/1e+9
print("")
print("Time Taken:", "{:.2f}".format(timetaken), "seconds")

##########################################
#   EXPORT
##########################################

# save keras model
model.save(model_name)
