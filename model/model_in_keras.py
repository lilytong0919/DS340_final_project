# try making model in kera

from tensorflow import keras

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.005
EPOCH = 1000
EPS = 1
GAMMA = 0.99


def create_model(input_layer,output_layer):
    inputs = layers.Input(shape=(6,1))
    