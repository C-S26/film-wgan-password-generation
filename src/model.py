import tensorflow as tf
from tensorflow.keras import layers

VOCAB_SIZE = 95
SEQ_LEN = 16
LATENT_DIM = 128


class FiLM(layers.Layer):

    def __init__(self, units):
        super().__init__()
        self.gamma = layers.Dense(units)
        self.beta = layers.Dense(units)

    def call(self, x, cond):
        g = self.gamma(cond)
        b = self.beta(cond)
        return g * x + b


def build_generator():

    noise = layers.Input(shape=(LATENT_DIM,))
    cond = layers.Input(shape=(1,))

    x = layers.Dense(256, activation="relu")(noise)

    film = FiLM(256)
    x = film(x, cond)

    x = layers.Dense(512, activation="relu")(x)

    film2 = FiLM(512)
    x = film2(x, cond)

    x = layers.Dense(SEQ_LEN * VOCAB_SIZE)(x)
    x = layers.Reshape((SEQ_LEN, VOCAB_SIZE))(x)

    output = layers.Softmax()(x)

    return tf.keras.Model([noise, cond], output)
