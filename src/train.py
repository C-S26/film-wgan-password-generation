import numpy as np
import tensorflow as tf
from model import build_generator

LATENT_DIM = 128
BATCH_SIZE = 96
EPOCHS = 60


def load_data():
    return np.load("train_data.npy")


def train():

    data = load_data()

    generator = build_generator()

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=2e-5,
        beta_1=0,
        beta_2=0.9
    )

    for epoch in range(EPOCHS):

        idx = np.random.randint(0, data.shape[0], BATCH_SIZE)
        real = data[idx]

        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
        cond = np.random.randint(0, 3, (BATCH_SIZE, 1))

        with tf.GradientTape() as tape:
            fake = generator([noise, cond])
            loss = tf.reduce_mean(fake)

        grads = tape.gradient(loss, generator.trainable_variables)
        optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        print("Epoch", epoch, "Loss:", float(loss))


if __name__ == "__main__":
    train()
