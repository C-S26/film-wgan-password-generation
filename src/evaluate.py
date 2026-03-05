import numpy as np


def uniqueness(samples):
    unique = len(set(samples))
    return unique / len(samples)


def evaluate():

    data = np.load("generated_samples.npy")

    uniq = uniqueness([tuple(x) for x in data])

    print("Uniqueness:", uniq)


if __name__ == "__main__":
    evaluate()
