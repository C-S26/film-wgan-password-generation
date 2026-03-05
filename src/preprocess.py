import numpy as np
import unicodedata
import string

MAX_LEN = 16

# 94 printable ASCII characters
VOCAB = list(string.printable[:-6])
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(VOCAB)}
PAD_TOKEN = 0


def normalize_password(p):
    p = unicodedata.normalize("NFKD", p)
    return "".join([c for c in p if c in CHAR_TO_IDX])


def encode_password(p):
    encoded = [CHAR_TO_IDX[c] for c in p[:MAX_LEN]]
    if len(encoded) < MAX_LEN:
        encoded += [PAD_TOKEN] * (MAX_LEN - len(encoded))
    return encoded


def preprocess_file(input_file):
    data = []

    with open(input_file, "r", encoding="latin-1") as f:
        for line in f:
            pwd = normalize_password(line.strip())
            if pwd:
                data.append(encode_password(pwd))

    data = np.array(data, dtype=np.int32)

    np.save("train_data.npy", data)

    with open("chars.txt", "w") as f:
        f.write("".join(VOCAB))

    print("Saved dataset:", data.shape)


if __name__ == "__main__":
    preprocess_file("passwords.txt")
