import cPickle as pickle
import numpy as np
import mmh3


def hash(name, mapping_size):
    mat = np.zeros((1, mapping_size))
    tokens = name.split()
    arr = [mmh3.hash(token) % mapping_size for token in tokens]
    for hash in arr:
        mat[0, hash] = 1
    return mat

model = pickle.load(open("model", "rb"))
while True:
    name = raw_input("Enter string: ")
    name_hash = hash(name, 200000)
    print(model.predict(name_hash))
