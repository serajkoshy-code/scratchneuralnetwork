import random

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def linear(x, w, b):
    return np.dot(x, w) + b

#vibecoded dataset
#VIBECODED SECTION
def generate_xor_dataset(n_per_cluster=100, noise=0.08, random_state=42):
    rng = np.random.RandomState(random_state)

    # cluster centers correspond to XOR truth table corners
    centers = np.array([
        [0, 0],  # class 0
        [1, 1],  # class 0
        [0, 1],  # class 1
        [1, 0],  # class 1
    ])
    labels = np.array([0, 0, 1, 1])

    X_list = []
    y_list = []

    for center, label in zip(centers, labels):
        points = center + noise * rng.randn(n_per_cluster, 2)
        X_list.append(points)
        y_list.append(np.full(n_per_cluster, label))

    X = np.vstack(X_list)
    y = np.concatenate(y_list).reshape(-1, 1)

    # shuffle so classes aren't grouped in order
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    return X, y

X, y = generate_xor_dataset(n_per_cluster=100, noise=0.08)
#END VIBECODED SECTION

b1, b2, b3, b4 = (random.random() for _ in range(4))
w1, w2, w3, w4 = (np.random.randn(2) for _ in range(4))

for x, label in zip(X, y):

    N_1 = sigmoid(linear(x, w1, b1))
    N_2 = sigmoid(linear(x, w2, b2))
    N_3 = sigmoid(linear(x, w3, b3))
    N_4 = sigmoid(linear(x, w4, b4))

    result = sigmoid(N_1 + N_2 + N_3 + N_4)