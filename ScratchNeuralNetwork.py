import random

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def linear(x, w, b):
    return np.dot(x, w) + b

#generating dataset
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

w_out = np.random.randn(4)
b_out = random.random()

def deriv_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

#binary cross entropy (loss function)
def BCE(prediction, label):
    return -(label * np.log(prediction) + (1 - label) * np.log(1 - prediction))

alpha = 0.01
n_epochs = 100 #for quick results

for epoch in range(n_epochs):

    epoch_loss = 0

    for x, label in zip(X, y):

        Z_1 = linear(x, w1, b1)
        Z_2 = linear(x, w2, b2)
        Z_3 = linear(x, w3, b3)
        Z_4 = linear(x, w4, b4)

        pre_neurons = [Z_1, Z_2, Z_3, Z_4]

        N_1 = sigmoid(Z_1)
        N_2 = sigmoid(Z_2)
        N_3 = sigmoid(Z_3)
        N_4 = sigmoid(Z_4)

        O_1 = np.array([N_1, N_2, N_3, N_4])
        output = linear(O_1, w_out, b_out)
        result = sigmoid(output)

        dz_out = result - label
        dw_out = dz_out * O_1
        db_out = dz_out

        dO_1 = dz_out * w_out
        dw_list = []
        db_list = []
        for Z_i, dO_i, in zip(pre_neurons, dO_1):
            dz_i = dO_i * deriv_sigmoid(Z_i)
            dw_i = dz_i * x
            db_i = dz_i
            dw_list.append(dw_i)
            db_list.append(db_i)

        #updating weights
        w1 = w1 - alpha * dw_list[0]
        w2 = w2 - alpha * dw_list[1]
        w3 = w3 - alpha * dw_list[2]
        w4 = w4 - alpha * dw_list[3]
        w_out = w_out - alpha * (dw_out)
        #updating biases
        b1 = b1 - alpha * db_list[0]
        b2 = b2 - alpha * db_list[1]
        b3 = b3 - alpha * db_list[2]
        b4 = b4 - alpha * db_list[3]
        b_out = b_out - alpha * (db_out)

        epoch_loss += BCE(result, label)

    avg_loss = epoch_loss / len(X)
    print(f"Epoch #{epoch}, Avg Loss: {avg_loss}")



#visualizing classifications
#VIBECODED SECTION
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

def forward(X_batch, w1, b1, w2, b2, w3, b3, w4, b4, w_out, b_out):
    """Vectorized forward pass over a batch of samples, shape (N, 2)."""
    Z1 = X_batch @ w1 + b1
    Z2 = X_batch @ w2 + b2
    Z3 = X_batch @ w3 + b3
    Z4 = X_batch @ w4 + b4

    N1, N2, N3, N4 = sigmoid(Z1), sigmoid(Z2), sigmoid(Z3), sigmoid(Z4)

    hidden = np.stack([N1, N2, N3, N4], axis=-1)  # shape (N, 4)
    output = hidden @ w_out + b_out               # shape (N,)
    return sigmoid(output)


def visualize_classification(X_data, y_data, w1, b1, w2, b2, w3, b3, w4, b4, w_out, b_out):
    y_data = y_data.flatten()
    data_0 = X_data[y_data == 0]
    data_1 = X_data[y_data == 1]

    fig = plt.figure(figsize=(4, 4), dpi=150)
    plt.scatter(data_0[:, 0], data_0[:, 1], edgecolor="#333", label="Class 0")
    plt.scatter(data_1[:, 0], data_1[:, 1], edgecolor="#333", label="Class 1")
    plt.title("Dataset samples")
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.legend()

    c0 = np.array(to_rgba("C0"))
    c1 = np.array(to_rgba("C1"))

    x1 = np.arange(-0.5, 1.5, step=0.01)
    x2 = np.arange(-0.5, 1.5, step=0.01)
    xx1, xx2 = np.meshgrid(x1, x2, indexing='ij')

    grid_inputs = np.stack([xx1.ravel(), xx2.ravel()], axis=-1)  # shape (N, 2)

    preds = forward(grid_inputs, w1, b1, w2, b2, w3, b3, w4, b4, w_out, b_out)
    preds = preds.reshape(xx1.shape)  # back to grid shape

    output_image = (1 - preds[..., None]) * c0[None, None] + preds[..., None] * c1[None, None]

    plt.imshow(output_image, origin='lower', extent=(-0.5, 1.5, -0.5, 1.5))
    plt.grid(False)
    return fig


_ = visualize_classification(X, y, w1, b1, w2, b2, w3, b3, w4, b4, w_out, b_out)
plt.show()
#END VIBECODED SECTION
