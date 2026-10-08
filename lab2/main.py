import numpy as np


def train_step(weights, x, y, alpha):
    vec_len = len(x)
    _, weights_cols = weights.shape
    if vec_len != weights_cols:
        print("Couldn't do this operation: dimensions mismatch")
        return 0
    output = weights @ x
    N = len(y)
    error = np.sum((output - y) ** 2)
    delta = 2 * (1 / N) * np.outer((output - y), x)
    w = weights - delta * alpha
    return w, output, error


def training(weights, x, y, alpha):
    cur_w = weights
    cur_x = x
    for i in range(1, 21):
        cur_w, cur_x, e = train_step(cur_w, x, y, alpha)
        if i == 5 or i == 20:
            print(cur_x[0], e)
    return cur_x


def neural_network(weights, x, y, alpha):
    if len(x) != len(y):
        print("Len mismatch x and y")
        return 0
    cur_x = x
    cur_w = weights
    for j in range(1000):
        print("Epoch ", j)
        for i in range(len(x)):
            x_len = len(x[i])
            _, weights_cols = cur_w.shape
            if x_len != weights_cols:
                print("Couldn't do this operation: dimensions mismatch")
                return 0
            cur_w, cur_x, e = train_step(cur_w, x[i], y[i], alpha)
            print("Seria ", i + 1)
            print("Output:\n", cur_x)
            print("Weights:\n", cur_w)
            print("Error: ", e)
        print("-----------------------------------------------------")
    return cur_x


def main():
    weights = np.array([[0.5]])
    x1 = np.array([2.0])
    x2 = np.array([0.1])
    y = np.array([0.8])

    print("Task 1\ntest 1")
    training(weights, x1, y, 0.1)
    print("test 2")
    training(weights, x1, y, 1)
    print("test 3")
    training(weights, x2, y, 1)
    print("Task 2\ntest 1")

    x22 = np.array(
        [[0.5, 0.75, 0.1], [0.1, 0.3, 0.7], [0.2, 0.1, 0.6], [0.8, 0.9, 0.2]]
    )
    y22 = np.array(
        [
            [0.1, 1.0, 0.1, 0.0, -0.1],
            [0.5, 0.2, -0.5, 0.3, 0.7],
            [0.1, 0.3, 0.2, 0.9, 0.1],
            [0.7, 0.6, 0.2, -0.1, 0.8],
        ]
    )
    W22 = np.array(
        [
            [0.1, 0.1, -0.3],
            [0.1, 0.2, 0.0],
            [
                0.0,
                0.7,
                0.1,
            ],
            [0.2, 0.4, 0.0],
            [-0.3, 0.5, 0.1],
        ]
    )
    neural_network(W22, x22, y22, 0.01)
    return 0


if __name__ == "__main__":
    main()
