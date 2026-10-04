import numpy as np


def neuron(input, weights, bias):
    res = np.dot(input, weights) + bias
    return res


def neural_network(input_vec, weights):
    vec_len = len(input_vec)
    weights_rows, weights_cols = weights.shape
    if vec_len != weights_cols:
        print("Couldn't do this operation: dimensions mismatch")
        return 0
    res = np.zeros(weights_rows)
    for i in range(weights_rows):
        res[i] = neuron(input_vec, weights[i], 0)
    return res


def deep_neural_network(input_vec, weights):
    current_input = input_vec
    for w in weights:
        current_input = neural_network(current_input, w)
    return current_input


def main():
    # tests for task 2
    input = np.array([0.5, 0.75, 0.1])
    weights = np.array(
        [
            [0.1, 0.1, -0.3],
            [0.1, 0.2, 0.0],
            [0.0, 0.7, 0.1],
            [0.2, 0.4, 0.0],
            [-0.3, 0.5, 0.1],
        ]
    )
    print(neural_network(input, weights))
    # tests for task 3
    Wh = np.array(
        [
            [0.1, 0.1, -0.3],
            [0.1, 0.2, 0.0],
            [0.0, 0.7, 0.1],
            [0.2, 0.4, 0.0],
            [-0.3, 0.5, 0.1],
        ]
    )
    Wy = np.array(
        [
            [0.7, 0.9, -0.4, 0.8, 0.1],
            [0.8, 0.5, 0.3, 0.1, 0.0],
            [-0.3, 0.9, 0.3, 0.1, -0.2],
        ]
    )
    W = [Wh, Wy]
    print(deep_neural_network(input, W))


if __name__ == "__main__":
    main()
