import numpy as np
import os


class FullyConnectedLayers:
    def __init__(self) -> None:
        self.weights = []
        self.input_size = None
        pass

    def predict(self, input_vec):
        if not self.weights:
            return input_vec
        current_input = input_vec
        for w in self.weights:
            if w.shape[1] != current_input.shape[0]:
                print("Couldn't do this operation: dimensions mismatch")
                return 0
            current_input = w @ current_input
        return current_input

    def load_weights(self, file_name):
        data = np.load(file_name)
        self.weights = [data[key] for key in data.files]
        self.input_size = self.weights[0].shape[1] if self.weights else None

    def save_weights(self, file_name):
        np.savez(file_name, *self.weights)

    def add_layer(self, n, weight_range=(-1.0, 1.0)):
        if n <= 0:
            raise ValueError("n must be positive")

        if self.input_size is None:
            self.input_size = n
            return

        n_prev = self.weights[-1].shape[0] if self.weights else self.input_size
        low, high = weight_range
        self.weights.append(np.random.uniform(low, high, size=(n, n_prev)))


def test_add_layer():
    net = FullyConnectedLayers()
    net.add_layer(4)
    assert net.input_size == 4
    assert net.weights == []

    net.add_layer(16)
    net.add_layer(8)
    net.add_layer(1)
    assert [w.shape for w in net.weights] == [(16, 4), (8, 16), (1, 8)]

    net = FullyConnectedLayers()
    net.add_layer(3)
    net.add_layer(5, weight_range=(0.0, 0.5))
    assert net.weights[0].min() >= 0.0 and net.weights[0].max() <= 0.5

    try:
        FullyConnectedLayers().add_layer(0)
        assert False, "ожидалась ошибка при n=0"
    except ValueError:
        pass

    print("add_layer: OK")


def test_predict():
    net = FullyConnectedLayers()
    net.add_layer(4)
    net.add_layer(3)
    net.add_layer(2)

    x = np.random.rand(4, 1)
    result = net.predict(x)
    expected = net.weights[1] @ (net.weights[0] @ x)
    assert result.shape == (2, 1)
    assert np.allclose(result, expected)

    assert net.predict(np.random.rand(4)).shape == (2,)

    assert np.array_equal(FullyConnectedLayers().predict(x), x)

    print("predict: OK")


def test_save_load():
    net = FullyConnectedLayers()
    net.add_layer(4)
    net.add_layer(3)
    net.add_layer(2)

    file_name = "test_model.npz"
    net.save_weights(file_name)

    loaded = FullyConnectedLayers()
    loaded.load_weights(file_name)

    assert len(loaded.weights) == len(net.weights)
    for a, b in zip(net.weights, loaded.weights):
        assert np.array_equal(a, b)
    assert loaded.input_size == 4

    x = np.random.rand(4, 1)
    assert np.allclose(net.predict(x), loaded.predict(x))

    loaded.add_layer(5)
    assert loaded.weights[-1].shape == (5, 2)

    os.remove(file_name)
    print("save/load: OK")


if __name__ == "__main__":
    test_add_layer()
    test_predict()
    test_save_load()
    print("All tests were passed")
