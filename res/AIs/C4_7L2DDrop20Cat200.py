import numpy as np

try:
    from neuralconnect4 import NeuralConnect4
except ModuleNotFoundError:
    from .neuralconnect4 import NeuralConnect4


class _7L2DDrop20Cat200(NeuralConnect4):

    def _decodeFromFloat(self, sequence: np.ndarray) -> np.ndarray:
        actions = []
        sequence = sequence[0]
        for lst in sequence:
            actions.append(np.argmax(lst) - 1)
        return np.array([actions])

    @property
    def _modelName(self) -> str:
        return "C4_7L2DDrop20Cat200.h5"

    @property
    def _batchSize(self) -> int:
        return 1
