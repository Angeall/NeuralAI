import numpy as np

try:
    from neuralconnect4 import NeuralConnect4
except ModuleNotFoundError:
    from .neuralconnect4 import NeuralConnect4


class _6L2DCat200Drop20Gen(NeuralConnect4):

    def _decodeFromFloat(self, sequence: np.ndarray) -> np.ndarray:
        actions = []
        sequence = sequence[0]
        for lst in sequence:
            maxi = np.argmax(lst) - 1
            actions.append(maxi)
        return np.array([actions])

    @property
    def _modelName(self) -> str:
        return "C4_6L2DCat200Drop20Gen.h5"

    @property
    def _batchSize(self) -> int:
        return 20
