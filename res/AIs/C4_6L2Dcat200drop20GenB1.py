import numpy as np

try:
    from neuralconnect4 import NeuralConnect4
except ModuleNotFoundError:
    from .neuralconnect4 import NeuralConnect4


class _6L2Dcat200drop20GenB1(NeuralConnect4):

    def _decodeFromFloat(self, sequence: np.ndarray) -> np.ndarray:
        actions = []
        print("seq_before", sequence)
        sequence = sequence[0]
        print("seq_after", sequence)
        for lst in sequence:
            print("lst:", lst)
            maxi = np.argmax(lst) - 1
            print(maxi)
            actions.append(maxi)
        return np.array([actions])

    @property
    def _modelName(self) -> str:
        return "C4_6L2Dcat200drop20GenB1.h5"

    @property
    def _batchSize(self) -> int:
        return 1
