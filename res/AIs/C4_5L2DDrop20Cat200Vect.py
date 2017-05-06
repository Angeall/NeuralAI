import numpy as np

try:
    from neuralconnect4 import NeuralConnect4
except ModuleNotFoundError:
    from .neuralconnect4 import NeuralConnect4


class _5L2DDrop20Cat200Vect(NeuralConnect4):

    def _encodeToFloat(self, sequence: np.ndarray) -> np.ndarray:
        sequence += 1
        encoded_seqs = []
        for features in sequence:
            encoded_seq = [0 for i in range(16)]
            encoded_seq[int(features[0])] += 1
            encoded_seq[int(features[1]) + 8] += 1
            encoded_seqs.append(encoded_seq)
        return np.array(encoded_seqs)

    def _decodeFromFloat(self, sequence: np.ndarray) -> np.ndarray:
        actions = []
        sequence = sequence[0]
        for lst in sequence:
            actions.append(np.argmax(lst) - 1)
        return np.array([actions])

    @property
    def _modelName(self) -> str:
        return "C4_5L2DDrop20Cat200Vect.h5"

    @property
    def _batchSize(self) -> int:
        return 1
