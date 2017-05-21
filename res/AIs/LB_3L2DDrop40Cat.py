import numpy as np
try:
    from neurallazerbike import NeuralLazerbike
except ModuleNotFoundError:
    from .neurallazerbike import NeuralLazerbike


class LB_3L2DDrop40Cat(NeuralLazerbike):

    def _decodeFromFloat(self, sequence: np.ndarray) -> np.ndarray:
        actions = []
        sequence = sequence[0]
        for lst in sequence:
            actions.append(np.argmax(lst) - 2)
        return np.array([actions])

    @property
    def _modelName(self) -> str:
        return "LB_3L2DDrop40Cat.h5"

    @property
    def _batchSize(self) -> int:
        return 1
