try:
    from neuralconnect4 import NeuralConnect4
except ModuleNotFoundError:
    from .neuralconnect4 import NeuralConnect4


class _7L2DDrop40(NeuralConnect4):
    @property
    def _modelName(self) -> str:
        return "C4_7L2DDrop40.h5"

    @property
    def _batchSize(self) -> int:
        return 1
