try:
    from neuralconnect4 import NeuralConnect4
except ModuleNotFoundError:
    from .neuralconnect4 import NeuralConnect4


class _3L2DNoDrop(NeuralConnect4):
    @property
    def _modelName(self) -> str:
        return "C4_1L1Dnodrop.h5"

    @property
    def _batchSize(self) -> int:
        return 20
