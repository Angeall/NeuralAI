try:
    from neurallazerbike import NeuralLazerbike
except ModuleNotFoundError:
    from .neurallazerbike import NeuralLazerbike


class _3L2DNoDrop(NeuralLazerbike):
    @property
    def _modelName(self) -> str:
        return "C4_3L2Dnodrop.h5"

    @property
    def _batchSize(self) -> int:
        return 20
