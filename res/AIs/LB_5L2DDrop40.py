try:
    from neurallazerbike import NeuralLazerbike
except ModuleNotFoundError:
    from .neurallazerbike import NeuralLazerbike


class LB_5L2DDrop40(NeuralLazerbike):

    @property
    def _modelName(self) -> str:
        return "LB_5L2DDrop40.h5"

    @property
    def _batchSize(self) -> int:
        return 1
