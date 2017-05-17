try:
    from neurallazerbike import NeuralLazerbike
except ModuleNotFoundError:
    from .neurallazerbike import NeuralLazerbike


class LB_3L2DDrop40B1(NeuralLazerbike):

    @property
    def _modelName(self) -> str:
        return "LB_3L2DDrop40B1.h5"

    @property
    def _batchSize(self) -> int:
        return 1
