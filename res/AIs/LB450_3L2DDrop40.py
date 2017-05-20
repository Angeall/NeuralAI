try:
    from neurallazerbike import NeuralLazerbike
except ModuleNotFoundError:
    from .neurallazerbike import NeuralLazerbike


class LB450_3L2DDrop40(NeuralLazerbike):

    @property
    def _modelName(self) -> str:
        return "LB450_3L2DDrop40.h5"

    @property
    def _batchSize(self) -> int:
        return 1
