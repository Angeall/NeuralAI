import random
from abc import ABCMeta
from os.path import join

import numpy as np
from pytgf.examples.lazerbike.control import LazerBikeBotPlayer
from pytgf.examples.lazerbike.rules import LazerBikeAPI

try:
    from neuralnetworkbot import NeuralNetworkBot
except ModuleNotFoundError:
    from .neuralnetworkbot import NeuralNetworkBot


class NeuralLazerbike(LazerBikeBotPlayer, NeuralNetworkBot, metaclass=ABCMeta):
    def _encodeToFloat(self, sequence: np.ndarray) -> np.ndarray:
        return (sequence + 2) * 3

    def _decodeFromFloat(self, sequence: np.ndarray) -> np.ndarray:
        return np.round((sequence / 3) - 2).astype(int)

    def _selectNewMove(self, game_state: LazerBikeAPI):
        move = super()._selectNewMove(game_state)
        succeeded, _ = game_state.simulateMove(self.playerNumber, move)
        while not succeeded or game_state.isMoveSuicidal(self.playerNumber, move):
            move = random.choice(self.possibleMoves)
            succeeded, _ = game_state.simulateMove(self.playerNumber, move)
        return move

    @property
    def _maxSequenceLength(self) -> int:
        return 112  # 113 actions maximum - 1 because of prediction

    @property
    def _neutralValue(self):
        return -2
