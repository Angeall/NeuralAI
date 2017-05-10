import random
from abc import ABCMeta
from os.path import join

import numpy as np
from pytgf.characters.moves import MoveDescriptor
from pytgf.controls.controllers import TeammateMessage
from pytgf.examples.lazerbike.control import LazerBikeBotPlayer
from pytgf.examples.lazerbike.rules import LazerBikeAPI

try:
    from neuralnetworkbot import NeuralNetworkBot
except ModuleNotFoundError:
    from .neuralnetworkbot import NeuralNetworkBot


class NeuralLazerbike(LazerBikeBotPlayer, NeuralNetworkBot, metaclass=ABCMeta):
    def _encodeToFloat(self, sequence: np.ndarray) -> np.ndarray:
        return (sequence + 2) / 3

    def _decodeFromFloat(self, sequence: np.ndarray) -> np.ndarray:
        return np.round((sequence * 3) - 2).astype(int)

    def _selectNewMove(self, game_state: LazerBikeAPI):
        move = super()._selectNewMove(game_state)
        succeeded, _ = game_state.simulateMove(self.playerNumber, move)
        suicidal_moves = []
        suicidal = game_state.isMoveSuicidal(self.playerNumber, move)
        while not succeeded or (suicidal and len(suicidal_moves) < 3):
            move = random.choice(self.possibleMoves)
            succeeded, _ = game_state.simulateMove(self.playerNumber, move)
            suicidal = game_state.isMoveSuicidal(self.playerNumber, move)
            if succeeded and suicidal and move not in suicidal_moves:
                suicidal_moves.append(move)
        return move

    def _isMoveInteresting(self, player_number: int, new_move_event: MoveDescriptor):
        return True

    def selectMoveFollowingTeammateMessage(self, teammate_number: int, message: TeammateMessage) -> None:
        pass

    @property
    def _maxSequenceLength(self) -> int:
        return 112  # 113 actions maximum - 1 because of prediction

    @property
    def _neutralValue(self):
        return -2
