import random

import numpy as np

from pytgf.examples.connect4.controllers import Connect4BotPlayer
from pytgf.examples.connect4.rules import Connect4API

try:
    from hmmbot import HMMBot
except ModuleNotFoundError:
    from .hmmbot import HMMBot


class Connect4HMM(Connect4BotPlayer, HMMBot):
    @property
    def _modelName(self) -> str:
        return "c4hmm.bin"

    def _encodeToOneHot(self, sequence: list) -> np.ndarray:
        encoded_sequence_x = []
        for features in sequence:
            features = [features[0] + 1, features[1] + 1]
            feature_idx = (features[0] * 8) + features[1]
            encoded_features = [1 if j == feature_idx else 0 for j in range(64)]
            encoded_sequence_x.append(encoded_features)
        return np.array(encoded_sequence_x)

    def _selectNewMove(self, game_state: Connect4API):
        winning_move = game_state.getDirectWinningMove(self.playerNumber)
        if winning_move is not None:
            return winning_move
        losing_move = game_state.getDirectLosingMove(self.playerNumber)
        if losing_move is not None:
            return losing_move  # Block the opponent
        move = super()._selectNewMove(game_state)
        succeeded, _ = self.gameState.simulateMove(self.playerNumber, move)
        while not succeeded:
            move = random.choice(self.possibleMoves)
            succeeded, _ = self.gameState.simulateMove(self.playerNumber, move)
        return move
