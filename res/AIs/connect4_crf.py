import random
from typing import List

import numpy as np

from pytgf.examples.connect4.controllers import Connect4BotPlayer
from pytgf.examples.connect4.rules import Connect4API
from pytgf.game import API

try:
    from crfbot import CRFBot
except ModuleNotFoundError:
    from .crfbot import CRFBot


class Connect4CRF(Connect4BotPlayer, CRFBot):
    def _getPrediction(self, seq: List[List[int]]):
        seq_dicts = [{'pl0': seq[i][0] + 1, 'pl1': seq[i][1] + 1} for i in range(len(seq))]
        return self._model.tag(seq_dicts)

    @property
    def _modelName(self) -> str:
        return "c4crf.bin"

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

