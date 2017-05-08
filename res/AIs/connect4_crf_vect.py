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


class Connect4CRFVect(Connect4BotPlayer, CRFBot):
    def _encode(self, lst, nb_player: int):
        encoded_seqs = []
        for features in lst:
            encoded_seq = [0 for i in range(16)]
            encoded_seq[int(features[0] + 1)] += 1
            encoded_seq[int(features[1] + 1) + 8] += 1
            encoded_seqs.append(encoded_seq)
        return np.array(encoded_seqs)

    def _getPrediction(self, seq: List[List[int]]):
        seq_dicts = [{'col' + str(j): seq[i][j] for j in range(16)} for i in range(len(seq))]
        return self._model.tag(seq_dicts)

    @property
    def _modelName(self) -> str:
        return "c4crf_vect.bin"

    def _selectNewMove(self, game_state: Connect4API):
        winning_move = game_state.getDirectWinningMove(self.playerNumber)
        if winning_move is not None:
            return winning_move
        losing_move = game_state.getDirectLosingMove(self.playerNumber)
        if losing_move is not None:
            return losing_move  # Block the opponent
        move = super()._selectNewMove(game_state)
        succeeded, _ = game_state.simulateMove(self.playerNumber, move)
        while not succeeded:
            move = random.choice(self.possibleMoves)
            succeeded, _ = game_state.simulateMove(self.playerNumber, move)
        return move

