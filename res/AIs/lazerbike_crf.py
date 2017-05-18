import random
from typing import List

from pytgf.examples.lazerbike.control import LazerBikeBotPlayer
from pytgf.examples.lazerbike.rules import LazerBikeAPI

from pytgf.game import API

try:
    from crfbot import CRFBot
except ModuleNotFoundError:
    from .crfbot import CRFBot


class LazerBikeCRF(LazerBikeBotPlayer, CRFBot):
    def _getPrediction(self, seq: List[List[int]]):
        seq_dicts = [{'pl0': seq[i][0] + 1, 'pl1': seq[i][1] + 1} for i in range(len(seq))]
        return self._model.tag(seq_dicts)

    @property
    def _modelName(self) -> str:
        return "lbcrf.bin"

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

