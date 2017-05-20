import random

import numpy as np
from pytgf.characters.moves import MoveDescriptor
from pytgf.controls.controllers import TeammateMessage
from pytgf.examples.lazerbike.controllers import LazerBikeBotPlayer
from pytgf.examples.lazerbike.rules import LazerBikeAPI

try:
    from hmmbot import HMMBot
except ModuleNotFoundError:
    from .hmmbot import HMMBot


class LazerBikeHMM(LazerBikeBotPlayer, HMMBot):
    @property
    def _modelName(self) -> str:
        return "lbhmm_player1.bin"

    def _encodeToOneHot(self, sequence: list) -> np.ndarray:
        encoded_sequence_x = []
        for features in sequence:
            features = [features[0] + 2, features[1] + 2]
            feature_idx = (features[0] * 4) + features[1]
            encoded_features = [1 if j == feature_idx else 0 for j in range(16)]
            encoded_sequence_x.append(encoded_features)
        return np.array(encoded_sequence_x)

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
