import os
import random
from abc import ABCMeta, abstractmethod
from typing import List
from pickle import load

from pytgf.characters.moves import MoveDescriptor
from pytgf.data.decode.action_sequence_decoder import ActionSequenceDecoder
from pytgf.controls.controllers import Bot
from pytgf.game import API
import numpy as np


class HMMBot(Bot, metaclass=ABCMeta):
    def __init__(self, player_number: int):
        super().__init__(player_number)
        self._model = None

    def getReady(self):
        self._model = self._loadModel()
        print("loaded")

    def _selectNewMove(self, game_state: API) -> MoveDescriptor:
        actions = np.array(game_state.getAllActionsHistories()).tolist()
        actions = ActionSequenceDecoder.getPlayersActionsSequence(actions)
        if len(actions) == 0:
            return random.choice(self.possibleMoves)
        actions_sequence = self._predict(actions, len(game_state.getPlayerNumbers()))
        action = actions_sequence[-1]
        succeeded, _ = self.gameState.simulateMove(self.playerNumber, action)
        while not succeeded:
            action = random.choice(self.possibleMoves)
            succeeded, _ = self.gameState.simulateMove(self.playerNumber, action)
        return game_state.decodeMove(self.playerNumber, action)

    def _loadModel(self):
        path = os.path.join(self._modelPath, self._modelName)
        file = open(path, "br")
        return load(file)

    def _decode(self, lst):
        return lst.astype(int)

    def _encode(self, lst, nb_player: int):
        ar = self._encodeToOneHot(lst)
        return ar

    def _predict(self, lst: List[List[int]], nb_player: int):
        tab = self._encode(lst, nb_player)
        return self._decode(self._model.predict(tab, [len(tab)])).reshape(1, -1).tolist()[0]

    @property
    def _modelPath(self) -> str:
        return os.path.join("res", "AIs")

    @property
    @abstractmethod
    def _modelName(self) -> str:
        pass

    @abstractmethod
    def _encodeToOneHot(self, sequence: list) -> np.ndarray:  # int => float
        pass