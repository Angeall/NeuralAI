import os
import random
from abc import ABCMeta, abstractmethod
from typing import List

from pytgf.characters.moves import MoveDescriptor
from pytgf.data.decode.action_sequence_decoder import ActionSequenceDecoder
from pytgf.controls.controllers import Bot
from pytgf.game import API, Core
from keras.models import load_model
import numpy as np


class NeuralNetworkBot(Bot, metaclass=ABCMeta):
    def __init__(self, player_number: int):
        super().__init__(player_number)
        self._model = None

    def getReady(self):
        self._model = self._loadModel()
        self._predict([], len(self.gameState.getPlayerNumbers()))

    def _selectNewMove(self, game_state: API) -> MoveDescriptor:
        actions = np.array(game_state.getAllActionsHistories()).tolist()
        actions = ActionSequenceDecoder.getPlayersActionsSequence(actions)
        if len(actions) == 0:
            return random.choice(self.possibleMoves)
        actions_sequence = self._predict(actions, len(game_state.getPlayerNumbers()))
        selected_action = None
        for i, action in enumerate(actions_sequence):
            if action >= 0:
                selected_action = action
                if i >= len(actions):
                    break
        if selected_action is None:
            return random.choice(self.possibleMoves)  # No actions selected by network
        return game_state.decodeMove(self.playerNumber, selected_action)

    def _loadModel(self):
        path = os.path.join(self._modelPath, self._modelName)
        model = load_model(path)
        return model

    def _decode(self, lst):
        lst2 = self._decodeFromFloat(lst)
        return lst2.astype(int)

    def _encode(self, lst, nb_player: int):
        while len(lst) != self._maxSequenceLength:
            lst.append([-1 for _ in range(nb_player)])
        ar = np.array(lst)
        ar = self._encodeToFloat(ar)
        return ar.reshape(1, self._maxSequenceLength, nb_player)

    def _predict(self, lst: List[List[int]], nb_player: int):
        return self._decode(self._model.predict(self._encode(lst, nb_player), batch_size=self._batchSize))\
                        .reshape(1, -1).tolist()[0]

    @property
    @abstractmethod
    def _maxSequenceLength(self) -> int:
        pass

    @property
    def _modelPath(self) -> str:
        return os.path.join("res", "AIs")
    
    @property
    @abstractmethod
    def _modelName(self) -> str:
        pass

    @property
    @abstractmethod
    def _batchSize(self) -> int:
        pass

    @abstractmethod
    def _encodeToFloat(self, sequence: np.ndarray) -> np.ndarray:  # int => float
        pass

    @abstractmethod
    def _decodeFromFloat(self, sequence: np.ndarray) -> np.ndarray:  # float -> int
        pass
