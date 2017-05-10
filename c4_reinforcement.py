import sys
import os

import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model

from pytgf.data.decode.action_sequence_decoder import ActionSequenceDecoder
from pytgf.data.gatherer import Gatherer
from pytgf.data.routines import ReinforcementRoutine
from pytgf.controls.controllers.passive import Passive
from pytgf.examples.connect4.builder import create_game
from pytgf.examples.connect4.rules import Connect4API

from c4_benchmark import get_class
from res.AIs.neuralnetworkbot import NeuralNetworkBot


MAX_LEN = 21
NB_VICTORIES = 100


def create_dataset(dataset: list, player_number: int, nb_players: int=2, default_value: int=-1000):
    # Padding sequence so they all have the same size
    dataX = pad_sequences(dataset, maxlen=MAX_LEN, padding='post', value=[default_value for _ in range(nb_players)])
    # The labels are a sequence so that each action leads to a predicted action
    dataY = [[sequence[i][player_number] for i in range(1, len(sequence))] for sequence in dataX]
    # Removing the last action of the original data set (the last move to predict)
    dataX = [sequence[:-1] for sequence in dataX]
    return np.array(dataX), np.array(dataY)


def learn(decoder: ActionSequenceDecoder, own_controller: NeuralNetworkBot, sequences: pd.DataFrame, pl_num: int,
          nb_victories: int):
    model = own_controller._model  # type: Model
    dataset = decoder._parseDataFrame(sequences)

    data_x, data_y = create_dataset(dataset, pl_num, 2, -1)
    data_x_copy = data_x.copy()
    data_x = data_x.reshape(data_x.shape[0] * data_x.shape[1], data_x.shape[2])
    data_x = (data_x + 1) / 7.
    data_y = (data_y + 1) / 7.
    data_x = data_x.reshape(data_x_copy.shape)
    data_y = data_y.reshape(-1, 20, 1)

    model.fit(data_x, data_y, batch_size=1, epochs=100)
    model.save(own_controller.__class__.__name__ + str(nb_victories) + ".h5")


if __name__ == "__main__":
    assert (len(sys.argv) > 2)

    pl_number = 1

    actions_decoder = ActionSequenceDecoder(-1, ".", pl_number - 1, 2, verbose=False)

    file_name_1 = sys.argv[1].split(os.path.sep)[-1]
    file_name_2 = sys.argv[2].split(os.path.sep)[-1]
    path_1 = sys.argv[1][:-len(file_name_1)]
    path_2 = sys.argv[2][:-len(file_name_2)]

    cls1 = get_class(path_1, file_name_1)
    cls2 = get_class(path_2, file_name_2)

    core = create_game({1: Passive, 2: Passive}, 2, 2, False).game
    state = Connect4API(core)

    own_controller = cls1(1)  # type: NeuralNetworkBot
    own_controller.gameState = core
    opponent_controllers = [cls2(2)]
    opponent_controllers[0].gameState = core
    battle = ReinforcementRoutine(own_controller, opponent_controllers, Gatherer([]), tuple(range(7)),
                                  lambda api: {player: 100 * api.hasWon(player) for player in (1, 2)},
                                  must_write_files=False, must_keep_temp_files=False, min_end_states=NB_VICTORIES,
                                  min_victories=NB_VICTORIES)

    games = []
    i = 0
    while True:
        actions_sequences = battle.routine(pl_number, state)
        games.append(actions_sequences.shape[0]//2)
        learn(actions_decoder, own_controller, actions_sequences, pl_number - 1, i*NB_VICTORIES)
        i += 1
        print("Took", games[-1], "games to obtain", NB_VICTORIES, "victories")
        pd.DataFrame(games).to_csv(own_controller.__class__.__name__ + "_learning.csv")
