import sys
import os

from pytgf.data.gatherer import Gatherer
from pytgf.data.routines import ReinforcementRoutine
from pytgf.controls.controllers.passive import Passive
from pytgf.examples.connect4.builder import create_game
from pytgf.examples.connect4.rules import Connect4API

from c4_benchmark import get_class

if __name__ == "__main__":
    assert (len(sys.argv) > 2)

    file_name_1 = sys.argv[1].split(os.path.sep)[-1]
    file_name_2 = sys.argv[2].split(os.path.sep)[-1]
    path_1 = sys.argv[1][:-len(file_name_1)]
    path_2 = sys.argv[2][:-len(file_name_2)]

    cls1 = get_class(path_1, file_name_1)
    cls2 = get_class(path_2, file_name_2)

    own_controller = cls1(1)
    opponent_controllers = [cls2(2)]
    battle = ReinforcementRoutine(own_controller, opponent_controllers, Gatherer([]), tuple(range(7)),
                                  lambda api: {player: 100 * api.hasWon(player) for player in (1, 2)})

    core = create_game({1: Passive, 2: Passive}, 2, 2, False).game
    state = Connect4API(core)
    battle.routine(1, state)
