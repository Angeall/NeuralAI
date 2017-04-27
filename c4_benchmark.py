import sys
import os
import inspect

import pandas as pd

from pytgf.data.benchmarking.benchmark import Benchmark
from pytgf.examples.connect4.builder import create_game
from pytgf.examples.connect4.controllers import Connect4BotPlayer


def perform_benchmark(nb_games):
    global df
    res = benchmark.benchmark(nb_games)
    df = pd.concat((df, pd.DataFrame([res[1], res[2]])), axis=1)  # type: pd.DataFrame
    df.to_csv(controllers_classes[1].__name__ + " VS " + controllers_classes[2].__name__ + ".csv")
    
def get_class(path, file_name):
    print(path)
    sys.path.insert(0, path)
    module_name = file_name[:-3]
    _module = __import__(module_name)
    for name, cls in inspect.getmembers(_module):  # Explore the classes inside the file
        if inspect.isclass(cls):
            if not inspect.isabstract(cls):  # The abstract type cannot be instantiated as it is
                if issubclass(cls, Connect4BotPlayer):
                    print("ok with", cls)
                    return cls


if __name__ == "__main__":
    assert(len(sys.argv) > 2)
    
    file_name_1 = sys.argv[1].split(os.path.sep)[-1]
    file_name_2 = sys.argv[2].split(os.path.sep)[-1]
    path_1 = sys.argv[1][:-len(file_name_1)]
    path_2 = sys.argv[2][:-len(file_name_2)]
    
    cls1 = get_class(path_1, file_name_1)
    cls2 = get_class(path_2, file_name_2)
    
    controllers_classes = {1: cls1, 2: cls2}
    print(cls1, cls2)
    loop = create_game(controllers_classes, 10, 10, False)
    starting_api = loop.api
    controllers = [loop.getWrapperFromPlayerNumber(1).controller, loop.getWrapperFromPlayerNumber(2).controller]
    benchmark = Benchmark(starting_api, controllers)
    name = ""
    if len(sys.argv) > 3:
        number = sys.argv[3]
    else:
        number = 100
    df = pd.DataFrame()
    remaining = number
    for _ in range(number//100):
        perform_benchmark(100)
        remaining -= 100
        print("!" + 10*'-' + str(_ + 1) + '/' + str(number//100) + 10*'-' + '!')
    if remaining > 0:
        perform_benchmark(remaining)
