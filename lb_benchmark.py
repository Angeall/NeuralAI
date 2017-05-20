import sys
import os
import inspect

import pandas as pd

from pytgf.data.benchmarking.benchmark import Benchmark
from pytgf.examples.lazerbike.builder import create_game
from pytgf.examples.lazerbike.controllers import LazerBikeBotPlayer


def perform_benchmark(nb_games):
    global df
    res = benchmark.benchmark(nb_games)
    df = pd.concat((df, pd.DataFrame([res[1], res[2]])), axis=1)  # type: pd.DataFrame
    df.to_csv(controllers_classes[0][1].__name__ + " VS " + controllers_classes[0][2].__name__ + ".csv")


def get_class(path, file_name):
    print(path)
    sys.path.insert(0, path)
    module_name = file_name[:-3]
    _module = __import__(module_name)
    for name, cls in inspect.getmembers(_module):  # Explore the classes inside the file
        if inspect.isclass(cls):
            if not inspect.isabstract(cls):  # The abstract type cannot be instantiated as it is
                print('find non-abstract class', cls, 'in file', file_name)
                if issubclass(cls, LazerBikeBotPlayer):
                    print("ok with", cls)
                    return cls
    raise TypeError("Didn't find a suitable class in file", file_name)


def count_in_file(path_to_file):
    results = [[0, 0, 0], [0, 0 ,0]]
    df = pd.read_csv(path_to_file)
    for i in range(2):
        for res in df.loc[i][1:]:
            res *= -1
            res += 1
            results[i][res] += 1
    return results


if __name__ == "__main__":
    sys.path.append(os.path.join("res", "AIs"))
    assert(len(sys.argv) > 2)
    
    file_name_1 = sys.argv[1].split(os.path.sep)[-1]
    file_name_2 = sys.argv[2].split(os.path.sep)[-1]
    path_1 = sys.argv[1][:-len(file_name_1)]
    path_2 = sys.argv[2][:-len(file_name_2)]
    
    cls1 = get_class(path_1, file_name_1)
    cls2 = get_class(path_2, file_name_2)
    
    controllers_classes = ({1: cls1, 2: cls2}, {1: 1, 2: 2})
    print(cls1, cls2)
    loop = create_game(controllers_classes, 10, 10, graphics=False)
    starting_api = loop.api
    controllers = [loop.getWrapperFromPlayerNumber(1).controller, loop.getWrapperFromPlayerNumber(2).controller]
    benchmark = Benchmark(starting_api, controllers)
    name = ""
    if len(sys.argv) > 3:
        number = int(sys.argv[3])
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
