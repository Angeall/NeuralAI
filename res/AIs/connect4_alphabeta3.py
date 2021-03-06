try:
    from connect4_alphabeta import Connect4AlphaBeta
except ModuleNotFoundError:
    from .connect4_alphabeta import Connect4AlphaBeta


class Connect4AlphaBeta3(Connect4AlphaBeta):

    @property
    def _maxDepth(self):
        return 3