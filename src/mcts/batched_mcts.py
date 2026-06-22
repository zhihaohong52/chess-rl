"""Batched PUCT MCTS over chess.Board, using a TransformerEvaluator."""

import numpy as np

from config import Config
from src.mcts.node import Node


def _terminal_value(board) -> float:
    """Value from the side-to-move's perspective for a finished game."""
    if board.is_checkmate():
        return -1.0  # side to move is mated
    return 0.0       # stalemate / draw


class BatchedMCTS:
    def __init__(self, evaluator, config=None, num_simulations=None, batch_size: int = 8,
                 tablebase=None):
        self.evaluator = evaluator
        self.config = config or Config()
        self.num_simulations = num_simulations or self.config.num_simulations
        self.c_puct = self.config.c_puct
        self.batch_size = batch_size
        # Optional Syzygy tablebase: when set, leaves it can probe get exact
        # endgame values instead of the neural net. None = unchanged behavior.
        self.tablebase = tablebase
        self._root = None
        self._tracked = None  # chess.Board matching self._root

    def reset(self):
        self._root = None
        self._tracked = None

    def _reuse_or_new_root(self, board):
        if (self._root is not None and self._tracked is not None
                and self._tracked.fen() == board.fen()):
            return self._root
        self._root = Node(prior=0.0)
        self._tracked = board.copy()
        return self._root

    def _backprop(self, path, value):
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    def search(self, board, add_noise: bool = False):
        root = self._reuse_or_new_root(board)
        if not root.is_expanded():
            policy, _ = self.evaluator.evaluate(board)
            root.expand_moves(policy)
        if add_noise and root.is_expanded():
            root.add_dirichlet_noise(self.config.dirichlet_alpha, self.config.dirichlet_epsilon)

        done = 0
        while done < self.num_simulations:
            paths, boards, leaves, terminals = [], [], [], []
            n = min(self.batch_size, self.num_simulations - done)
            for _ in range(n):
                node = root
                b = board.copy()
                path = [node]
                while node.is_expanded() and not b.is_game_over():
                    move, node = node.select_child(self.c_puct)
                    b.push(move)
                    path.append(node)
                for nd in path:
                    nd.add_virtual_loss()
                if b.is_game_over():
                    terminals.append((path, _terminal_value(b)))
                else:
                    tb_val = (self.tablebase.probe_value(b)
                              if self.tablebase is not None else None)
                    if tb_val is not None:
                        # Exact endgame value — treat like a terminal: no NN
                        # evaluation, no expansion, just back up the true value.
                        terminals.append((path, tb_val))
                    else:
                        paths.append(path); boards.append(b); leaves.append(node)
                done += 1

            if boards:
                evals = self.evaluator.evaluate_batch(boards, [0] * len(boards))
                for path, node, (policy, value) in zip(paths, leaves, evals):
                    node.expand_moves(policy)
                    for nd in path:
                        nd.remove_virtual_loss()
                    self._backprop(path, value)
            for path, value in terminals:
                for nd in path:
                    nd.remove_virtual_loss()
                self._backprop(path, value)
        return root

    def choose_move(self, board, temperature: float = 0.0, add_noise: bool = False):
        root = self.search(board, add_noise=add_noise)
        moves, probs = root.get_policy()
        if not moves:
            return None
        if temperature == 0:
            return moves[int(np.argmax(probs))]
        adjusted = np.asarray(probs, dtype=np.float64) ** (1.0 / temperature)
        adjusted /= adjusted.sum()
        return moves[int(np.random.choice(len(moves), p=adjusted))]

    def advance(self, move):
        """Promote the subtree for `move` (tree reuse during play)."""
        if self._root is not None and move in self._root.children:
            self._root = self._root.children[move]
            if self._tracked is not None:
                self._tracked.push(move)
        else:
            self.reset()
