import chess
from src.mcts.node import Node


def test_expand_moves_keys_by_move():
    root = Node(prior=0.0)
    m1 = chess.Move.from_uci("e2e4")
    m2 = chess.Move.from_uci("d2d4")
    root.expand_moves({m1: 0.7, m2: 0.3})
    assert set(root.children.keys()) == {m1, m2}
    assert abs(root.children[m1].prior - 0.7) < 1e-6


def test_virtual_loss_discourages_then_restores():
    root = Node(prior=0.0)
    m1 = chess.Move.from_uci("e2e4")
    m2 = chess.Move.from_uci("d2d4")
    root.expand_moves({m1: 0.5, m2: 0.5})
    # root.visit_count=2 so sqrt(2)~1.414; m1 has visits+strong Q so it wins on score
    root.visit_count = 2
    # give m1 a strong positive record from parent's view (low child.value)
    root.children[m1].visit_count = 5
    root.children[m1].value_sum = -5.0  # child.value=-1 -> q=+1 attractive
    first, _ = root.select_child(c_puct=1.5)
    assert first == m1
    root.children[m1].add_virtual_loss(3)
    second, _ = root.select_child(c_puct=1.5)
    assert second == m2  # virtual loss pushed selection away from m1
    root.children[m1].remove_virtual_loss(3)
    assert root.children[m1].visit_count == 5
    assert abs(root.children[m1].value_sum - (-5.0)) < 1e-9
