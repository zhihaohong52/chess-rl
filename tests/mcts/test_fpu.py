"""First-play-urgency (FPU) reduction in PUCT selection.

Default (fpu_reduction=None) must leave numerics identical to the historical
behavior where an unvisited child contributes q=0 to its PUCT score. When a
reduction is set, an unvisited child's q becomes parent_Q - fpu_reduction,
making the search exploit known-good moves before exploring fresh ones.
"""

import chess

from src.mcts.node import Node


def _two_child_root():
    root = Node(prior=0.0)
    a = chess.Move.from_uci("e2e4")
    b = chess.Move.from_uci("d2d4")
    root.expand_moves({a: 0.5, b: 0.5})
    # Parent has a positive Q from its own perspective.
    root.visit_count = 4
    root.value_sum = 2.0  # parent value = +0.5
    # Child a is visited with a neutral record; child b is unvisited.
    root.children[a].visit_count = 3
    root.children[a].value_sum = 0.0  # a.value=0 -> q_a=0
    return root, a, b


def test_default_no_fpu_matches_historical_selection():
    # Without FPU the unvisited child b has q=0 and a large exploration bonus
    # (1/(1+0)), so it outscores the visited child a.
    root, a, b = _two_child_root()
    choice, _ = root.select_child(c_puct=1.5)
    assert choice == b
    # Passing fpu_reduction=None is explicitly the same as omitting it.
    choice_none, _ = root.select_child(c_puct=1.5, fpu_reduction=None)
    assert choice_none == b


def test_fpu_reduction_flips_selection_to_exploit():
    # With a strong reduction, b's q drops to parent_Q - red = 0.5 - 3.0 = -2.5,
    # collapsing its score below the visited child a -> selection flips to a.
    root, a, b = _two_child_root()
    choice, _ = root.select_child(c_puct=1.5, fpu_reduction=3.0)
    assert choice == a


def test_fpu_does_not_change_visited_child_q():
    # A visited child's q must remain -child.value regardless of fpu_reduction;
    # only unvisited children use the FPU value. Verify by making b also visited
    # so FPU has no unvisited child to touch and the choice is reduction-stable.
    root, a, b = _two_child_root()
    root.children[b].visit_count = 3
    root.children[b].value_sum = -3.0  # b.value=-1 -> q_b=+1, clearly best
    without = root.select_child(c_puct=1.5)[0]
    with_fpu = root.select_child(c_puct=1.5, fpu_reduction=3.0)[0]
    assert without == b
    assert with_fpu == b
