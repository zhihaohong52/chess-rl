from src.data.labeled_position import LabeledPosition


def test_labeled_position_fields():
    lp = LabeledPosition(
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        policy=[("e2e4", 0.6), ("d2d4", 0.4)],
        wdl=(0.5, 0.4, 0.1),
        moves_left=80.0,
        repetition_count=0,
    )
    assert lp.fen.startswith("rnbqkbnr")
    assert abs(sum(p for _, p in lp.policy) - 1.0) < 1e-6
    assert len(lp.wdl) == 3
