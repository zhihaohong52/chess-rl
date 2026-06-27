"""Pinned evaluation datasets. Binaries stay local (gitignored); only metrics
are committed. The dense val shard is regenerated deterministically on demand.
"""

import os

from src.eval.puzzles import load_chessbench_puzzles

PUZZLE_CSV = "data/puzzles.csv"
TEST_BAG = "data/test/action_value_data.bag"
DENSE_VAL_PATH = "data/fixtures/dense_val.npz"
DENSE_VAL_SIZE = 5000


def load_gate_puzzles(n: int):
    """First n ChessBench puzzles (deterministic: file order)."""
    return load_chessbench_puzzles(path=PUZZLE_CSV, max_puzzles=n)


def ensure_dense_val(n: int = DENSE_VAL_SIZE) -> str:
    """Regenerate the pinned dense val shard from the 62k test bag if absent.

    Deterministic: the first n grouped positions of the test bag. Returns the
    shard path.
    """
    if os.path.exists(DENSE_VAL_PATH):
        return DENSE_VAL_PATH
    import itertools
    from src.data.chessbench import iter_chessbench
    from src.data.preencode import write_shard
    if not os.path.exists(TEST_BAG):
        raise FileNotFoundError(
            f"{TEST_BAG} not found; download it with scripts/download_chessbench.py --test"
        )
    os.makedirs(os.path.dirname(DENSE_VAL_PATH), exist_ok=True)
    positions = list(itertools.islice(iter_chessbench(TEST_BAG), n))
    write_shard(positions, DENSE_VAL_PATH)
    return DENSE_VAL_PATH


def mate_in_one_positions():
    """(fen, best_uci) where best_uci delivers immediate checkmate.

    These three FENs are pre-verified: pushing best_uci yields is_checkmate().
    """
    return [
        # Back-rank mate: Re8#.
        ("6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1", "e1e8"),
        # Queen mate supported by the king on h6: Qg7#.
        ("6k1/8/7K/8/8/8/1Q6/8 w - - 0 1", "b2g7"),
        # Black back-rank: Re1#.
        ("4r1k1/8/8/8/8/8/5PPP/6K1 b - - 0 1", "e8e1"),
    ]
