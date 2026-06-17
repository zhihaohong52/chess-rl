def test_imports_and_tf():
    import numpy as np
    import chess
    import tensorflow as tf
    assert tf.__version__ >= "2.16"
    assert chess.square_mirror(0) == 56  # a1 -> a8
