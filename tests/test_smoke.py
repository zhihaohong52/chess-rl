def test_imports_and_torch():
    import numpy as np
    import chess
    import torch
    assert torch.__version__ >= "2.4"
    # MPS may or may not be available depending on hardware; just confirm the API exists.
    assert isinstance(torch.backends.mps.is_available(), bool)
    assert chess.square_mirror(0) == 56  # a1 -> a8
