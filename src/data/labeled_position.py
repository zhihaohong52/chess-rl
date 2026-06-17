"""Universal labeled-position record consumed by the pre-encoder."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class LabeledPosition:
    fen: str
    policy: List[Tuple[str, float]]   # (uci_move, prob) over legal moves, sums to 1
    wdl: Tuple[float, float, float]   # (win, draw, loss) from side-to-move POV
    moves_left: float                 # plies to game end (>= 0)
    repetition_count: int = 0
