"""Source readers that emit LabeledPosition records.

- iter_kaggle_csv: a value-only SMOKE source (FEN + centipawn eval). Policy is set
  uniform over legal moves purely to exercise the pipeline shape; NOT for real
  training.
- iter_chessbench: the real training source (DeepMind ChessBench action-value
  .bag set: per-legal-move win% + state win% + ply/length). Documented here; the
  full run wires it in. Requires the `searchless_chess` data tooling and is not
  exercised by unit tests (no network).
"""

import csv
import chess

from src.data.labeled_position import LabeledPosition
from src.data.targets import cp_to_winprob, winprob_to_wdl


def _parse_cp(text: str) -> float:
    text = text.strip().replace("+", "")
    if text.startswith("#"):  # mate score, e.g. "#+3"
        return 10000.0 if "-" not in text else -10000.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def iter_kaggle_csv(path: str):
    """Yield value-only LabeledPositions from a Kaggle chessData.csv (FEN,Evaluation)."""
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fen = row["FEN"].strip()
            board = chess.Board(fen)
            legal = list(board.legal_moves)
            if not legal:
                continue
            cp = _parse_cp(row["Evaluation"])
            wp = cp_to_winprob(cp)
            uniform = 1.0 / len(legal)
            policy = [(m.uci(), uniform) for m in legal]
            yield LabeledPosition(
                fen=fen, policy=policy, wdl=winprob_to_wdl(wp), moves_left=60.0
            )


def iter_chessbench(path: str):
    """Real ChessBench action-value source. Implement against the searchless_chess
    .bag readers: for each position emit per-legal-move win% as `policy`
    (via src.data.targets.scores_to_policy on win% logits), `wdl` from the state
    win%, and `moves_left` from the game's ply index and length. Not unit-tested
    (requires the external dataset)."""
    raise NotImplementedError(
        "Wire to DeepMind ChessBench .bag readers for the full run; see docstring."
    )
