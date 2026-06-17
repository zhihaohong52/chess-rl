import os
import tensorflow as tf
import chess
from src.data.labeled_position import LabeledPosition
from src.data.preencode import write_shard


def test_write_shard_roundtrip(tmp_path):
    lps = [
        LabeledPosition(chess.STARTING_FEN, [("e2e4", 1.0)], (0.5, 0.4, 0.1), 80.0),
        LabeledPosition(chess.STARTING_FEN, [("d2d4", 0.5), ("c2c4", 0.5)], (0.4, 0.5, 0.1), 79.0),
    ]
    path = str(tmp_path / "shard0.tfrecord")
    n = write_shard(lps, path)
    assert n == 2
    assert os.path.exists(path)
    count = sum(1 for _ in tf.data.TFRecordDataset([path]))
    assert count == 2
