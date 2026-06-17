import time
import pytest
import chess
from src.data.labeled_position import LabeledPosition
from src.data.preencode import write_shard


@pytest.mark.slow
def test_preencode_throughput(tmp_path):
    lps = [LabeledPosition(chess.STARTING_FEN, [("e2e4", 1.0)], (0.5, 0.4, 0.1), 80.0)
           for _ in range(2000)]
    path = str(tmp_path / "bench.tfrecord")
    t0 = time.time()
    n = write_shard(lps, path)
    rate = n / (time.time() - t0)
    print(f"preencode rate: {rate:.0f} positions/sec")
    assert rate > 200  # must comfortably beat GPU consumption
