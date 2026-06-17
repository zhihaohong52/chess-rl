import os
from src.data.chessbench import iter_kaggle_csv
from src.data.labeled_position import LabeledPosition

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "sample_chessdata.csv")


def test_iter_kaggle_csv_yields_value_only_positions():
    lps = list(iter_kaggle_csv(FIX))
    assert len(lps) == 2
    assert all(isinstance(lp, LabeledPosition) for lp in lps)
    # value-only smoke source: policy is uniform over legal moves and sums to 1
    for lp in lps:
        assert abs(sum(p for _, p in lp.policy) - 1.0) < 1e-6
        assert all(0.0 <= w <= 1.0 for w in lp.wdl)
