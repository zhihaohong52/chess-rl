from src.data.bagz import read_records, write_records


def test_bagz_roundtrip(tmp_path):
    recs = [b"alpha", b"", b"gamma-record", bytes([0, 1, 2, 3, 255])]
    path = str(tmp_path / "x.bag")
    n = write_records(recs, path)
    assert n == len(recs)
    assert list(read_records(path)) == recs


def test_bagz_empty(tmp_path):
    path = str(tmp_path / "e.bag")
    write_records([], path)
    assert list(read_records(path)) == []
