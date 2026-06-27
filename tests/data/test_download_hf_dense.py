from scripts.download_hf_dense import shard_filename, shard_url


def test_shard_filename_zero_padded_of_1024():
    assert shard_filename(0) == "train-00000-of-01024.msgpack.zst"
    assert shard_filename(7) == "train-00007-of-01024.msgpack.zst"
    assert shard_filename(1023) == "train-01023-of-01024.msgpack.zst"


def test_shard_url_points_at_hf_resolve_main():
    url = shard_url(0)
    assert url == (
        "https://huggingface.co/datasets/prdev/chessbench-full-policy-value/"
        "resolve/main/train-00000-of-01024.msgpack.zst"
    )
