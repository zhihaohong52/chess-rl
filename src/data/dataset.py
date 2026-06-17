"""tf.data loader: TFRecord shards -> (inputs, targets) batches per contract."""

import tensorflow as tf

_FEATURES = {
    "square_tokens": tf.io.FixedLenFeature([], tf.string),
    "state_features": tf.io.FixedLenFeature([18], tf.float32),
    "legal_indices": tf.io.VarLenFeature(tf.int64),
    "legal_probs": tf.io.VarLenFeature(tf.float32),
    "wdl": tf.io.FixedLenFeature([3], tf.float32),
    "moves_left": tf.io.FixedLenFeature([1], tf.float32),
}


def _make_parse(policy_size: int):
    def _parse(record):
        ex = tf.io.parse_single_example(record, _FEATURES)
        sq = tf.cast(tf.io.decode_raw(ex["square_tokens"], tf.int8), tf.int32)  # [64]
        sq = tf.ensure_shape(sq, [64])
        sf = ex["state_features"]
        idx = tf.sparse.to_dense(ex["legal_indices"])     # [k]
        prob = tf.sparse.to_dense(ex["legal_probs"])      # [k]
        policy = tf.scatter_nd(idx[:, None], prob, [policy_size])  # [P]
        return (sq, sf), (policy, ex["wdl"], ex["moves_left"])
    return _parse


def make_dataset(shard_paths, batch_size: int, policy_size: int,
                 shuffle: bool = True, shuffle_buffer: int = 8192):
    ds = tf.data.TFRecordDataset(shard_paths, num_parallel_reads=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(_make_parse(policy_size), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
