"""Minimal reader/writer for the 'bagz' record container used by DeepMind's
ChessBench (github.com/google-deepmind/searchless_chess, Apache-2.0).

Uncompressed `.bag` layout:
    [record_0][record_1]...[record_{n-1}][offset table]
The offset table holds one little-endian signed int64 per record: the cumulative
END byte offset of that record. The final offset equals the size of the records
section (= the table's start), so reading the last 8 bytes as an unsigned int64
yields the index pointer.

`.bagz` files store each record individually zstd-compressed.
"""

import struct


def read_records(path, decompress=None):
    """Yield raw record bytes from a .bag (or zstd .bagz) file."""
    if decompress is None:
        decompress = path.endswith(".bagz")
    with open(path, "rb") as fh:
        data = fh.read()
    size = len(data)
    if size == 0:
        return
    if size < 8:
        raise ValueError(f"bagz file too small: {path}")
    index_start = struct.unpack("<Q", data[-8:])[0]
    if index_start > size or (size - index_start) % 8 != 0:
        raise ValueError(f"corrupt bagz index in {path}")
    num_records = (size - index_start) // 8
    dctx = None
    if decompress:
        import zstandard as zstd  # lazy: only .bagz needs it
        dctx = zstd.ZstdDecompressor()
    prev = 0
    for i in range(num_records):
        off = index_start + i * 8
        (end,) = struct.unpack("<q", data[off:off + 8])
        chunk = data[prev:end]
        prev = end
        if dctx is not None and chunk:
            chunk = dctx.decompress(chunk)
        yield chunk


def write_records(records, path):
    """Write an iterable of raw record bytes to an uncompressed .bag file.

    Returns the number of records written. Used by tests/tooling; the real
    ChessBench files are produced by DeepMind's pipeline.
    """
    n = 0
    offsets = []
    with open(path, "wb") as fh:
        pos = 0
        for r in records:
            fh.write(r)
            pos += len(r)
            offsets.append(pos)
            n += 1
        for off in offsets:
            fh.write(struct.pack("<q", off))
    return n
