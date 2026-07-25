#!/usr/bin/env python3
"""Prepend ferrum's FeNN v1 header to bullet's raw quantised gen-2 net -> gen2_*.bin.

gen-2 is a DATA experiment, not an architecture one: same plain-Chess768
768 -> 512x2 -> 1 net as gen-0, so the output is byte-compatible with gen0.bin and
loads in the deployed engine with no code change.

  header (16B) = b"FeNN" + version=1 + reserved + hidden(u16) + qa/qb/scale(i16) + reserved(u16)
  payload      = l0w[768 x H] + l0b[H] + l1w[2H] + l1b[1]   (all i16, feature-major)

For H=512: payload = 768*512*2 + 512*2 + 2*512*2 + 2 = 789,506 bytes
           gen2.bin = 16 + 789,506 = 789,522 bytes (identical to gen0.bin)

Usage: build_gen2_bin.py checkpoints/gen2_v1-40/quantised.bin gen2_v1.bin
"""
import os
import struct
import sys

HIDDEN, QA, QB, SCALE = 512, 255, 64, 400
PAYLOAD = 768 * HIDDEN * 2 + HIDDEN * 2 + 2 * HIDDEN * 2 + 2  # 789,506
TOTAL = 16 + PAYLOAD  # 789,522 — must equal gen0.bin's size


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "quantised.bin"
    dst = sys.argv[2] if len(sys.argv) > 2 else "gen2.bin"
    raw = open(src, "rb").read()
    print(f"{src}: {len(raw)} bytes (expected {PAYLOAD} payload + any alignment pad)")
    if len(raw) < PAYLOAD:
        sys.exit(f"ERROR: {src} too small ({len(raw)} < {PAYLOAD}) — layout differs from spec; inspect before proceeding")
    payload = raw[:PAYLOAD]  # strip bullet's trailing 64-byte alignment pad, as gen-0/gen-1 did
    # v1 header: the trailing u16 is reserved (gen-1's v2 header reuses it as num_buckets).
    header = b"FeNN" + struct.pack("<BBHhhhH", 1, 0, HIDDEN, QA, QB, SCALE, 0)
    assert len(header) == 16, len(header)
    with open(dst, "wb") as f:
        f.write(header)
        f.write(payload)
    total = os.path.getsize(dst)
    ok = total == TOTAL
    print(f"{dst}: {total} bytes (expected {TOTAL}); ok={ok}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
