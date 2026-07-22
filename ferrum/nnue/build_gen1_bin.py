#!/usr/bin/env python3
"""Prepend ferrum's FeNN v2 header to bullet's raw quantised gen-1 net -> gen1.bin.

Mirrors build_gen0_bin.py but for the king-bucketed v2 format:
  header (16B) = b"FeNN" + version=2 + reserved + hidden(u16) + qa/qb/scale(i16) + num_buckets(u16)
  payload      = l0w[B*768 x H] + l0b[H] + l1w[2H] + l1b[1]   (all i16, feature-major, per gen-0)

For H=1024, B=4:  payload = 4*768*1024*2 + 1024*2 + 2*1024*2 + 2 = 6,297,602 bytes
                  gen1.bin = 16 + 6,297,602 = 6,297,618 bytes
"""
import struct, sys, os

HIDDEN, QA, QB, SCALE, NUM_BUCKETS = 1024, 255, 64, 400, 4
PAYLOAD = NUM_BUCKETS * 768 * HIDDEN * 2 + HIDDEN * 2 + 2 * HIDDEN * 2 + 2  # 6,297,602

def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "quantised.bin"
    dst = sys.argv[2] if len(sys.argv) > 2 else "gen1.bin"
    raw = open(src, "rb").read()
    print(f"{src}: {len(raw)} bytes (expected {PAYLOAD} payload + any alignment pad)")
    if len(raw) < PAYLOAD:
        sys.exit(f"ERROR: {src} too small ({len(raw)} < {PAYLOAD}) — layout differs from spec; inspect before proceeding")
    payload = raw[:PAYLOAD]  # strip any trailing alignment pad, as gen-0 did
    header = b"FeNN" + struct.pack("<BBHhhhH", 2, 0, HIDDEN, QA, QB, SCALE, NUM_BUCKETS)
    assert len(header) == 16, len(header)
    with open(dst, "wb") as f:
        f.write(header)
        f.write(payload)
    total = os.path.getsize(dst)
    ok = total == 16 + PAYLOAD
    print(f"{dst}: {total} bytes (expected {16 + PAYLOAD} = 6,297,618); ok={ok}")
    if not ok:
        sys.exit(1)

if __name__ == "__main__":
    main()
