#!/usr/bin/env python3
"""Rebuild bullet-loadable f32 weights from the deployed gen0.bin (for the gen-2 V2 fine-tune).

Why this exists: bullet fine-tunes by loading an optimiser checkpoint
(`<net_id>-<n>/optimiser_state/weights.bin`), but gen-0's checkpoint died with
its (deleted) training instance — all that survives is the quantised, deployed
`gen0.bin`. Quantisation is a pure scale, so the f32 weights are recoverable to
within the i16 step (l0*: 1/255 ~= 0.004, l1w: 1/64, l1b: 1/16320). That is
plenty for an INIT (gen2_train.rs loads weights only, not momentum/velocity, and
uses a correspondingly low LR).

Output format is bullet's own weight-store wire format (crates/trainer/src/model/
weights.rs::write_to_byte_buffer @ cebc78a), per weight, concatenated:
    <ascii id> b"\\n"  +  u64 LE element count  +  count * f32 LE
`WeightsStore::load_from` keys off the id, so entry order does not matter and a
file containing only these four ids is sufficient.

Usage: gen0_to_bullet_weights.py [gen0.bin] [gen0_weights.bin]
Feed the result to gen2_train.rs as GEN2_INIT.
"""
import os
import struct
import sys

HIDDEN, QA, QB = 512, 255, 64
HEADER = 16

# (bullet weight id, element count, quantisation scale used by gen-0's save_format)
LAYOUT = [
    ("l0w", 768 * HIDDEN, QA),
    ("l0b", HIDDEN, QA),
    ("l1w", 2 * HIDDEN, QB),
    ("l1b", 1, QA * QB),
]
PAYLOAD = sum(n for _, n, _ in LAYOUT) * 2  # 789,506 bytes of i16


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "gen0.bin"
    dst = sys.argv[2] if len(sys.argv) > 2 else "gen0_weights.bin"

    raw = open(src, "rb").read()
    if raw[:4] != b"FeNN":
        sys.exit(f"ERROR: {src} has bad magic {raw[:4]!r} — not a ferrum net")
    version, _, hidden, qa, qb, scale, _ = struct.unpack("<BBHhhhH", raw[4:HEADER])
    print(f"{src}: FeNN v{version} hidden={hidden} qa={qa} qb={qb} scale={scale} ({len(raw)} bytes)")
    if (version, hidden, qa, qb) != (1, HIDDEN, QA, QB):
        sys.exit(f"ERROR: expected FeNN v1 hidden={HIDDEN} qa={QA} qb={QB}; refusing to guess a layout")
    if len(raw) != HEADER + PAYLOAD:
        sys.exit(f"ERROR: expected {HEADER + PAYLOAD} bytes, got {len(raw)}")

    out = bytearray()
    offset = HEADER
    for name, count, scale_q in LAYOUT:
        ints = struct.unpack_from(f"<{count}h", raw, offset)
        offset += count * 2
        floats = [v / scale_q for v in ints]
        # Self-check: dequantise-then-requantise must return the ORIGINAL i16s, i.e.
        # this step adds no error of its own on top of gen-0's original quantisation.
        requantised = [round(f * scale_q) for f in floats]
        assert requantised == list(ints), f"{name}: dequantisation is not round-trip exact"
        out += name.encode("ascii") + b"\n"
        out += struct.pack("<Q", count)
        out += struct.pack(f"<{count}f", *floats)
        print(f"  {name}: {count} f32, range [{min(floats):+.4f}, {max(floats):+.4f}], round-trip exact")

    assert offset == len(raw), (offset, len(raw))
    with open(dst, "wb") as f:
        f.write(out)
    print(f"{dst}: {os.path.getsize(dst)} bytes")


if __name__ == "__main__":
    main()
