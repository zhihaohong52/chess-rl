# ferrum

A from-scratch UCI chess engine in Rust. Alpha-beta search; NNUE evaluation
from M2 onward. Part of a staged program targeting genuine 3000+ CCRL-blitz
Elo (design spec:
../docs/superpowers/specs/2026-07-14-ferrum-nnue-engine-design.md).

This crate lives in the `ferrum/` subdirectory of the chess-rl repo; run all
commands below from `ferrum/`.

## Build & run

    cargo build --release
    ./target/release/ferrum            # UCI mode
    ./target/release/ferrum bench      # node-count checksum
    ./target/release/ferrum perft 5    # perft from startpos

## Testing

    cargo test --release
