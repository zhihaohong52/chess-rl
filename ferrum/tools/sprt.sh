#!/usr/bin/env bash
# SPRT-lite A/B test for a ferrum search patch.
#
#   tools/sprt.sh <baseline_bin> [candidate_bin]
#
# baseline_bin: a ferrum binary saved from the pre-patch HEAD, e.g.
#   cd ferrum && cargo build --release && cp target/release/ferrum /tmp/ferrum-base
# candidate_bin: defaults to a fresh working-tree release build.
#
# Env overrides: FASTCHESS, BOOK, TC, ROUNDS, CONCURRENCY, ELO0, ELO1, STOCKFISH.
set -euo pipefail

BASE="${1:?usage: sprt.sh <baseline_bin> [candidate_bin]}"
FERRUM_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FASTCHESS="${FASTCHESS:-/Users/james/Documents/GitHub/fastchess/fastchess}"
BOOK="${BOOK:-$FERRUM_DIR/books/openings_m1.epd}"
TC="${TC:-8+0.08}"
ROUNDS="${ROUNDS:-1000}"          # up to 2*ROUNDS games (paired, -repeat)
CONCURRENCY="${CONCURRENCY:-4}"
ELO0="${ELO0:-0}"; ELO1="${ELO1:-8}"

if [ -n "${2:-}" ]; then
  CAND="$2"
else
  ( cd "$FERRUM_DIR" && cargo build --release )
  CAND="$FERRUM_DIR/target/release/ferrum"
fi

[ -x "$BASE" ] || { echo "baseline not executable: $BASE" >&2; exit 1; }
[ -x "$CAND" ] || { echo "candidate not executable: $CAND" >&2; exit 1; }
[ -f "$BOOK" ] || { echo "opening book missing: $BOOK (run Task 2)" >&2; exit 1; }

PGN="$(mktemp -t ferrum-sprt-XXXXXX.pgn)"
echo "candidate: $CAND"
echo "baseline:  $BASE"
echo "book:      $BOOK   tc: $TC   sprt: [$ELO0,$ELO1]   cap: $((ROUNDS*2)) games"

"$FASTCHESS" \
  -engine cmd="$CAND" name=cand \
  -engine cmd="$BASE" name=base \
  -each tc="$TC" \
  -openings file="$BOOK" format=epd order=random \
  -rounds "$ROUNDS" -games 2 -repeat \
  -concurrency "$CONCURRENCY" \
  -sprt elo0="$ELO0" elo1="$ELO1" alpha=0.05 beta=0.05 \
  -pgnout file="$PGN"

echo "pgn: $PGN"
