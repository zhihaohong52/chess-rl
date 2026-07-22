#!/usr/bin/env bash
# Anchored gauntlet: ferrum vs each pinned anchor engine, rated with Ordo.
#   tools/gauntlet.sh [ferrum_bin]
# Anchors live in bench/anchors/bin/; their CCRL ratings are fixed via
# bench/anchors/anchors.txt (one "Name Rating" per line, matching -engine names).
set -euo pipefail
FERRUM_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FASTCHESS="${FASTCHESS:-/Users/james/Documents/GitHub/fastchess/fastchess}"
BOOK="${BOOK:-$FERRUM_DIR/books/openings_m1.epd}"
BIN="$FERRUM_DIR/bench/anchors/bin"
TC="${TC:-8+0.08}"
ROUNDS="${ROUNDS:-140}"          # per opponent, paired; 4 anchors => 1,120 games
CONCURRENCY="${CONCURRENCY:-4}"
FERRUM="${1:-$FERRUM_DIR/target/release/ferrum}"

PGN="$(mktemp -t ferrum-gauntlet-XXXXXX.pgn)"
# Optionally start ferrum with an NNUE net loaded (M2+); HCE if EVALFILE unset.
FERRUM_OPTS=(); [ -n "${EVALFILE:-}" ] && FERRUM_OPTS=( option.EvalFile="$EVALFILE" )
ENGINES=( -engine cmd="$FERRUM" name=ferrum "${FERRUM_OPTS[@]}" )
for e in "$BIN"/*; do
  n="$(basename "$e")"
  [ "$n" = ordo ] && continue
  [ -x "$e" ] || continue
  ENGINES+=( -engine cmd="$e" name="$n" )
done

"$FASTCHESS" "${ENGINES[@]}" \
  -each tc="$TC" \
  -openings file="$BOOK" format=epd order=random \
  -rounds "$ROUNDS" -games 2 -repeat \
  -concurrency "$CONCURRENCY" \
  -tournament gauntlet \
  -pgnout file="$PGN"

echo "pgn: $PGN"
# NOTE: this is a ROUGH single-number estimate only, so the runner is smoke-testable
# now. The real fixed-anchor rating (each opponent pinned to its CCRL rating) is run in
# Task 17 with the exact, version-checked Ordo invocation. Do not treat this line as the
# M1 exit number.
"$BIN/ordo" -Q -p "$PGN" || true
