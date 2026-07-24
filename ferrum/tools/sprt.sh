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
# Load an NNUE net into BOTH engines (M2+); HCE if EVALFILE unset. Search-feature
# SPRTs MUST set this — otherwise the A/B compares HCE engines, not the real one.
# Per-engine nets: CAND_EVALFILE/BASE_EVALFILE override the shared EVALFILE.
# Used by net-vs-net SPRTs (M6: gen-2 cand vs gen-0 base). Falls back to the
# shared EVALFILE for search-feature SPRTs (both engines same net).
CAND_NET="${CAND_EVALFILE:-${EVALFILE:-}}"
BASE_NET="${BASE_EVALFILE:-${EVALFILE:-}}"
CAND_OPTS=(); [ -n "$CAND_NET" ] && CAND_OPTS=( option.EvalFile="$CAND_NET" )
BASE_OPTS=(); [ -n "$BASE_NET" ] && BASE_OPTS=( option.EvalFile="$BASE_NET" )
# Optional adjudication — a THROUGHPUT lever only. At a fixed TC, games/hour is
# wall-clock bound, so the only ways to go faster are more concurrency, a shorter
# TC (which changes what is measured), or ending decided/dead games sooner. In an
# even-strength A/B the pool is draw-heavy (M5 run 1: 69% draws, nearly all by
# 3-fold repetition, i.e. played to the bitter end), so draw adjudication is the
# cheapest real speedup. Both flags affect the two engines symmetrically.
# OFF by default: M1-M4 gates ran without them, and leaving them off keeps new
# gates comparable to those unless a run explicitly opts in.
ADJ=()
[ -n "${DRAW_ADJ:-}" ]   && ADJ+=( -draw movenumber=40 movecount=8 score=10 )
[ -n "${RESIGN_ADJ:-}" ] && ADJ+=( -resign movecount=3 score=400 )
echo "candidate: $CAND"
echo "baseline:  $BASE"
echo "cand net:  ${CAND_NET:-<none, HCE>}"
echo "base net:  ${BASE_NET:-<none, HCE>}"
echo "adjudicate: draw=${DRAW_ADJ:-off} resign=${RESIGN_ADJ:-off}"
echo "book:      $BOOK   tc: $TC   sprt: [$ELO0,$ELO1]   cap: $((ROUNDS*2)) games"

"$FASTCHESS" \
  -engine cmd="$CAND" name=cand "${CAND_OPTS[@]}" \
  -engine cmd="$BASE" name=base "${BASE_OPTS[@]}" \
  -each tc="$TC" \
  -openings file="$BOOK" format=epd order=random \
  -rounds "$ROUNDS" -games 2 -repeat \
  -concurrency "$CONCURRENCY" \
  ${ADJ[@]+"${ADJ[@]}"} \
  -sprt elo0="$ELO0" elo1="$ELO1" alpha=0.05 beta=0.05 \
  -pgnout file="$PGN"

echo "pgn: $PGN"
