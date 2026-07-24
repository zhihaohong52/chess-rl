#!/usr/bin/env bash
# Parallel self-play datagen. Spawns N workers, each a `ferrum selfplay` with a
# distinct --seed writing its own shard files, and prints a running total.
#
#   tools/datagen.sh <workers> <games_per_worker> <out_dir> [nodes]
#
# Env: FERRUM (binary), EVALFILE (net, REQUIRED — HCE datagen is a bug), NODES.
# Launch this UNDER the setsid daemon (see docs) for long runs; it survives on
# its own only while the terminal lives.
set -euo pipefail
WORKERS="${1:?usage: datagen.sh <workers> <games_per_worker> <out_dir> [nodes]}"
GAMES="${2:?games_per_worker}"
OUT_DIR="${3:?out_dir}"
NODES="${4:-${NODES:-5000}}"
FERRUM_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FERRUM="${FERRUM:-$FERRUM_DIR/target/release/ferrum}"
EVALFILE="${EVALFILE:?set EVALFILE to nnue/gen0.bin — datagen must use the NNUE net}"

mkdir -p "$OUT_DIR"
echo "datagen: $WORKERS workers x $GAMES games @ $NODES nodes/move -> $OUT_DIR"
echo "net: $EVALFILE"

pids=()
for w in $(seq 1 "$WORKERS"); do
  seed=$(( w * 1000 + 7 ))
  EVALFILE="$EVALFILE" "$FERRUM" selfplay \
    --seed "$seed" --games "$GAMES" --nodes "$NODES" \
    --out "$OUT_DIR/shard" --net "$EVALFILE" \
    > "$OUT_DIR/worker.$seed.log" 2>&1 &
  pids+=($!)
done
echo "spawned pids: ${pids[*]}"
wait "${pids[@]}"
total=$(cat "$OUT_DIR"/shard.*.txt 2>/dev/null | wc -l | tr -d ' ')
echo "datagen complete: $total positions in $OUT_DIR"
