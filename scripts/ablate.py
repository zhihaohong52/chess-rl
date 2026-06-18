"""Run the gate suite over one or more checkpoints and emit a comparison table.

Usage:
  python scripts/ablate.py --ckpt checkpoints/baseline-v1/best.pt \
      --out docs/ablations/run.md
Each checkpoint must have a sidecar (or pass --objective/--preset for bare ones).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.routing import load_for_eval
from src.eval.gates import run_gates
from src.training.checkpoint_meta import read_sidecar


def run_row(ckpt_path, device="cpu", objective=None, preset=None,
            gate_kwargs=None) -> dict:
    gate_kwargs = gate_kwargs or {}
    meta = read_sidecar(ckpt_path) or {}
    preset = preset or meta.get("preset")
    net, ev = load_for_eval(ckpt_path, objective=objective, preset=preset,
                            device=device)
    row = {"preset": preset, "params": sum(p.numel() for p in net.parameters())}
    row.update(run_gates(net, ev, device, **gate_kwargs))
    return row


def _columns(rows):
    cols = ["preset", "params"]
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    return cols


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def to_markdown(rows) -> str:
    cols = _columns(rows)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(_fmt(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join([head, sep] + body) + "\n"


def to_csv(rows) -> str:
    cols = _columns(rows)
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(_fmt(r.get(c, "")) for c in cols))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="output .md path (.csv written alongside)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--objective", default=None)
    ap.add_argument("--preset", default=None)
    ap.add_argument("--fast", action="store_true",
                    help="skip MCTS+dense val for a quick run")
    args = ap.parse_args()

    gate_kwargs = (dict(puzzle_counts=(300, 1000), mcts_puzzles=0, dense_val=False)
                   if args.fast else {})
    rows = [run_row(c, device=args.device, objective=args.objective,
                    preset=args.preset, gate_kwargs=gate_kwargs)
            for c in args.ckpt]
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(to_markdown(rows))
    csv_path = os.path.splitext(args.out)[0] + ".csv"
    with open(csv_path, "w") as fh:
        fh.write(to_csv(rows))
    print(f"wrote {args.out} and {csv_path}")
    print(to_markdown(rows))


if __name__ == "__main__":
    main()
