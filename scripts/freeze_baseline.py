# scripts/freeze_baseline.py
"""Copy the current trained policy model to checkpoints/baseline-v1/ and write
its sidecar (preset=baseline-v1, objective=policy). The .pt stays local."""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.checkpoint_meta import write_sidecar

SRC = "checkpoints/distill/best.pt"
DST_DIR = "checkpoints/baseline-v1"
DST = os.path.join(DST_DIR, "best.pt")


def main():
    if not os.path.exists(SRC):
        raise SystemExit(f"{SRC} not found; cannot freeze baseline")
    os.makedirs(DST_DIR, exist_ok=True)
    shutil.copyfile(SRC, DST)
    write_sidecar(DST, {"preset": "baseline-v1", "objective": "policy",
                        "train_data": "chessbench-test-bag-62k"})
    print(f"froze baseline-v1 -> {DST} (+ sidecar)")


if __name__ == "__main__":
    main()
