import os
import subprocess
import sys
import types

import scripts.arena_eval as ae

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_load_book_tb_none_when_unset():
    args = types.SimpleNamespace(book=None, syzygy=None, max_pieces=5)
    assert ae._load_book_tb(args) == (None, None)


def test_load_book_tb_skips_missing_paths():
    # given-but-missing paths must skip gracefully (warn, no crash)
    args = types.SimpleNamespace(book="/no/such/book.bin", syzygy="/no/such/tb", max_pieces=5)
    book, tablebase = ae._load_book_tb(args)
    assert book is None and tablebase is None


def test_cli_help_shows_book_and_syzygy_flags():
    res = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts/arena_eval.py"), "--help"],
        capture_output=True, text=True, cwd=REPO,
    )
    assert res.returncode == 0
    for flag in ("--book", "--syzygy", "--max-pieces", "--tb-one-side"):
        assert flag in res.stdout
