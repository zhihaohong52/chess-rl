import chess

from src.engine.opening_book import OpeningBook, book_available
from src.engine.syzygy_tb import syzygy_available


class StubReader:
    """Polyglot reader stand-in: returns a fixed move or raises a no-entry error."""

    def __init__(self, move=None, raise_exc=None):
        self._move = move
        self._raise = raise_exc

    def _entry(self):
        if self._raise is not None:
            raise self._raise
        return type("Entry", (), {"move": self._move})()

    def find(self, board, *args, **kwargs):
        return self._entry()

    def weighted_choice(self, board, *args, **kwargs):
        return self._entry()

    def close(self):
        pass


def test_book_available(tmp_path):
    assert book_available(None) is False
    assert book_available(str(tmp_path / "missing.bin")) is False
    p = tmp_path / "book.bin"
    p.write_bytes(b"\x00" * 16)
    assert book_available(str(p)) is True


def test_lookup_returns_legal_book_move():
    e2e4 = chess.Move.from_uci("e2e4")
    book = OpeningBook(_reader=StubReader(move=e2e4))
    assert book.lookup(chess.Board()) == e2e4


def test_lookup_rejects_illegal_move():
    illegal = chess.Move.from_uci("e2e5")  # pawn can't jump 3 from start
    book = OpeningBook(_reader=StubReader(move=illegal))
    assert book.lookup(chess.Board()) is None


def test_lookup_none_when_out_of_book():
    book = OpeningBook(_reader=StubReader(raise_exc=IndexError()))
    assert book.lookup(chess.Board()) is None


def test_syzygy_available(tmp_path):
    assert syzygy_available(None) is False
    assert syzygy_available(str(tmp_path / "nope")) is False   # missing dir
    assert syzygy_available(str(tmp_path)) is False            # empty dir
    (tmp_path / "KQvK.rtbw").write_bytes(b"\x00")
    assert syzygy_available(str(tmp_path)) is True
