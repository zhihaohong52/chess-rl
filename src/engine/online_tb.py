"""Online Syzygy tablebase via the public Lichess endpoint (≤7 men).

Wraps an optional *local* `SyzygyTablebase` so the engine keeps its instant
5-man fast-path and its MCTS leaf-probing behaviour unchanged, while extending
the **root** move oracle out to 7 men over the network. Leaf probing
(`probe_value`) is never networked — it delegates straight to the local tables
(None when absent / out of local scope), so MCTS never touches the API.

The Lichess `standard` endpoint takes a FEN and returns the position verdict
plus every legal move ranked best-first, so `moves[0].uci` is already the
DTZ/50-move-optimal move — the zeroing correctness our local `_rank_key`
implements is done server-side, and against the position's real half-move clock.
Any failure (out of scope, network error, empty/garbage response, illegal move)
returns None so the caller falls back to the local tables / MCTS.

Use for a single root probe per move in a game. Do NOT call it per MCTS leaf or
for bulk dataset generation — that hammers a free public service.
"""

from __future__ import annotations

import json
import ssl
import urllib.parse
import urllib.request
from typing import Optional

import chess

LICHESS_STANDARD = "https://tablebase.lichess.ovh/standard"


def _default_ssl_context() -> ssl.SSLContext:
    """A cert-verifying context that works off macOS too.

    Python's `urllib` ignores the system keychain, so on macOS it can't find a CA
    bundle and every HTTPS probe dies with CERTIFICATE_VERIFY_FAILED. Point it at
    certifi's bundle when available (verification stays ON); fall back to the
    stdlib default elsewhere.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


class OnlineSyzygyTablebase:
    """Duck-typed like `SyzygyTablebase`: `best_dtz_move`, `probe_value`,
    `in_scope`, `close` — drop-in for `build_hybrid_mover(..., tablebase=)`.
    """

    def __init__(
        self,
        local=None,
        max_pieces: int = 7,
        timeout: float = 5.0,
        endpoint: str = LICHESS_STANDARD,
        _opener=None,
    ) -> None:
        self.local = local
        self.max_pieces = max_pieces
        self.timeout = timeout
        self.endpoint = endpoint
        self._opener = _opener  # injectable fetch(url)->bytes, for offline tests
        self._ssl_ctx = None    # lazily built cert-verifying context (real fetches)
        self.calls = 0          # network probes issued (telemetry)
        self.hits = 0           # network probes that yielded a legal move

    # --- leaf probing: LOCAL only, never networked -----------------------
    def in_scope(self, board: chess.Board) -> bool:
        return self.local.in_scope(board) if self.local is not None else False

    def probe_value(self, board: chess.Board) -> Optional[float]:
        return self.local.probe_value(board) if self.local is not None else None

    # --- root move oracle: local fast-path -> online ≤7 ------------------
    def best_dtz_move(self, board: chess.Board) -> Optional[chess.Move]:
        if self.local is not None:
            mv = self.local.best_dtz_move(board)
            if mv is not None:
                return mv  # instant 5-man hit, no network
        if chess.popcount(board.occupied) > self.max_pieces:
            return None
        return self._online_best_move(board)

    def _online_best_move(self, board: chess.Board) -> Optional[chess.Move]:
        url = f"{self.endpoint}?{urllib.parse.urlencode({'fen': board.fen()})}"
        self.calls += 1
        try:
            data = json.loads(self._fetch(url))
            moves = data.get("moves") or []
            if not moves:
                return None
            uci = moves[0].get("uci")
            mv = chess.Move.from_uci(uci) if uci else None
        except Exception:
            return None  # any failure -> caller falls back to MCTS
        if mv is None or mv not in board.legal_moves:
            return None
        self.hits += 1
        return mv

    def _fetch(self, url: str) -> bytes:
        if self._opener is not None:
            return self._opener(url)
        if self._ssl_ctx is None:
            self._ssl_ctx = _default_ssl_context()
        req = urllib.request.Request(
            url, headers={"User-Agent": "chess-rl/endgame-arena"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout,
                                    context=self._ssl_ctx) as resp:
            return resp.read()

    def close(self) -> None:
        if self.local is not None:
            self.local.close()

    def __enter__(self) -> "OnlineSyzygyTablebase":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
