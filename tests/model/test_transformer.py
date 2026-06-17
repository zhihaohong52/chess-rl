import tempfile
import torch
import chess
from config import Config
from src.model.transformer import ChessTransformer
from src.game.token_encoder import encode_batch
from src.game.move_encoder import get_move_encoder


def _net():
    return ChessTransformer(Config)


def _inp(boards):
    sq, sf = encode_batch(boards, [0] * len(boards))
    return torch.tensor(sq, dtype=torch.long), torch.tensor(sf, dtype=torch.float32)


def test_forward_shapes():
    net = _net()
    sq, sf = _inp([chess.Board(), chess.Board()])
    pol, wdl, ml = net(sq, sf)
    P = get_move_encoder().policy_size
    assert tuple(pol.shape) == (2, P)
    assert tuple(wdl.shape) == (2, 3)
    assert tuple(ml.shape) == (2, 1)


def test_param_count_in_budget():
    net = _net()
    n = sum(p.numel() for p in net.parameters() if p.requires_grad)
    assert 8_000_000 < n < 14_000_000, n


def test_save_load_roundtrip():
    net = _net().eval()
    sq, sf = _inp([chess.Board()])
    with torch.no_grad():
        p1, _, _ = net(sq, sf)
    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/w.pt"
        torch.save(net.state_dict(), path)
        net2 = _net()
        net2.load_state_dict(torch.load(path))
        net2.eval()
    with torch.no_grad():
        p2, _, _ = net2(sq, sf)
    assert torch.allclose(p1, p2, atol=1e-5)
