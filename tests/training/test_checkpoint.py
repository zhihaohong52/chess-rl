import numpy as np
import torch
import chess
from config import Config
from src.model.transformer import ChessTransformer
from src.game.token_encoder import encode_batch


def test_checkpoint_roundtrip(tmp_path):
    net = ChessTransformer(Config)
    sq, sf = encode_batch([chess.Board()], [0])
    sq_t = torch.tensor(sq, dtype=torch.long)
    sf_t = torch.tensor(sf, dtype=torch.float32)

    with torch.no_grad():
        p1, _, _ = net(sq_t, sf_t)

    path = str(tmp_path / "ck.pt")
    torch.save(net.state_dict(), path)

    net2 = ChessTransformer(Config)
    net2.load_state_dict(torch.load(path, map_location="cpu"))

    with torch.no_grad():
        p2, _, _ = net2(sq_t, sf_t)

    assert np.allclose(p1.numpy(), p2.numpy(), atol=1e-5)
