import torch
from src.model.presets import resolve_config
from src.model.transformer import ChessTransformer


def test_dropout_changes_train_not_eval():
    cfg = resolve_config("baseline-v1")
    cfg.transformer_dropout = 0.5
    net = ChessTransformer(cfg)
    sq = torch.zeros(2, 64, dtype=torch.long); sf = torch.zeros(2, 18)
    net.train()
    torch.manual_seed(0); a = net(sq, sf)[0]
    torch.manual_seed(1); b = net(sq, sf)[0]
    assert not torch.allclose(a, b)
    net.eval()
    c = net(sq, sf)[0]; d = net(sq, sf)[0]
    assert torch.allclose(c, d)
