from config import Config


def test_transformer_config_present():
    assert Config.d_model == 256
    assert Config.n_layers == 8
    assert Config.n_heads == 8
    assert Config.d_ff == 1024
    assert Config.smolgen_compress == 32
    assert Config.smolgen_hidden == 128
    assert Config.smolgen_gen == 128
    assert Config.state_dim == 18
    assert Config.distill_batch_size == 1024
