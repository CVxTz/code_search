import torch

from code_search.models import TextEncoder


def test_text_encoder():
    n_vocab = 100

    source = torch.randint(low=0, high=n_vocab, size=(20, 16))
    target = torch.randint(low=0, high=n_vocab, size=(20, 32))

    text_encoder = TextEncoder(vocab_size=n_vocab)

    out = text_encoder((source, target))

    assert out[0].size() == (20, 256)
