import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup


class PositionalEncoding(nn.Module):
    #  https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0:, :, 0::2] = torch.sin(position * div_term)
        pe[0:, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """

        x = x + self.pe[:, : x.size(1)]

        return self.dropout(x)


class TokenEmbedding(nn.Module):
    #  https://pytorch.org/tutorials/beginner/translation_transformer.html
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class TextEncoder(pl.LightningModule):
    def __init__(
        self,
        vocab_size,
        channels=256,
        dropout=0.3,
        lr=1e-4,
    ):
        super().__init__()

        self.lr = lr
        self.dropout = dropout
        self.vocab_size = vocab_size

        self.embeddings = TokenEmbedding(vocab_size=self.vocab_size, emb_size=channels)

        self.pos_encoder = PositionalEncoding(d_model=channels, dropout=dropout)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            batch_first=True, d_model=channels, nhead=4, dim_feedforward=4 * channels
        )

        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=6
        )

        self.layer_norm = torch.nn.LayerNorm(normalized_shape=channels)
        self.linear = torch.nn.Linear(channels, channels, bias=False)

        self.do = nn.Dropout(p=self.dropout)

    def init_weights(self) -> None:
        init_range = 0.1
        self.embeddings.weight.data.uniform_(-init_range, init_range)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-init_range, init_range)

    def encode(self, x):

        x = self.embeddings(x)
        x = self.pos_encoder(x)

        x = self.encoder(x)

        x = x[:, 0, :]

        x = torch.tanh(self.layer_norm(x))

        return x

    def encode_plus(self, x):
        x = self.do(self.encode(x))
        x = self.linear(x)

        return x

    def forward(self, x):
        x1, x2 = x

        x1 = self.encode_plus(x1)
        x2 = self.encode(x2)

        return x1, x2

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, name="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, name="valid")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, name="test")

    def _step(self, batch, batch_idx, name="train"):
        x1, x2 = self(batch)

        y = torch.arange(x1.size(0), device=x1.device)

        y_hat = torch.mm(x1, x2.t())

        loss = F.cross_entropy(y_hat, y, label_smoothing=0.1)

        _, predicted = torch.max(y_hat, 1)
        accuracy = (predicted == y).double().mean()

        self.log(f"{name}_loss", loss)
        self.log(f"{name}_accuracy", accuracy)

        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=5e-2)

        num_cycles = 10000
        num_training_steps = 1200 * num_cycles

        lr_schedulers = {
            "scheduler": get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer=opt,
                num_warmup_steps=5000,
                num_training_steps=num_training_steps,
                num_cycles=num_cycles,
            ),
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }
        return [opt], [lr_schedulers]


if __name__ == "__main__":
    n_vocab = 100

    source = torch.randint(low=0, high=n_vocab, size=(20, 16))
    target = torch.randint(low=0, high=n_vocab, size=(20, 32))

    text_encoder = TextEncoder(vocab_size=n_vocab)

    out = text_encoder((source, target))
    print(out[0].size())
    print(out)
