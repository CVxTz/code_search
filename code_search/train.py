import pathlib
import random
from functools import partial

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from code_search.models import TextEncoder
from code_search.preprocessing_utils import PAD_IDX, VOCAB_SIZE, tokenize

MAX_LEN = 256


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        code = df.loc[idx, "code"]
        question = df.loc[idx, "question"]

        x1 = tokenize(code).ids
        x2 = tokenize(question).ids

        if random.random() < 0.5:
            x1, x2 = x2, x1

        x1 = torch.tensor(x1, dtype=torch.long)
        x2 = torch.tensor(x2, dtype=torch.long)

        return x1, x2


def generate_batch(data_batch, pad_idx):
    input_1, input_2 = [], []
    for (x1, x2) in data_batch:
        input_1.append(x1[:MAX_LEN])
        input_2.append(x2[:MAX_LEN])
    input_1 = pad_sequence(input_1, padding_value=pad_idx, batch_first=True)
    input_2 = pad_sequence(input_2, padding_value=pad_idx, batch_first=True)

    return input_1, input_2


if __name__ == "__main__":

    batch_size = 100
    epochs = 100000

    data_path = pathlib.Path(__file__).parents[1] / "outputs" / "data.csv"
    # model_path = pathlib.Path(__file__).parents[1] / "models" / "save_text_encoder.ckpt"

    df = pd.read_csv(data_path)

    train, val = df[df.split == 1], df[df.split == 0]

    base_path = pathlib.Path(__file__).parents[1]

    train_data = Dataset(dataframe=train)
    val_data = Dataset(dataframe=val)

    print("len(train_data)", len(train_data))
    print("len(val_data)", len(val_data))

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=5,
        shuffle=True,
        collate_fn=partial(generate_batch, pad_idx=PAD_IDX),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=5,
        shuffle=True,
        collate_fn=partial(generate_batch, pad_idx=PAD_IDX),
    )

    model = TextEncoder(lr=1e-4, vocab_size=VOCAB_SIZE)

    # model.load_state_dict(torch.load(model_path)["state_dict"])

    logger = TensorBoardLogger(
        save_dir=str(base_path),
        name="logs",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        mode="min",
        dirpath=base_path / "models",
        filename="text_encoder",
        save_weights_only=True,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=1,
        logger=logger,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=1,
        precision=16,
    )
    trainer.fit(model, train_loader, val_loader)
