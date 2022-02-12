import pathlib
import pickle
from functools import partial

import pandas as pd
import torch
import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from code_search.models import TextEncoder
from code_search.preprocessing_utils import (PAD_IDX, VOCAB_SIZE, four_indent,
                                             tokenize)

MAX_LEN = 256


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        code = df.loc[idx, "code"]

        x1 = tokenize(code).ids

        x1 = torch.tensor(x1, dtype=torch.long)

        return x1, code


def generate_batch(data_batch, pad_idx):
    input_1, input_2 = [], []
    for (x1, code) in data_batch:
        input_1.append(x1[:MAX_LEN])
        input_2.append(four_indent(code))
    input_1 = pad_sequence(input_1, padding_value=pad_idx, batch_first=True)

    return input_1, input_2


if __name__ == "__main__":

    batch_size = 100

    base_path = pathlib.Path(__file__).parents[1]

    data_path = pathlib.Path(__file__).parents[1] / "outputs" / "data.csv"

    df = pd.read_csv(data_path)

    model_path = base_path / "models" / "text_encoder.ckpt"
    output_file = base_path / "outputs" / "code_vectors.pkl"

    prediction_data = Dataset(dataframe=df)

    print("len(prediction_data)", len(prediction_data))

    prediction_loader = DataLoader(
        prediction_data,
        batch_size=batch_size,
        num_workers=5,
        shuffle=False,
        collate_fn=partial(generate_batch, pad_idx=PAD_IDX),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TextEncoder(lr=1e-4, vocab_size=VOCAB_SIZE)

    model.load_state_dict(torch.load(model_path)["state_dict"])

    model.eval()

    model.to(device)

    samples = []

    for tensor, codes in tqdm.tqdm(prediction_loader):

        with torch.no_grad():
            vectors = model.encode_plus(tensor.to(device)).cpu().numpy()

        for i, code in enumerate(codes):
            samples.append({"code": code, "vector": vectors[i, :].tolist()})

    with open(output_file, "wb") as f:
        pickle.dump(samples, f)
