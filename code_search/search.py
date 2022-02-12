import pathlib
import pickle
import time

import torch
from annoy import AnnoyIndex

from code_search.models import TextEncoder
from code_search.preprocessing_utils import VOCAB_SIZE, tokenize

MAX_LEN = 256


if __name__ == "__main__":
    # Search

    # search_term = "start an ipython shell"
    # search_term = "escape html special characters"
    # search_term = "set numpy random state"
    # search_term = "how to validate email with regex"
    # search_term = "Get data from XML file Python"
    # search_term = "Scrape data from a table with scrapy"
    # search_term = "get max from from array"

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--search_term", default="get all keys from dict")
    args = parser.parse_args()

    search_term = args.search_term

    base_path = pathlib.Path(__file__).parents[1]

    model_path = base_path / "models" / "text_encoder.ckpt"

    output_file = base_path / "outputs" / "code_vectors.pkl"

    with open(output_file, "rb") as f:
        samples = pickle.load(f)

    t = AnnoyIndex(len(samples[0]["vector"]), "angular")

    for i, sample in enumerate(samples):
        t.add_item(i, sample["vector"])

    t.build(100)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TextEncoder(lr=1e-4, vocab_size=VOCAB_SIZE)

    model.load_state_dict(torch.load(model_path)["state_dict"])

    model.eval()
    model.to(device)

    # Dry run
    with torch.no_grad():
        _ = model.encode(torch.tensor([[0, 0, 0]], dtype=torch.long).to(device))

    start_time_encoding = time.time()
    x1 = tokenize(search_term).ids
    x1 = torch.tensor([x1], dtype=torch.long).to(device)

    with torch.no_grad():
        search_vector = model.encode(x1).squeeze().cpu().numpy().tolist()
    end_time_encoding = time.time()

    start_time_search = time.time()
    indexes = t.get_nns_by_vector(search_vector, n=3, search_k=-1)
    end_time_search = time.time()

    for i, index in enumerate(indexes):
        print(f"Result {i+1} -->")
        sample = samples[index]

        print(sample["code"])

    print(f"Query encoded in {end_time_encoding - start_time_encoding} seconds")
    print(f"Search results found in in {end_time_search - start_time_search} seconds")
