import pathlib
import pickle
import random
from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", default="/media/jenazzad/Data/ML/stack_overflow_code"
    )
    args = parser.parse_args()

    OUTPUT_PATH = pathlib.Path(__file__).parents[1] / "outputs"

    INPUT_PATH = Path(args.input_path)

    samples = []

    with open(
        INPUT_PATH
        / "python_how_to_do_it_qid_by_classifier_unlabeled_single_code_answer_qid_to_code.pickle",
        "rb",
    ) as f:
        single_code = pickle.load(f)

    with open(
        INPUT_PATH
        / "python_how_to_do_it_qid_by_classifier_unlabeled_single_code_answer_qid_to_title.pickle",
        "rb",
    ) as f:
        single_question = pickle.load(f)

    for qid in single_question:

        samples.append({"question": single_question[qid], "code": single_code[qid]})

    with open(
        INPUT_PATH / "python_how_to_do_it_by_classifier_multiple_iid_to_code.pickle",
        "rb",
    ) as f:
        multiple_code = pickle.load(f)

    with open(
        INPUT_PATH / "python_how_to_do_it_by_classifier_multiple_qid_to_title.pickle",
        "rb",
    ) as f:
        multiple_question = pickle.load(f)

    for qid in multiple_question:

        code = "\n".join(
            [multiple_code[(qid, i)] for i in range(10) if (qid, i) in multiple_code]
        )

        samples.append({"question": multiple_question[qid], "code": code})

    df = pd.DataFrame(samples)

    df["split"] = df.apply(lambda x: 1 if random.random() < 0.9 else 0, axis=1)

    df.to_csv(OUTPUT_PATH / "data.csv", index=False)
