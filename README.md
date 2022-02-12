# code_search

## Data

[Data source](https://github.com/LittleYUYU/StackOverflow-Question-Code-Dataset/)

Download files "python_how_to_do_it_qid_*.pickle" from this repository.

## Run

```commandline
python -m pip install -e .
```

### Steps

```commandline
python code_search/build_dataset.py
python code_search/train.py
python code_search/predict.py
```
To search for a code snippet using:

```commandline
python code_search/search.py --search_term "how to validate email with regex"
```