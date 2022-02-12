import re
from pathlib import Path

from tokenizers import Tokenizer

TOKENIZER_PATH = Path(__file__).parent / "tokenizer.json"
TOKENIZER = Tokenizer.from_file(str(TOKENIZER_PATH))
PAD_IDX = TOKENIZER.token_to_id("[PAD]")
VOCAB_SIZE = TOKENIZER.get_vocab_size()


def tokenize(text):
    output = TOKENIZER.encode(replace_with_space(text))
    return output


def replace_special_tokens(text):

    return (
        text.replace(" DCNL DCSP ", "\n ")
        .replace(" DCNL ", "\n")
        .replace(" DCSP ", " ")
    )


special_chars = re.compile(r"([_\-=()\"\':,.])")
camel_case = re.compile(r"(?<=[a-z])(?=[A-Z])")


def replace_with_space(string):
    string = re.sub(special_chars, r" \1 ", string)
    string = re.sub(camel_case, r" ", string)

    return string


def four_indent(text):
    lines = text.split("\n")
    new_lines = []

    for line in lines:

        leading_space = len(line) - len(line.lstrip())

        new_lines.append(" " * leading_space * 4 + line.lstrip())

    return "\n".join(new_lines)
