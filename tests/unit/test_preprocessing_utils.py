from code_search.preprocessing_utils import (four_indent,
                                             replace_special_tokens,
                                             replace_with_space, tokenize)


def test_replace_special_tokens_1():

    x = "@deprecated('scrapy.utils.python.to_unicode') DCNL def str_to_unicode(text, encoding=None, errors='strict'):"

    out = "@deprecated('scrapy.utils.python.to_unicode')\ndef str_to_unicode(text, encoding=None, errors='strict'):"

    assert replace_special_tokens(x) == out


def test_replace_special_tokens_2():

    x = (
        " DCSP sender = email['From'] DCNL DCSP m = re.match('(.*)\\s<.*>', sender) DCNL DCSP if m: DCNL DCSP  DCSP "
        "return m.group(1) DCNL DCSP return sender"
    )

    out = (
        " sender = email['From']\n"
        " m = re.match('(.*)\\s<.*>', sender)\n"
        " if m:\n"
        "  return m.group(1)\n"
        " return sender"
    )

    assert replace_special_tokens(x) == out


def test_four_indent_1():

    x = (
        " DCSP sender = email['From'] DCNL DCSP m = re.match('(.*)\\s<.*>', sender) DCNL DCSP if m: DCNL DCSP  DCSP "
        "return m.group(1) DCNL DCSP return sender"
    )

    out = (
        "    sender = email['From']\n"
        "    m = re.match('(.*)\\s<.*>', sender)\n"
        "    if m:\n"
        "        return m.group(1)\n"
        "    return sender"
    )

    assert four_indent(replace_special_tokens(x)) == out


def test_replace_with_space_1():

    x = "def get_all_phrases():"

    out = "def get _ all _ phrases (  )  : "

    assert replace_with_space(x) == out


def test_replace_with_space_2():

    x = "def replaceAcronyms(text):"

    out = "def replace Acronyms ( text )  : "

    assert replace_with_space(x) == out


def test_tokenize_1():

    x = "def replaceAcronyms(text):"

    out = [
        "[CLS]",
        "def",
        "replace",
        "acron",
        "##y",
        "##ms",
        "(",
        "text",
        ")",
        ":",
        "[SEP]",
    ]

    assert tokenize(x).tokens == out
