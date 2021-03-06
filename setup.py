import os

from setuptools import setup

path = os.path.abspath(os.path.dirname(__file__))

try:
    with open(os.path.join(path, "requirements.txt"), encoding="utf-8") as f:
        REQUIRED = f.read().split("\n")
except FileNotFoundError:
    REQUIRED = []

setup(
    name="code_search",
    version="0.0.1",
    author="Youness MANSAR",
    author_email="mansaryounessecp@gmail.com",
    description="nlp",
    license="GNU",
    keywords="nlp",
    url="https://github.com/CVxTz/code_search",
    packages=["code_search"],
    classifiers=[
        "Topic :: Utilities",
    ],
    data_files=[('code_search', ['code_search/tokenizer.json'])],
    include_package_data=True,
    install_requires=REQUIRED,
)
