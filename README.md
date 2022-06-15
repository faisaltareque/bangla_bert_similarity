# Bangla BERT Similarity

This is a module to calculate similarity score between two bangla sentence using pretrained BERT(bangla-bert-base)

## Installation
First install [Pytorch](https://pytorch.org/get-started/locally/) on your virtual environment.

Then clone this repo

Run the following to install:
```bash
$ python setup.py bdist_wheel 
$ pip install -e .
```

# Developing Bangla BERT Similarity

To install Bangla BERT Similarity, along with the tools you need to develop and run tests, run the following in your virtual environment:

```bash
$ python setup.py bdist_wheel
$ pip install -e .[dev]
```

## Usage

```python
from bangla_bert_similarity import similarity_score
score = similarity_score('তোমার সাথে দেখা হয়ে ভালো লাগলো।', 'আপনার সাথে দেখা হয়ে ভালো লাগলো।')
```