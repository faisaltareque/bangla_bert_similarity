# Bangla BERT Similarity

This is a module to calculate similarity score between two bangla sentence using pretrained BERT(bangla-bert-base)

## Installation
First install [Pytorch](https://pytorch.org/get-started/locally/) on your virtual environment.

Then clone this repo

Run the following to install:
```bash
$ python setup.py bdist_wheel # to build
$ pip install -e .
```
To install directly from Github:
```bash
$ pip install git+https://github.com/faisaltareque/bangla_bert_similarity
```

## Developing Bangla BERT Similarity

To install Bangla BERT Similarity, along with the tools you need to develop and run tests, run the following in your virtual environment:

```bash
$ python setup.py bdist_wheel # to build
$ pip install -e .[dev]
```

## Usage
For first usage it will download pretrained BERT(bangla-bert-base).
```python
from bangla_bert_similarity import similarity
score = similarity.score('তোমার সাথে দেখা হয়ে ভালো লাগলো।', 'আপনার সাথে দেখা হয়ে ভালো লাগলো।')
```
To limit maximum sequence length. Can be less that 512. Default is 512. 

```python
from bangla_bert_similarity import similarity
score = similarity.score('তোমার সাথে দেখা হয়ে ভালো লাগলো।', 'আপনার সাথে দেখা হয়ে ভালো লাগলো।' ,  max_seq = 512)
```

To computer Euclidean(Lower means high similarity) similarity. Default is Cosine(Higher means high similarity).

```python
from bangla_bert_similarity import similarity
score = similarity.score('তোমার সাথে দেখা হয়ে ভালো লাগলো।', 'আপনার সাথে দেখা হয়ে ভালো লাগলো।' ,  similarity_method = 'euclidean')
```
