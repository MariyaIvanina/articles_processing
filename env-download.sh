#!/bin/sh

source activate usaid
conda env list

python -m nltk.downloader punkt
python -m nltk.downloader words
python -m nltk.downloader stopwords
python -m nltk.downloader wordnet
python -m nltk.downloader averaged_perceptron_tagger

python -m spacy download en
python -m spacy validate