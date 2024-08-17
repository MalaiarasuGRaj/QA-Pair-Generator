#!/bin/bash

# Install Tesseract OCR and its dependencies
sudo apt-get update
sudo apt-get install -y tesseract-ocr
sudo apt-get install -y libtesseract-dev

# Download NLTK data
python -m nltk.downloader punkt
python -m nltk.downloader stopwords

# Install SpaCy model
python -m spacy download en_core_web_sm
