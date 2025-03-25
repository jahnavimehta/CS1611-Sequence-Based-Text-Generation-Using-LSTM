# Sequence-Based-Text-Generation-Using-LSTM

## Table of Contents

- [Project Description](#project-description)
- [Tech Stack](#tech-stack)
- [LSTM](#lstm)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation and usage](#installation-and-usage)

## Project Description

This project brings words to life using LSTM networks, a powerful DL model for predicting and generating text. The model is trained on a given corpus of text (Shakespeare’s hamlet) to predict the next word in a sequence. Allowing it to generate coherent sentences based on learned patterns. By leveraging NLP techniques, it demonstrates how LSTMs can be used to model language and generate human-like text.

## Tech Stack

- Programming Language: Python  
- Libraries & Frameworks:
  1. NLTK - text processing.
  2. Pandas and NumPy - data manipulation and numerical computations.
  3. TensorFlow Keras - text tokenization and sequence padding.
  4. Scikit-learn - dataset splitting.

## LSTM

- LSTM (Long Short-Term Memory) is a special type of recurrent reural retwork (RNN)
- It handles long-term dependencies in sequential data.
- Standard RNNs suffer from the vanishing gradient problem, which makes them ineffective for learning long-term dependencies.
- LSTMs overcome this by using gates that regulate the flow of information:
  1. Forget Gate – decides what information should be discarded from memory.
  2. Input Gate – decides what new information should be stored.
  3. Output Gate – decides what information to output.
  This allows LSTMs to remember important past information and discard irrelevant details, making them powerful for text-based tasks.

## Dataset

- The dataset consists of text corpus used for training the LSTM model.
- The dataset used for training is extracted from Project Gutenberg using the nltk.corpus.gutenberg module.
- The full text of Hamlet is loaded and saved locally for preprocessing and training.
- The text is tokenized and converted into numerical sequences before being fed into the model.
- Steps:
  1. Tokenization – splitting text into words/tokens.
  2. Sequence Generation – creating input-output pairs for training.
  3. Padding – ensuring sequences have the same length.
  4. Training the LSTM Model – learning patterns in the text.

## Model architecture

The model consists of:
1. Embedding layer - Converts words into dense vectors.
2. LSTM layers - captures sequential dependencies in text.
3. Dropout layer - prevents overfitting.
4. Dense layer with softmax activation - predicts the next word.

## Installation and usage

1. Create a virtual environment and install dependencies.
  - `python -m venv venv`
  - `venv\Scripts\activate`
  - `pip install -r requirements.txt`
2. Execute the jupyter notebook.
