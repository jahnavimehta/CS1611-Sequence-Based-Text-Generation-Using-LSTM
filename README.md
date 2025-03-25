# Sequence-Based-Text-Generation-Using-LSTM

## Table of Contents

- [Project Description](#project-description)
- [LSTM](#lstm)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation and usage](#installation-and-usage)
- [Contact](#contact)

## Project Description

This project brings words to life using LSTM networks, a powerful deep learning model for predicting and generating text. The model is trained on a given corpus of text (Shakespeare’s Hamlet) to predict the next word in a sequence. Allowing it to generate coherent sentences based on learned patterns. By leveraging Natural Language Processing (NLP) techniques, it demonstrates how LSTMs can be used to model language and generate human-like text.

## LSTM

- LSTM (Long Short-Term Memory) is a special type of recurrent reural retwork (RNN)
- It handles long-term dependencies in sequential data.
- Standard RNNs suffer from the vanishing gradient problem, which makes them ineffective for learning long-term dependencies.
- LSTMs overcome this by using gates that regulate the flow of information:
  1. Forget Gate – Decides what information should be discarded from memory.
  2. Input Gate – Decides what new information should be stored.
  3. Output Gate – Decides what information to output.
  This allows LSTMs to remember important past information and discard irrelevant details, making them powerful for text-based tasks.

## Dataset

- The dataset consists of text corpus used for training the LSTM model.
- The dataset used for training is extracted from Project Gutenberg using the nltk.corpus.gutenberg module.
- The full text of Hamlet is loaded and saved locally for preprocessing and training.
- The text is tokenized and converted into numerical sequences before being fed into the model.
- Preprocessing Steps:
  1. Tokenization – Splitting text into words/tokens.
  2. Sequence Generation – Creating input-output pairs for training.
  3. Padding – Ensuring sequences have the same length.
  4. Training the LSTM Model – Learning patterns in the text.

## Model Architecture
The model consists of:
1. Embedding Layer: Converts words into dense vectors.
2. LSTM Layers: Captures sequential dependencies in text.
3. Dropout Layer: Prevents overfitting.
4. Dense Layer with Softmax Activation: Predicts the next word.

## Installation and usage
1. Create a virtual environment and install dependencies.
  - `python -m venv venv`
  - `venv\Scripts\activate`
  - `pip install -r requirements.txt`
2.Execute the Jupyter Notebook.
