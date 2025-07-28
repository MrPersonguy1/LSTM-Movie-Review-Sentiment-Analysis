# -*- coding: utf-8 -*-
"""
##🟦Importing
"""

import nltk
nltk.download('punkt_tab')

import torch
import torch.nn as nn

import glob
import os
import pandas as pd

from tqdm import tqdm
import time

import kagglehub

"""
### Dataset Downloading
"""

path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")

print(f"Path: {path}")


!ls "/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv"

"""
## 🟧Data Preprocessing
"""

dataset = pd.read_csv('/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')

print(len(dataset))

print(dataset.shape)

pd.set_option('display.max_colwidth', 200)

print(dataset.head(5))

"""
### Data Cleaning
"""

def cleaner(text):
    text = text.replace('<br />', ' ')
    text = text.replace('<br/>', ' ')
    text = text.replace('<br>', ' ')

    text = text.lower()

    for punctuation in ".,!?:;":
      text = text.replace(punctuation, '')
    return text

"""
###Vocab
"""

from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

tokens = []

def build_vocab_simple(texts, max_words = 10000):
  for text in texts:
    for word in word_tokenize(cleaner(text)):
      tokens.append(word)

  freq_dist = FreqDist(tokens)

  vocab = ['<PAD>', '<UNK>'] + [word for word, _ in freq_dist.most_common(max_words)]

  most_common = freq_dist.most_common(max_words - 2)

  return {word: idx for idx, word in enumerate(vocab)}

l = ["a", 'b', 'c']

for idx, word, in enumerate(l):
  print(idx, word)

time_start = time.time()
vocab = build_vocab_simple(dataset['review'])
print("Vocabulary size:", len(vocab))
time_end = time.time()
print(f'Time to build vocab: {time_end - time_start}')

"""
###Top 20 Words
"""

top_20_words = list(vocab.items())[:20]

for word, index in top_20_words:
  print(f'Word: {word}')

"""
###⌛Preprocess Movie Reviews
"""

from torch.utils.data import TensorDataset, DataLoader

def prepare_movie_reviews(reviews, labels, vocab, max_len = 200):
  all_input_ids = []
  all_labels = []

  for review, label, in zip(reviews, labels):
    tokens = word_tokenize(review)[:max_len]

    indices = [vocab.get(token, vocab['<UNK>']) for token in tokens]

    if len(indices) < max_len:
      indices += [0] * (max_len - len(indices))

    all_input_ids.append(indices)
    all_labels.append(1 if label == 'positive' else 0)

  inputs_tensor = torch.tensor(all_input_ids)
  labels_tensor = torch.tensor(all_labels)

  dataset = TensorDataset(inputs_tensor, labels_tensor)

  return dataset

"""
### Train and Test Data
"""

from sklearn.model_selection import train_test_split

reviews = dataset['review']
labels = dataset['sentiment']

x_train, x_test, y_train, y_test = train_test_split(reviews, labels)

print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))

"""
###Train Dataloader
"""

start = time.time()
train_dataset = prepare_movie_reviews(x_train, y_train, vocab)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
end = time.time()
print(f'Time to prepare data: {end - start}')

sample_batch = next(iter(train_loader))

print(sample_batch[0].shape)
print(sample_batch[1].shape)

"""
## 🟩Model Building

###⌛Building The LSTM
"""

class SentimentLSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim = 200, hidden_dim = 256, n_layers = 2, drop_prob = .3):
      super().__init__()

      self.embeddings = nn.Embedding(vocab_size, embedding_dim)
      self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout= drop_prob, batch_first = True)
      self.linear1 = nn.Linear(hidden_dim, 1)
      self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
      embedded = self.dropout(self.embeddings(x))

      lstm_out, _ = self.lstm(embedded)

      out = self.linear1(lstm_out[:, -1, :])

      return torch.sigmoid(out)

model = SentimentLSTM(vocab_size = len(vocab))
model

"""
### Model Device
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

"""
### Essential Hyperparameters
"""

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

"""
## 🟥Training
"""

review, label = next(iter(train_loader))

print(review.shape)

print(label.shape)

def train_model(model, device, train_loader, criterion, optimizer, epochs):
  print(device)
  model.to(device)

  model.train()

  for epoch in range(epochs):
      total_loss = 0

      for inputs, labels, in tqdm(train_loader):
        inputs, labels = inputs.to(device), label.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()

        output = model(inputs)

        output = output.squeeze(1)

        loss = criterion(output, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
  avg_loss = total_loss/len(train_loader)
  print(f"Epoch: {epoch+1} Loss: {avg_loss}")

train_model(model, device, train_loader, criterion, optimizer, epochs = 15)

"""
## 🟪Testing
"""

test_dataset = prepare_movie_reviews(x_tets, y_test, vocab)

test_loader = DataLoader(test_dataset, batch_size = 32)

"""
###⌛Model Testing
"""

def test_model(model, device, test_loader, criterion):
  model.to(device)
  model.eval()

  total_loss = 0
  correct_predictions = 0
  total_predictions = 0

  with torch.no_grad():
    for inputs, labels,

def predict_probability(model, vocab, text, device, max_len=200):
    # Step 1: Tokenize the input text
    tokens = word_tokenize(text)

    # Step 2: Convert tokens to indices using the vocabulary
    indices = [vocab.get(token, 1) for token in tokens]

    # Pad the input if necessary
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))

    # Convert the indices to a tensor
    input_tensor = torch.tensor(indices).unsqueeze(0).to(device)  # Add batch dimension and move to the same device

    # Step 3: Run the model to get the prediction
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        output = model(input_tensor)  # Get the output logits

    # Step 4: Return the prediction result
    return output.item()

from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

text = """familiar -- in both negative and positive ways -- Ne Zha 2 is a distinct fantasy epic and a technical achievement that stands up to the best that Disney, DreamWorks, Aardman or Studio Ghibli can offer."""

print(f"Our model's positivity value: {predict_probability(model, vocab, text, device)}")

print(f"nltk model's positivity value: {(sia.polarity_scores(text)['compound']+1)/2}")
