🎬 LSTM Movie Review Sentiment Analysis
This project uses machine learning to figure out if movie reviews are positive or negative. It uses natural language processing (NLP) tools and a type of neural network called LSTM (Long Short-Term Memory) to do this.

📚 What You’ll Learn
How to use the NLTK library to work with text

How to preprocess and clean up data

How to build an LSTM model using TensorFlow/Keras

How to train and test your model

How to use sentiment analysis to classify reviews

🛠️ Technologies Used
Python 🐍

NLTK (Natural Language Toolkit)

TensorFlow / Keras

Matplotlib (for graphs and charts)

VADER (for quick sentiment checking)

🧠 How It Works
Text is broken down using NLTK (into words and sentences).

We remove stop words like "the", "is", etc. that don’t add meaning.

We use stemming and lemmatization to simplify words (e.g., "running" → "run").

We check the frequency of words and visualize them.

Then we use LSTM to train a model on sample reviews.

Finally, we test it to see how good it is at predicting whether a review is happy (positive) or sad (negative).

🚀 How to Run
Make sure you have Python installed.

Install the required libraries by running:

bash
Copy
Edit
pip install nltk tensorflow matplotlib
Open the .py file or copy the code into a Jupyter Notebook or Google Colab.

Run the code step-by-step to see how everything works.

📁 File Overview
LSTM_Movie_Review_Sentiment_Analysis.py: The main project file. It contains all the code from importing libraries to building and testing the LSTM model.

🎯 Goals
Understand the basics of NLP

Learn how to work with LSTM models

Build a simple sentiment analysis system for movie reviews
