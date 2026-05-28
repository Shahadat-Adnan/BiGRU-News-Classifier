# Fake News Detection System
A Deep Learning-powered Natural Language Processing (NLP) application that classifies news articles as Real, Fake, or Uncertain. This project utilizes a Bidirectional Gated Recurrent Unit (BiGRU) neural network built with TensorFlow/Keras and features an interactive web interface powered by Streamlit.

🔗 Live Application Link: 👉 (https://datageek7001-news-authenticator.hf.space/) 👈

# 🚀 Features
## Deep Learning Backend: Powered by a Bidirectional GRU network optimized for text sequence classification.

## Robust NLP Pipeline: Includes lowercasing, regex cleaning (URL, handle, and digit removal), punctuation stripping, NLTK tokenization, stopword removal, and WordNet lemmatization.

## Intuitive UI: A clean, dark-themed Streamlit user interface featuring a real-time analytics panel.

# 🧠 Model Architecture & Performance
## The model processes text sequences of up to 64 tokens using a sequential architecture:
## Embedding Layer: Maps an 8,000-word vocabulary into a 32-dimensional dense vector space.
## Bidirectional GRU Layer: Captures forward and backward contextual relationships using 32 memory cells.
## Dropout Layer (0.3): Regularizes the network to prevent overfitting.
## Dense Output Layer: Uses a sigmoid activation function for final probability output.

# 📊 Tech Stack
## Machine Learning & NLP
- TensorFlow / Keras
- Scikit-learn
- NLTK
- NumPy
- Pandas
## Frontend & Deployment
- Streamlit
- Docker
- Hugging Face Spaces
