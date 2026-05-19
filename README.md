# Fake News Detection using Deep Learning

A fake news detection web app built using TensorFlow, NLP, and Streamlit.

The model uses a Bidirectional GRU network trained on fake and real news articles to classify whether a news article is likely fake or real.

---

## Features

- NLP text preprocessing
- Deep Learning based classification
- Bidirectional GRU model
- Streamlit web app
- Real-time predictions
- Confidence score display

---

## Tech Stack

- Python
- TensorFlow / Keras
- NLTK
- Streamlit
- Scikit-learn

---

## Model Pipeline

```text
Text Preprocessing
↓
Tokenization
↓
Padding
↓
Embedding Layer
↓
Bidirectional GRU
↓
Dense Output Layer
```

---

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

---

## Project Structure

```text
├── app.py
├── training.ipynb
├── prediction.ipynb
├── better_fake_news_model.h5
├── better_tokenizer.pkl
├── Fake.csv
├── True.csv
└── README.md
```

---

## Notes

The project focuses on reducing:
- data leakage
- overfitting
- tokenizer leakage
- preprocessing mismatch

to improve real-world generalization.

---

## Author

Adnan Shahadat
