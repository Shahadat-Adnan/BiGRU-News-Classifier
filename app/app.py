
import streamlit as st
import pickle
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# CUSTOM CSS

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.stTextArea textarea {
    background-color: #262730;
    color: white;
    border-radius: 10px;
    font-size: 16px;
}

.stButton>button {
    width: 100%;
    background-color: #FF4B4B;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    height: 3em;
    border: none;
}

.stButton>button:hover {
    background-color: #ff2e2e;
    color: white;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
}

.real-news {
    background-color: #0f5132;
    color: #d1e7dd;
}

.fake-news {
    background-color: #842029;
    color: #f8d7da;
}

.uncertain {
    background-color: #664d03;
    color: #fff3cd;
}

.score-box {
    background-color: #1e1e1e;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-size: 18px;
    color: white;
}

</style>
""", unsafe_allow_html=True)

import nltk

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# LOAD MODEL

@st.cache_resource
def load_fake_news_model():

    model = load_model('model_news_claf.h5')

    return model

# LOAD TOKENIZER

@st.cache_resource
def load_tokenizer():

    with open('tokenizer.pkl','rb') as file:
        tokenizer = pickle.load(file)

    return tokenizer

model = load_fake_news_model()

tokenizer = load_tokenizer()

# PARAMETERS

max_len = 64

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ============================================================
# PREPROCESSING FUNCTION
# ============================================================

def preprocess_text(text):

    text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+'," ",text)
    text = re.sub(r'@\w+'," ",text)
    text = re.sub(r'[^a-zA-Z\s]',"",text)
    text = re.sub(r'\d+'," ",text)
    text = re.sub(r'\s+'," ",text).strip()

    # Tokenization
    tokens = word_tokenize(text)

    # Stopword Removal + Lemmatization
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    # Final processed text
    processed_text = ' '.join(processed_tokens)

    return processed_text

# PREDICTION FUNCTION

def predict_news(news_text):

    # Preprocessing
    processed_text = preprocess_text(news_text)

    # Convert to sequence
    sequence = tokenizer.texts_to_sequences([processed_text])

    # Padding
    padded_sequence = pad_sequences(sequence,maxlen=max_len,padding='post')

    # Prediction
    prediction = model.predict(padded_sequence,verbose=0)[0][0]

    return prediction

# UI HEADER

st.markdown(
    "<h1 style='text-align: center;'>📰 Fake News Detection System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; font-size:18px;'>"
    "Deep Learning Powered Fake News Classifier using BiGRU"
    "</p>",
    unsafe_allow_html=True
)

st.divider()

# ============================================================
# TEXT INPUT
# ============================================================

news_input = st.text_area(
    "Enter News Article/Text",
    height=250,
    placeholder="Paste any news article here..."
)

# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button("Analyze News"):

    # Empty Input Check
    if news_input.strip() == "":

        st.warning(
            "Please enter some news text."
        )

    else:

        prediction = predict_news(
            news_input
        )

        fake_probability = prediction

        real_probability = 1 - prediction

        # ====================================================
        # RESULT LOGIC
        # ====================================================

        if prediction >= 0.75:

            result = "🚨 FAKE NEWS"

            result_class = "fake-news"

        elif prediction <= 0.25:

            result = "✅ REAL NEWS"

            result_class = "real-news"

        else:

            result = "⚠️ UNCERTAIN"

            result_class = "uncertain"

        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.markdown(
            f"""
            <div class="result-box {result_class}">
                {result}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ====================================================
        # CONFIDENCE SCORES
        # ====================================================

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                f"""
                <div class="score-box">
                    <h3>Fake Probability</h3>
                    <h2>{fake_probability:.4f}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:

            st.markdown(
                f"""
                <div class="score-box">
                    <h3>Real Probability</h3>
                    <h2>{real_probability:.4f}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ====================================================
        # PROCESSED TEXT
        # ====================================================

        with st.expander("View Processed Text"):

            st.write(
                preprocess_text(news_input)
            )

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("ℹ️ About")

    st.write("""
    This application uses a Deep Learning
    Bidirectional GRU model trained on
    Fake and Real News datasets.
    """)

    st.write("""
    ### Model Pipeline
    - NLP Preprocessing
    - Tokenization
    - Padding
    - Embedding Layer
    - Bidirectional GRU
    - Dense Classification
    """)

    st.write("""
    ### Threshold Logic
    - ≥ 0.75 → Fake
    - ≤ 0.25 → Real
    - Otherwise → Uncertain
    """)

    st.divider()

    st.write("Developed using:")
    st.write("- TensorFlow")
    st.write("- Streamlit")
    st.write("- NLTK")

# ============================================================
# FOOTER
# ============================================================

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown(
    """
    <p style='text-align:center; color:gray;'>
    Fake News Detection System • Deep Learning + NLP
    </p>
    """,
    unsafe_allow_html=True
)