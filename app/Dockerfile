FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Hugging Face enforces running as a non-root user (UID 1000)
RUN useradd -m -u 1000 user
WORKDIR /app

# Install compilation essentials
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Fix folder ownership permissions for downloading NLTK models
RUN mkdir -p /home/user/nltk_data && chown -R user:user /home/user
ENV NLTK_DATA=/home/user/nltk_data

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK resources so your app launches fast
RUN python -m nltk.downloader -d /home/user/nltk_data stopwords punkt punkt_tab wordnet

COPY . .
RUN chown -R user:user /app

USER user

# Hugging Face requires port 7860
EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]