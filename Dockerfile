FROM python:3.10-slim

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# Set working directory
WORKDIR /app

# Install required system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK resources
RUN python -m nltk.downloader punkt stopwords wordnet

# Copy project files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Render uses dynamic ports
ENV PORT=8501

# Start Streamlit app
CMD ["sh", "-c", "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"]
