FROM python:3.11

# Install Ollama CLI
RUN curl -s https://ollama.com/install.sh | bash

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy FastAPI app
COPY . .

# Pull Ollama models (replace with your actual model names)
RUN ollama pull nomic-embed-text
RUN ollama pull deepseek-r1:1.5b

# Expose FastAPI port
EXPOSE 8000

# Start Ollama server in background, then start FastAPI
CMD ollama serve & uvicorn main:app --host 0.0.0.0 --port 8000
