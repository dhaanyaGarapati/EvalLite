FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

# Installing pip packages
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

# Streamlit configuration
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ENABLECORS=true
ENV STREAMLIT_SERVER_ENABLEXSRSFPROTECTION=false

CMD ["streamlit", "run", "app.py"]
