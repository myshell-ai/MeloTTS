FROM python:3.10-slim
WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y \
    build-essential libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
    
RUN pip install -e .
RUN pip install fastapi uvicorn
RUN python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"
RUN python -c "import nltk; nltk.download('averaged_perceptron_tagger')"
RUN python -c "import nltk; nltk.download('punkt')"
RUN python -c "import nltk; nltk.download('universal_tagset')"
RUN python -m unidic download
RUN python melo/init_downloads.py

EXPOSE 8888
#CMD ["python", "./melo/app.py", "--host", "0.0.0.0", "--port", "8888"]
CMD [ "python", "server.py" ]