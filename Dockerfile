FROM python:3.9-slim
RUN apt-get update && apt-get install -y \
    build-essential libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install -e .
RUN python -m unidic download
RUN python melo/init_downloads.py

# Copy entrypoint script and make it executable
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Set the entrypoint script
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]