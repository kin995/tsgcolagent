FROM python:3.12-slim

ARG WITH_JAVA=0
RUN if [ "$WITH_JAVA" = "1" ]; then \
        apt-get update && \
        apt-get install -y --no-install-recommends default-jdk-headless maven && \
        rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV TSGTEST_OLLAMA_BASE_URL=http://ollama:11434 \
    PYTHONUNBUFFERED=1
EXPOSE 8770

CMD ["python", "-m", "tsgtest", "serve", "--host", "0.0.0.0", "--port", "8770"]
