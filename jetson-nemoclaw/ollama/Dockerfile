# The sample's own model runtime, so a clean device needs nothing pre-installed.
FROM ollama/ollama:latest

# Models live on the app's persist volume, so they survive redeploys and a first run is
# paid for once. Keep the model resident: a cold load costs roughly 24 seconds on an AGX
# class board against about 2 warm, and that delay lands on the first question.
ENV OLLAMA_MODELS=/models \
    OLLAMA_HOST=0.0.0.0:11434 \
    OLLAMA_KEEP_ALIVE=24h \
    OLLAMA_NUM_PARALLEL=4

COPY start.sh /usr/local/bin/model-runtime-start
RUN chmod +x /usr/local/bin/model-runtime-start

EXPOSE 11434
ENTRYPOINT ["/bin/bash", "/usr/local/bin/model-runtime-start"]
