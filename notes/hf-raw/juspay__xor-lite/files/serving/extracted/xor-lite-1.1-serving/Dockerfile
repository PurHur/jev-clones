FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c

# git installs the pinned PEFT commit; gcc and libc6-dev let Triton build its CUDA helper
# when torch compiles a kernel on the first request.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git gcc libc6-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu130 \
      -r /app/requirements.txt

COPY --chown=10001:10001 app.py smoke_test.py /app/
COPY --chown=10001:10001 strands_decider /app/strands_decider
USER 10001:10001
# The container root is read-only; every cache goes to the /tmp tmpfs.
ENV PYTHONPATH=/app HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HOME=/tmp HF_HOME=/tmp/hf \
    TRITON_CACHE_DIR=/tmp/triton TORCHINDUCTOR_CACHE_DIR=/tmp/inductor
EXPOSE 30002
HEALTHCHECK --interval=10s --timeout=3s --start-period=120s --retries=12 CMD ["python3", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:30002/health', timeout=2).read()"]
ENTRYPOINT ["python3", "/app/app.py"]
