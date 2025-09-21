FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY . /workspace
RUN python3 -m pip install --upgrade pip && \
    pip install "torch>=2.3" "transformers>=4.42,<4.49" "datasets>=2.18" \
                "evaluate>=0.4.3" "trl>=0.9,<0.13" "peft>=0.10" \
                "bitsandbytes>=0.43" "unsloth>=0.9" "mlflow>=2.12" && \
    pip install -e .
