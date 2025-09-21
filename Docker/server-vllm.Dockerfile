FROM vllm/vllm-openai:latest
RUN printf '#!/bin/sh\nnvidia-smi || exit 1\npython3 - <<PY\nimport torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)\nPY\n' > /usr/local/bin/gpu-health && chmod +x /usr/local/bin/gpu-health
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD /usr/local/bin/gpu-health || exit 1
