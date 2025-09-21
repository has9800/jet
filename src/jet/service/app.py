import os, subprocess, shlex
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class TrainSpec(BaseModel):
    model: str
    dataset_id: str
    split: str = "train"
    text_field: str = "text"
    epochs: int = 1
    per_device_batch: int = 1
    grad_accum: int = 8
    lr: float = 2e-4
    engine: str = "auto"
    output_dir: str = "/workspace/outputs/run"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/train")
def train(spec: TrainSpec):
    cmd = (
        f"mlflow run . -e train "
        f"-P model={shlex.quote(spec.model)} -P dataset_id={shlex.quote(spec.dataset_id)} "
        f"-P split={shlex.quote(spec.split)} -P text_field={shlex.quote(spec.text_field)} "
        f"-P epochs={spec.epochs} -P per_device_batch={spec.per_device_batch} "
        f"-P grad_accum={spec.grad_accum} -P lr={spec.lr} -P engine={shlex.quote(spec.engine)} "
        f"-P output_dir={shlex.quote(spec.output_dir)}"
    )
    try:
        p = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        return {"ok": True, "stdout": p.stdout}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail={"stderr": e.stderr})

class DeploySpec(BaseModel):
    model: str
    api_key: str = "token-abc123"
    port: int = 8000
    lora_adapters: dict | None = None

@app.post("/deploy")
def deploy(spec: DeploySpec):
    adapter_args = ""
    if spec.lora_adapters:
        joined = " ".join([f"--lora-modules {k}={v}" for k, v in spec.lora_adapters.items()])
        adapter_args = f" {joined}"
    cmd = (
        f"docker run --rm --gpus all -p {spec.port}:8000 server-vllm:latest "
        f"python3 -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8000 "
        f"--api-key {shlex.quote(spec.api_key)} --model {shlex.quote(spec.model)}{adapter_args}"
    )
    return {"command": cmd}
