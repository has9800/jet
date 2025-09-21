.PHONY: build-trainer build-serve test run-eval run-train

build-trainer:
	docker build -f Docker/train.Dockerfile -t jet-trainer:latest .

build-serve:
	docker build -f Docker/server-vllm.Dockerfile -t server-vllm:latest .

test:
	python3 -m venv .venv && . .venv/bin/activate && pip install -e . && pip install pytest mlflow fastapi uvicorn && pytest -q

run-eval:
	mlflow run . -e evaluate -P model_dir_or_id=sshleifer/tiny-gpt2 --env-manager=local

run-train:
	mlflow run . -e train -P model=sshleifer/tiny-gpt2 -P dataset_id=databricks/databricks-dolly-15k -P engine=auto --env-manager=local
