.PHONY: setup dev lint test up down bench bench-diffusion bench-kernels report grafana

setup:
	UV_PYTHON_INSTALL_DIR="$(PWD)/.uv_python" UV_CACHE_DIR="$(PWD)/.uv_cache" ./uv venv --python 3.11
	UV_PYTHON_INSTALL_DIR="$(PWD)/.uv_python" UV_CACHE_DIR="$(PWD)/.uv_cache" ./uv pip install -e .[dev]

dev:
	.venv/bin/uvicorn src.gateway.app:app --reload --port 8000

lint:
	.venv/bin/ruff check --fix .
	.venv/bin/ruff format .
	XDG_CACHE_HOME="$(PWD)/.cache" .venv/bin/pyright

test:
	.venv/bin/pytest tests/unit/

test-integration:
	.venv/bin/pytest tests/integration/

up:
	docker compose up -d gateway prometheus grafana

down:
	docker compose down

bench:
	.venv/bin/python scripts/run_bench.py

bench-diffusion:
	.venv/bin/python scripts/run_bench_diffusion.py

bench-kernels:
	.venv/bin/python scripts/run_kernel_bench.py

report:
	.venv/bin/python -m src.bench.report

grafana:
	open http://localhost:3000
