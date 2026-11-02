PY ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

.PHONY: install dev coordinator worker test lint docker screenshots dataset clean

install:          ## install runtime + dev dependencies
	$(PY) -m pip install -r requirements-dev.txt

dev:              ## coordinator + 3 local workers
	./scripts/run_local_cluster.sh

coordinator:      ## coordinator only
	$(PY) -m minicluster.coordinator

worker:           ## one worker: make worker COORD=192.168.1.10 NAME=PC-01
	$(PY) -m minicluster.worker --coordinator $(or $(COORD),127.0.0.1) --name $(or $(NAME),$(shell hostname))

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check minicluster tests scripts

docker:
	docker compose up --build

screenshots:      ## regenerate docs/images (needs playwright)
	$(PY) scripts/capture_screenshots.py

dataset:
	$(PY) scripts/generate_dataset.py --rows 100000 --out data/student_data.csv

clean:
	rm -rf data .pytest_cache .ruff_cache build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
