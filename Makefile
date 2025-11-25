.PHONY: help install install-dev clean lint format test run docs

help:
	@echo "RadarTargetingSystem - Development Commands"
	@echo "============================================"
	@echo "make install        - Install dependencies from requirements.txt"
	@echo "make install-dev    - Install with development dependencies"
	@echo "make run            - Run the Streamlit dashboard"
	@echo "make test           - Run pytest tests"
	@echo "make lint           - Run code quality checks (flake8, mypy, pylint)"
	@echo "make format         - Auto-format code with black and isort"
	@echo "make clean          - Remove build artifacts and cache files"
	@echo "make docs           - Build documentation (if available)"

install:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev]"

venv:
	python -m venv .venv
	@echo "Virtual environment created. Activate with: source .venv/bin/activate"

run:
	streamlit run app.py

test:
	PYTHONPATH=src pytest -v

lint:
	flake8 src app.py
	mypy src app.py || true
	pylint src app.py || true

format:
	black src app.py
	isort src app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name *.egg-info -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name .coverage -delete
	rm -rf build dist .eggs *.egg-info

docs:
	@echo "Documentation build not yet configured"

.DEFAULT_GOAL := help
