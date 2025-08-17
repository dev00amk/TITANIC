# Makefile for Titanic Enterprise ML Pipeline
.PHONY: help install install-dev setup test test-unit test-integration lint format type-check security clean docker-build docker-run pipeline train predict

# Default target
help: ## Show this help message
	@echo "Titanic Enterprise ML Pipeline - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Environment setup
install: ## Install production dependencies
	uv sync --no-dev

install-dev: ## Install development dependencies
	uv sync

setup: install-dev ## Set up development environment
	uv run pre-commit install
	uv run pre-commit install --hook-type commit-msg
	@echo "Development environment setup complete!"

# Testing
test: ## Run all tests
	uv run pytest tests/ -v --cov=titanic_enterprise --cov-report=html --cov-report=term

test-unit: ## Run unit tests only
	uv run pytest tests/unit/ -v -m "not integration"

test-integration: ## Run integration tests only
	uv run pytest tests/integration/ -v -m integration

test-watch: ## Run tests in watch mode
	uv run pytest-watch tests/ -- -v

# Code quality
lint: ## Run linting with ruff
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

format: ## Format code with ruff and black
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/
	uv run isort src/ tests/

type-check: ## Run type checking with mypy
	uv run mypy src/titanic_enterprise/

security: ## Run security checks
	uv run bandit -r src/ -f json -o security-report.json
	uv run safety check

pre-commit: ## Run pre-commit hooks on all files
	uv run pre-commit run --all-files

# Cleaning
clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf build/ dist/ htmlcov/ .coverage security-report.json

# Docker
docker-build: ## Build Docker image
	docker build -t titanic-enterprise:latest .

docker-run: ## Run Docker container
	docker run -it --rm -v $(PWD):/workspace titanic-enterprise:latest

docker-dev: ## Run development Docker container
	docker-compose up -d

docker-down: ## Stop development Docker containers
	docker-compose down

# ML Pipeline commands
pipeline: ## Run the complete ML pipeline
	uv run titanic-pipeline --stage all

train: ## Train the model
	uv run titanic-train

predict: ## Generate predictions
	uv run titanic-predict --model-path models/latest --input-file data/test.csv

# Data and model management
data-validate: ## Validate data quality
	uv run python -m titanic_enterprise.data.validator

data-preprocess: ## Preprocess data
	uv run python -m titanic_enterprise.data.preprocessor

model-info: ## Show model information
	uv run titanic-pipeline info

# DVC commands (when implemented)
dvc-init: ## Initialize DVC
	dvc init
	dvc remote add -d local ./dvc-storage

dvc-add-data: ## Add data to DVC tracking
	dvc add data/raw/train.csv data/raw/test.csv

dvc-pipeline: ## Run DVC pipeline
	dvc repro

# MLflow commands
mlflow-ui: ## Start MLflow UI
	uv run mlflow ui --host 0.0.0.0 --port 5000

mlflow-server: ## Start MLflow tracking server
	uv run mlflow server --host 0.0.0.0 --port 5000 --default-artifact-root ./mlruns

# Development utilities
jupyter: ## Start Jupyter lab
	uv run jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

notebook-clean: ## Clean notebook outputs
	uv run nbstripout notebooks/*.ipynb

# CI/CD simulation
ci-test: lint type-check security test ## Run CI pipeline locally

# Project information
info: ## Show project information
	@echo "Project: Titanic Enterprise ML Pipeline"
	@echo "Version: 1.0.0"
	@echo "Python: $(shell python --version)"
	@echo "UV: $(shell uv --version)"
	@echo ""
	@echo "Dependencies:"
	@uv tree --depth 1

# Environment variables for development
export PYTHONPATH := $(PWD)/src:$(PYTHONPATH)
export MLFLOW_TRACKING_URI := file://$(PWD)/mlruns
export DVC_CACHE_DIR := $(PWD)/.dvc/cache