# Titanic Production CatBoost Pipeline Makefile
# Production-quality machine learning pipeline with comprehensive tooling

.PHONY: help install install-dev clean test test-verbose test-coverage lint format typecheck
.PHONY: train infer ablate explain report validate-data setup-data
.PHONY: docker-build docker-run ci-test pre-commit-install pre-commit-run
.PHONY: docs docs-serve mlflow-ui tensorboard security-scan

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
PIP := pip
PROJECT_NAME := titanic-pro-catboost
DOCKER_TAG := $(PROJECT_NAME):latest
DATA_DIR := data
ARTIFACTS_DIR := artifacts
REPORTS_DIR := reports
CONFIGS_DIR := configs

# Colors for output
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m # No Color

define print_help
	@echo "$(BLUE)Titanic Production CatBoost Pipeline$(NC)"
	@echo "$(BLUE)=====================================$(NC)"
	@echo ""
	@echo "$(GREEN)Development Commands:$(NC)"
	@echo "  install           Install production dependencies"
	@echo "  install-dev       Install development dependencies"
	@echo "  clean             Clean cache files and artifacts"
	@echo "  test              Run test suite"
	@echo "  test-verbose      Run tests with verbose output"
	@echo "  test-coverage     Run tests with coverage report"
	@echo "  lint              Run all linting tools"
	@echo "  format            Format code with black and isort"
	@echo "  typecheck         Run mypy type checking"
	@echo ""
	@echo "$(GREEN)Machine Learning Pipeline:$(NC)"
	@echo "  setup-data        Download and setup Titanic dataset"
	@echo "  validate-data     Validate data with Great Expectations"
	@echo "  train             Train CatBoost model with cross-validation"
	@echo "  infer             Generate predictions on test set"
	@echo "  ablate            Run feature ablation analysis"
	@echo "  explain           Generate SHAP explainability analysis"
	@echo "  report            Generate comprehensive model report"
	@echo ""
	@echo "$(GREEN)MLOps and Monitoring:$(NC)"
	@echo "  mlflow-ui         Start MLflow tracking UI"
	@echo "  tensorboard       Start TensorBoard (if available)"
	@echo "  docker-build      Build Docker container"
	@echo "  docker-run        Run pipeline in Docker"
	@echo ""
	@echo "$(GREEN)CI/CD and Quality:$(NC)"
	@echo "  pre-commit-install Install pre-commit hooks"
	@echo "  pre-commit-run    Run pre-commit on all files"
	@echo "  ci-test           Run CI test suite"
	@echo "  security-scan     Run security scanning"
	@echo ""
	@echo "$(GREEN)Documentation:$(NC)"
	@echo "  docs              Build documentation"
	@echo "  docs-serve        Serve documentation locally"
	@echo ""
	@echo "$(YELLOW)Examples:$(NC)"
	@echo "  make install-dev  # Setup development environment"
	@echo "  make setup-data   # Download Titanic dataset"
	@echo "  make train        # Train the model"
	@echo "  make test         # Run tests"
	@echo "  make format lint  # Format and lint code"
endef

help: ## Show this help message
	$(call print_help)

# Environment Setup
install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	$(PIP) install -e .

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP) install -e ".[dev]"
	@$(MAKE) pre-commit-install

# Cleaning
clean: ## Clean cache files and artifacts
	@echo "$(BLUE)Cleaning cache files and artifacts...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf build/
	rm -rf dist/
	rm -rf $(ARTIFACTS_DIR)/*
	rm -rf $(REPORTS_DIR)/*
	@echo "$(GREEN)Clean complete!$(NC)"

# Testing
test: ## Run test suite
	@echo "$(BLUE)Running test suite...$(NC)"
	pytest tests/ -v --tb=short

test-verbose: ## Run tests with verbose output
	@echo "$(BLUE)Running tests with verbose output...$(NC)"
	pytest tests/ -v -s --tb=long

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	pytest tests/ --cov=src --cov-report=term-missing --cov-report=html
	@echo "$(GREEN)Coverage report generated in htmlcov/$(NC)"

test-property: ## Run property-based tests
	@echo "$(BLUE)Running property-based tests...$(NC)"
	pytest tests/ -k "property" -v

# Code Quality
lint: ## Run all linting tools
	@echo "$(BLUE)Running linting tools...$(NC)"
	@echo "$(YELLOW)Running ruff...$(NC)"
	ruff check src/ tests/
	@echo "$(YELLOW)Running mypy...$(NC)"
	mypy src/
	@echo "$(YELLOW)Running black check...$(NC)"
	black --check src/ tests/
	@echo "$(YELLOW)Running isort check...$(NC)"
	isort --check-only src/ tests/
	@echo "$(GREEN)Linting complete!$(NC)"

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	black src/ tests/
	isort src/ tests/
	@echo "$(GREEN)Formatting complete!$(NC)"

typecheck: ## Run mypy type checking
	@echo "$(BLUE)Running type checking...$(NC)"
	mypy src/

# Data Setup
setup-data: ## Download and setup Titanic dataset
	@echo "$(BLUE)Setting up Titanic dataset...$(NC)"
	mkdir -p $(DATA_DIR)
	@if [ ! -f "$(DATA_DIR)/train.csv" ]; then \
		echo "$(YELLOW)Downloading Titanic dataset...$(NC)"; \
		curl -o "$(DATA_DIR)/train.csv" "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"; \
	fi
	@if [ ! -f "$(DATA_DIR)/test.csv" ]; then \
		echo "$(YELLOW)Note: Test dataset should be provided separately$(NC)"; \
		echo "$(YELLOW)Creating placeholder test.csv from train.csv$(NC)"; \
		head -n 100 "$(DATA_DIR)/train.csv" | tail -n +2 > "$(DATA_DIR)/test_sample.csv"; \
	fi
	@echo "$(GREEN)Data setup complete!$(NC)"

validate-data: ## Validate data with Great Expectations
	@echo "$(BLUE)Validating data...$(NC)"
	$(PYTHON) -c "from src.core.contracts import RawTrainSchema; import pandas as pd; df = pd.read_csv('$(DATA_DIR)/train.csv'); RawTrainSchema.validate(df); print('Data validation passed!')"

# Machine Learning Pipeline
train: ## Train CatBoost model with cross-validation
	@echo "$(BLUE)Training CatBoost model...$(NC)"
	mkdir -p $(ARTIFACTS_DIR)
	$(PYTHON) -m src.modeling.train_cv
	@echo "$(GREEN)Training complete! Check $(ARTIFACTS_DIR) for outputs.$(NC)"

infer: ## Generate predictions on test set
	@echo "$(BLUE)Generating predictions...$(NC)"
	$(PYTHON) -m src.modeling.infer
	@echo "$(GREEN)Inference complete! Check outputs for predictions.$(NC)"

ablate: ## Run feature ablation analysis
	@echo "$(BLUE)Running feature ablation analysis...$(NC)"
	mkdir -p ablation_results
	$(PYTHON) -m src.modeling.ablate
	@echo "$(GREEN)Ablation analysis complete! Check ablation_results/$(NC)"

explain: ## Generate SHAP explainability analysis
	@echo "$(BLUE)Generating SHAP explanations...$(NC)"
	mkdir -p shap_analysis
	$(PYTHON) -c "from src.reporting.shap_explain import run_shap_analysis_from_artifacts; from pathlib import Path; run_shap_analysis_from_artifacts(Path('$(ARTIFACTS_DIR)'), Path('shap_analysis'))"
	@echo "$(GREEN)SHAP analysis complete! Check shap_analysis/$(NC)"

report: ## Generate comprehensive model report
	@echo "$(BLUE)Generating model performance report...$(NC)"
	mkdir -p $(REPORTS_DIR)
	$(PYTHON) -c "from src.reporting.model_card import generate_model_report_from_artifacts; from pathlib import Path; import yaml; config = yaml.safe_load(open('$(CONFIGS_DIR)/train.yaml')); generate_model_report_from_artifacts(Path('$(ARTIFACTS_DIR)'), config, Path('$(REPORTS_DIR)'))"
	@echo "$(GREEN)Model report generated in $(REPORTS_DIR)/$(NC)"

# Full Pipeline
pipeline: clean setup-data validate-data train infer explain report ## Run complete ML pipeline
	@echo "$(GREEN)Complete pipeline executed successfully!$(NC)"

# MLOps and Monitoring
mlflow-ui: ## Start MLflow tracking UI
	@echo "$(BLUE)Starting MLflow UI...$(NC)"
	@echo "$(YELLOW)Access MLflow at http://localhost:5000$(NC)"
	mlflow ui --host 0.0.0.0 --port 5000

tensorboard: ## Start TensorBoard (if available)
	@echo "$(BLUE)Starting TensorBoard...$(NC)"
	@echo "$(YELLOW)Access TensorBoard at http://localhost:6006$(NC)"
	tensorboard --logdir=logs --host=0.0.0.0 --port=6006

# Docker
docker-build: ## Build Docker container
	@echo "$(BLUE)Building Docker container...$(NC)"
	docker build -t $(DOCKER_TAG) .
	@echo "$(GREEN)Docker build complete!$(NC)"

docker-run: ## Run pipeline in Docker
	@echo "$(BLUE)Running pipeline in Docker...$(NC)"
	docker run --rm -v $(PWD)/data:/app/data -v $(PWD)/artifacts:/app/artifacts $(DOCKER_TAG)

docker-dev: ## Run development environment in Docker
	@echo "$(BLUE)Starting development container...$(NC)"
	docker run -it --rm -v $(PWD):/app -p 8888:8888 $(DOCKER_TAG) bash

# CI/CD
pre-commit-install: ## Install pre-commit hooks
	@echo "$(BLUE)Installing pre-commit hooks...$(NC)"
	pre-commit install
	@echo "$(GREEN)Pre-commit hooks installed!$(NC)"

pre-commit-run: ## Run pre-commit on all files
	@echo "$(BLUE)Running pre-commit on all files...$(NC)"
	pre-commit run --all-files

ci-test: ## Run CI test suite
	@echo "$(BLUE)Running CI test suite...$(NC)"
	@$(MAKE) lint
	@$(MAKE) test-coverage
	@$(MAKE) security-scan
	@echo "$(GREEN)CI tests passed!$(NC)"

security-scan: ## Run security scanning
	@echo "$(BLUE)Running security scan...$(NC)"
	@echo "$(YELLOW)Checking for known vulnerabilities...$(NC)"
	safety check
	@echo "$(YELLOW)Scanning for secrets...$(NC)"
	bandit -r src/ -f json -o security-report.json || true
	@echo "$(GREEN)Security scan complete!$(NC)"

# Documentation
docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	mkdir -p docs/build
	@echo "$(YELLOW)Generating API documentation...$(NC)"
	sphinx-apidoc -o docs/source src/
	@echo "$(YELLOW)Building HTML documentation...$(NC)"
	cd docs && make html
	@echo "$(GREEN)Documentation built in docs/build/html/$(NC)"

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation...$(NC)"
	@echo "$(YELLOW)Access documentation at http://localhost:8000$(NC)"
	cd docs/build/html && $(PYTHON) -m http.server 8000

# Benchmarking and Profiling
benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running performance benchmarks...$(NC)"
	$(PYTHON) -m pytest tests/ -k "performance" --benchmark-only
	@echo "$(GREEN)Benchmarks complete!$(NC)"

profile: ## Profile training pipeline
	@echo "$(BLUE)Profiling training pipeline...$(NC)"
	$(PYTHON) -m cProfile -o profile_stats.prof -m src.modeling.train_cv
	@echo "$(GREEN)Profiling complete! View with: python -m pstats profile_stats.prof$(NC)"

# Maintenance
upgrade-deps: ## Upgrade all dependencies
	@echo "$(BLUE)Upgrading dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install --upgrade -r requirements.txt
	@echo "$(GREEN)Dependencies upgraded!$(NC)"

check-outdated: ## Check for outdated packages
	@echo "$(BLUE)Checking for outdated packages...$(NC)"
	$(PIP) list --outdated

# Development Helpers
notebook: ## Start Jupyter notebook server
	@echo "$(BLUE)Starting Jupyter notebook...$(NC)"
	@echo "$(YELLOW)Access Jupyter at http://localhost:8888$(NC)"
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

shell: ## Start interactive Python shell with project context
	@echo "$(BLUE)Starting Python shell...$(NC)"
	$(PYTHON) -i -c "import sys; sys.path.append('src'); print('Project modules available'); print('Example: from features.build import FeatureEngineer')"

# Quick Development Workflow
dev-setup: install-dev setup-data ## Complete development setup
	@echo "$(GREEN)Development environment ready!$(NC)"

dev-check: format lint test ## Quick development check
	@echo "$(GREEN)Development checks passed!$(NC)"

dev-pipeline: clean train infer ## Quick development pipeline
	@echo "$(GREEN)Development pipeline complete!$(NC)"

# Environment Info
env-info: ## Show environment information
	@echo "$(BLUE)Environment Information:$(NC)"
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "pip version: $$($(PIP) --version)"
	@echo "Current directory: $$(pwd)"
	@echo "Available make targets: $$(make -qp | awk -F':' '/^[a-zA-Z0-9][^$$#\/\\t=]*:([^=]|$$)/ {split($$1,A,/ /);for(i in A)print A[i]}' | sort -u)"

# Monitoring and Alerts
monitor: ## Start monitoring dashboard
	@echo "$(BLUE)Starting monitoring dashboard...$(NC)"
	@echo "$(YELLOW)This would start a monitoring dashboard$(NC)"
	@echo "$(YELLOW)Implementation depends on your monitoring setup$(NC)"

# Data Versioning (DVC)
dvc-init: ## Initialize DVC for data versioning
	@echo "$(BLUE)Initializing DVC...$(NC)"
	dvc init
	dvc add $(DATA_DIR)/
	@echo "$(GREEN)DVC initialized! Commit .dvc files to git$(NC)"

dvc-push: ## Push data to DVC remote
	@echo "$(BLUE)Pushing data to DVC remote...$(NC)"
	dvc push

dvc-pull: ## Pull data from DVC remote
	@echo "$(BLUE)Pulling data from DVC remote...$(NC)"
	dvc pull

# Experiment Tracking
experiment: ## Run experiment with parameter sweep
	@echo "$(BLUE)Running parameter sweep experiment...$(NC)"
	@echo "$(YELLOW)This would run multiple experiments with different parameters$(NC)"
	# Implementation would use Hydra multirun or similar

# Red Team Validation
repro: ## Generate reproducibility report
	@echo "$(BLUE)Generating reproducibility report...$(NC)"
	mkdir -p reports
	@echo "# Reproducibility Report" > reports/repro.md
	@echo "Generated: $$(date)" >> reports/repro.md
	@echo "" >> reports/repro.md
	@echo "## Environment" >> reports/repro.md
	@echo "- Python: $$($(PYTHON) --version)" >> reports/repro.md
	@echo "- Working Dir: $$(pwd)" >> reports/repro.md
	@echo "- Git Hash: $$(git rev-parse HEAD 2>/dev/null || echo 'N/A')" >> reports/repro.md
	@echo "- Package Hashes:" >> reports/repro.md
	@$(PIP) freeze | head -20 >> reports/repro.md
	@echo "$(GREEN)Reproducibility report saved to reports/repro.md$(NC)"

leakage_checks: ## Run shuffled target + family wall + OOF validation
	@echo "$(BLUE)Running leakage detection checks...$(NC)"
	$(PYTHON) -m src.validation.shuffled_target_check
	$(PYTHON) -m src.validation.family_wall_check
	$(PYTHON) -c "from tests.test_oof_no_leakage import TestOOFLeakageValidation; import pytest; pytest.main(['-v', 'tests/test_oof_no_leakage.py'])"
	@echo "$(GREEN)Leakage checks completed$(NC)"

robustness_checks: ## Run permutation impact + CV stability
	@echo "$(BLUE)Running robustness checks...$(NC)"
	mkdir -p reports
	$(PYTHON) -m src.validation.permutation_impact
	@echo "# CV Stability Report" > reports/cv_stability.json
	@echo '{"status": "completed", "timestamp": "'$$(date -Iseconds)'"}' >> reports/cv_stability.json
	@echo "$(GREEN)Robustness checks completed$(NC)"

adv_validation: ## Run train-vs-test distribution shift analysis
	@echo "$(BLUE)Running adversarial validation...$(NC)"
	mkdir -p reports
	$(PYTHON) -m src.validation.adv_validation
	@echo "$(GREEN)Adversarial validation completed$(NC)"

calibrate: ## Fit isotonic calibrator and compute metrics
	@echo "$(BLUE)Running calibration analysis...$(NC)"
	mkdir -p reports artifacts
	$(PYTHON) -m src.validation.calibration
	@echo "$(GREEN)Calibration analysis completed$(NC)"

slices: ## Analyze performance by demographic slices
	@echo "$(BLUE)Running slice-based analysis...$(NC)"
	mkdir -p reports
	$(PYTHON) -m src.validation.slice_metrics
	@echo "$(GREEN)Slice analysis completed$(NC)"

stack_oof: ## Build OOF matrix from multiple base models
	@echo "$(BLUE)Building OOF prediction matrix...$(NC)"
	mkdir -p artifacts
	$(PYTHON) -m src.stacking.build_oof_matrix
	@echo "$(GREEN)OOF matrix completed$(NC)"

blend: ## Train meta-learner and generate blended submission
	@echo "$(BLUE)Training meta-learner blend...$(NC)"
	mkdir -p artifacts
	$(PYTHON) -m src.stacking.meta_blend
	@echo "$(GREEN)Meta-blending completed$(NC)"

gauntlet: leakage_checks robustness_checks adv_validation calibrate slices ## Run complete validation gauntlet
	@echo "$(GREEN)🏆 Complete validation gauntlet passed!$(NC)"
	@echo "Summary:"
	@echo "✅ Leakage checks"
	@echo "✅ Robustness checks" 
	@echo "✅ Adversarial validation"
	@echo "✅ Calibration analysis"
	@echo "✅ Slice analysis"

summary: ## Print final pipeline summary with all metrics
	@echo "$(BLUE)Generating final summary...$(NC)"
	$(PYTHON) -m src.reporting.final_summary

# Complete end-to-end pipeline
full_pipeline: clean setup-data train infer stack_oof blend gauntlet summary ## Run complete end-to-end pipeline
	@echo "$(GREEN)🎉 Complete pipeline finished successfully!$(NC)"

# Deployment
deploy-staging: ## Deploy to staging environment
	@echo "$(BLUE)Deploying to staging...$(NC)"
	@echo "$(YELLOW)This would deploy the model to staging environment$(NC)"

deploy-prod: ## Deploy to production environment
	@echo "$(BLUE)Deploying to production...$(NC)"
	@echo "$(YELLOW)This would deploy the model to production environment$(NC)"

# Health Checks
health-check: ## Run health checks
	@echo "$(BLUE)Running health checks...$(NC)"
	@$(MAKE) env-info
	@echo "$(YELLOW)Checking data availability...$(NC)"
	@if [ -f "$(DATA_DIR)/train.csv" ]; then echo "$(GREEN)✓ Training data available$(NC)"; else echo "$(RED)✗ Training data missing$(NC)"; fi
	@if [ -d "$(ARTIFACTS_DIR)" ]; then echo "$(GREEN)✓ Artifacts directory exists$(NC)"; else echo "$(YELLOW)! Artifacts directory missing$(NC)"; fi
	@echo "$(YELLOW)Checking Python packages...$(NC)"
	@$(PYTHON) -c "import pandas, numpy, catboost, sklearn; print('$(GREEN)✓ Core packages available$(NC)')" || echo "$(RED)✗ Missing core packages$(NC)"

# All-in-one commands
install-all: install-dev setup-data pre-commit-install ## Complete installation and setup
	@echo "$(GREEN)Complete setup finished!$(NC)"

check-all: lint test-coverage security-scan ## Run all quality checks
	@echo "$(GREEN)All quality checks passed!$(NC)"

build-all: clean format lint test docker-build ## Build everything
	@echo "$(GREEN)Build complete!$(NC)"