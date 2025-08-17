# Titanic Survival Prediction - Production CatBoost Pipeline

[![CI/CD Pipeline](https://github.com/username/titanic-pro-catboost/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/username/titanic-pro-catboost/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-quality machine learning pipeline for predicting Titanic passenger survival using CatBoost with advanced feature engineering, comprehensive testing, and MLOps best practices.

## 🚀 Features

- **Advanced Feature Engineering**: Social titles extraction, cabin deck analysis, family dynamics, and interaction features
- **CatBoost Ensemble**: Native categorical handling with multi-seed cross-validation
- **Out-of-Fold Target Encoding**: Prevents leakage with Bayesian smoothing
- **Group-Aware Cross-Validation**: Respects family boundaries while maintaining stratification
- **SHAP Explainability**: Comprehensive model interpretability analysis
- **Feature Ablation**: Systematic feature importance analysis
- **Data Validation**: Pandera schemas and Great Expectations integration
- **MLflow Tracking**: Experiment tracking and model registry
- **Comprehensive Testing**: Unit, integration, and property-based tests
- **CI/CD Pipeline**: GitHub Actions with multi-OS testing
- **Docker Support**: Production-ready containerization
- **Documentation**: Auto-generated API docs and model cards

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Development](#-development)
- [Testing](#-testing)
- [CI/CD](#-cicd)
- [Docker](#-docker)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [License](#-license)

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/username/titanic-pro-catboost.git
cd titanic-pro-catboost

# Set up development environment
make install-dev

# Download and setup data
make setup-data

# Run the complete pipeline
make pipeline

# View results
ls artifacts/
ls reports/
```

## 💻 Installation

### Prerequisites

- Python 3.8+
- Git
- Make (optional, for convenience commands)
- Docker (optional, for containerized execution)

### Development Setup

```bash
# Clone repository
git clone https://github.com/username/titanic-pro-catboost.git
cd titanic-pro-catboost

# Install development dependencies
make install-dev

# Or manually:
pip install -e ".[dev]"

# Install pre-commit hooks
make pre-commit-install
```

### Production Setup

```bash
# Install production dependencies only
make install

# Or manually:
pip install -e .
```

## 🎯 Usage

### Training Pipeline

```bash
# Complete training pipeline with cross-validation
make train

# Or run directly with Hydra
python -m src.modeling.train_cv

# With custom configuration
python -m src.modeling.train_cv --config-name=train_custom
```

### Inference

```bash
# Generate predictions on test set
make infer

# Or run directly
python -m src.modeling.infer
```

### Feature Analysis

```bash
# Run feature ablation analysis
make ablate

# Generate SHAP explanations
make explain

# Create comprehensive model report
make report
```

### Data Validation

```bash
# Validate data schemas
make validate-data

# Setup data from remote source
make setup-data
```

## 📁 Project Structure

```
titanic-pro-catboost/
├── 📁 .github/                    # GitHub Actions workflows
├── 📁 configs/                    # Hydra configuration files
│   ├── features.yaml              # Feature engineering config
│   └── train.yaml                 # Training configuration
├── 📁 data/                       # Data directory
│   ├── train.csv                  # Training data
│   └── test.csv                   # Test data
├── 📁 src/                        # Source code
│   ├── 📁 core/                   # Core utilities
│   │   ├── contracts.py           # Data validation schemas
│   │   └── utils.py               # Utility functions
│   ├── 📁 features/               # Feature engineering
│   │   ├── build.py               # Feature engineering pipeline
│   │   └── target_encoding.py     # OOF target encoding
│   ├── 📁 modeling/               # Model training and inference
│   │   ├── cat_model.py           # CatBoost model factory
│   │   ├── train_cv.py            # Training pipeline
│   │   ├── infer.py               # Inference pipeline
│   │   └── ablate.py              # Feature ablation
│   └── 📁 reporting/              # Analysis and reporting
│       ├── shap_explain.py        # SHAP analysis
│       └── model_card.py          # Model performance reports
├── 📁 tests/                      # Test suite
│   ├── conftest.py                # Test fixtures
│   ├── test_features.py           # Feature engineering tests
│   ├── test_modeling.py           # Modeling tests
│   └── test_integration.py        # Integration tests
├── 📁 artifacts/                  # Model artifacts
├── 📁 reports/                    # Generated reports
├── pyproject.toml                 # Project configuration
├── Makefile                       # Development commands
├── Dockerfile                     # Container configuration
└── README.md                      # This file
```

## 🛠️ Development

### Available Commands

```bash
# Development setup
make install-dev          # Install development dependencies
make setup-data           # Download and setup dataset
make pre-commit-install   # Install pre-commit hooks

# Code quality
make format               # Format code with black and isort
make lint                 # Run all linting tools
make typecheck            # Run mypy type checking
make test                 # Run test suite
make test-coverage        # Run tests with coverage

# Machine learning pipeline
make train                # Train model with cross-validation
make infer                # Generate predictions
make ablate               # Feature ablation analysis
make explain              # SHAP explainability
make report               # Model performance report

# Complete workflows
make pipeline             # Run complete ML pipeline
make dev-check            # Quick development check
make ci-test              # Full CI test suite
```

### Configuration

The project uses Hydra for configuration management. Configuration files are in the `configs/` directory:

- `train.yaml`: Training configuration (model parameters, CV settings, etc.)
- `features.yaml`: Feature engineering configuration

Example configuration override:
```bash
python -m src.modeling.train_cv catboost.iterations=1000 cv.n_splits=10
```

### Adding New Features

1. Implement feature in `src/features/build.py`
2. Add tests in `tests/test_features.py`
3. Update configuration in `configs/features.yaml`
4. Run tests: `make test`

### Model Development

1. Implement model in `src/modeling/`
2. Add comprehensive tests
3. Update training pipeline if needed
4. Run integration tests: `make test-integration`

## 🧪 Testing

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end pipeline testing
- **Property Tests**: Data validation and invariant testing
- **Performance Tests**: Memory and execution time validation

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-coverage

# Run specific test categories
pytest tests/test_features.py -v
pytest tests/test_modeling.py -v
pytest tests/test_integration.py -v

# Run property-based tests
pytest tests/ -k "property" -v

# Run performance tests
pytest tests/ -k "performance" --benchmark-only
```

### Test Configuration

Tests use pytest with the following plugins:
- `pytest-cov`: Coverage reporting
- `pytest-benchmark`: Performance testing
- `pytest-mock`: Mocking utilities

## 🔄 CI/CD

### GitHub Actions Pipeline

The CI/CD pipeline includes:

1. **Code Quality**: Linting, formatting, type checking
2. **Security**: Safety scanning, bandit analysis
3. **Testing**: Multi-OS and multi-Python version testing
4. **Integration**: End-to-end pipeline testing
5. **Performance**: Benchmark testing
6. **Docker**: Container building and testing
7. **Documentation**: API docs generation
8. **Release**: Automated tagging and artifact publishing

### Pipeline Triggers

- **Push to main/develop**: Full pipeline
- **Pull requests**: Code quality and testing
- **Weekly schedule**: Dependency security scanning

### Status Badges

Check the repository badges for current pipeline status.

## 🐳 Docker

### Building and Running

```bash
# Build Docker image
make docker-build

# Run pipeline in container
make docker-run

# Development container with volume mounts
make docker-dev

# Manual Docker commands
docker build -t titanic-catboost .
docker run --rm -v $(pwd)/data:/app/data titanic-catboost
```

### Multi-stage Build

The Dockerfile uses multi-stage builds:
- **Builder stage**: Compiles dependencies
- **Production stage**: Minimal runtime image
- **Development stage**: Additional dev tools

## 📖 Documentation

### API Documentation

```bash
# Build documentation
make docs

# Serve locally
make docs-serve
```

### Model Cards

Generated model cards include:
- Performance metrics
- Feature importance
- Data statistics
- Limitations and considerations
- Ethical considerations

Access model cards in `reports/model_card.html` after running `make report`.

### Configuration Reference

See `configs/` directory for full configuration options and examples.

## 🚀 Production Deployment

### MLflow Integration

```bash
# Start MLflow UI
make mlflow-ui

# View experiments at http://localhost:5000
```

### Model Serving

The trained models can be served using:
- MLflow Model Serving
- Docker containers
- Cloud ML platforms (AWS SageMaker, Azure ML, GCP AI Platform)

### Monitoring

The pipeline supports monitoring through:
- MLflow experiment tracking
- Custom logging and metrics
- Performance benchmarking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run quality checks: `make dev-check`
5. Commit your changes: `git commit -m 'feat: add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Coding Standards

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Write comprehensive tests
- Document all public APIs
- Use conventional commit messages

### Pre-commit Hooks

The project uses pre-commit hooks for:
- Code formatting (black, isort)
- Linting (ruff, mypy)
- Security scanning (bandit)
- General file checks

## 📊 Performance

### Benchmarks

The pipeline achieves:
- **Feature Engineering**: < 2 seconds for 1000 samples
- **Model Training**: < 30 seconds for full CV (with test config)
- **Memory Usage**: < 100MB for standard dataset
- **Inference**: < 1 second for 400+ predictions

### Optimization

Key optimizations include:
- Efficient pandas operations
- CatBoost native categorical handling
- Memory-efficient cross-validation
- Optimized Docker multi-stage builds

## 🔒 Security

### Security Measures

- Dependency scanning with Safety
- Code security analysis with Bandit
- Secrets detection with detect-secrets
- Container security scanning
- No hardcoded credentials or secrets

### Reporting Security Issues

Please report security vulnerabilities via GitHub Security Advisories.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Kaggle for the Titanic dataset
- CatBoost team for the excellent gradient boosting library
- Open source community for the amazing ML ecosystem

## 📞 Support

- 📧 Email: ml-team@example.com
- 💬 Discord: [ML Community](https://discord.gg/ml)
- 📖 Documentation: [Full Docs](https://username.github.io/titanic-pro-catboost)
- 🐛 Issues: [GitHub Issues](https://github.com/username/titanic-pro-catboost/issues)

---

**Built with ❤️ by the ML Engineering Team**