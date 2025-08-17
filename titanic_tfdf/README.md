# Titanic Enterprise ML Infrastructure

A production-ready machine learning pipeline for Titanic survival prediction using TensorFlow Decision Forests with enterprise-grade MLOps practices.

## Features

- **Modern Python Development**: Built with `uv` for fast, reliable dependency management
- **Enterprise-Grade Architecture**: Modular, testable, and maintainable codebase
- **MLOps Best Practices**: Experiment tracking, model registry, data versioning
- **Automated Quality Assurance**: Pre-commit hooks, linting, type checking, security scanning
- **Comprehensive Testing**: Unit tests, integration tests, property-based testing
- **Configuration Management**: Hierarchical configuration with Hydra
- **Data Validation**: Schema validation and data quality checks with Pandera
- **Containerization**: Docker support for consistent environments
- **CI/CD Ready**: GitHub Actions workflows for automated testing and deployment

## Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- Git
- Docker (optional, for containerized development)

### Installing uv

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.sh | iex"
```

## Quick Start

### 1. Setup Environment

**Linux/macOS:**
```bash
./scripts/setup_env.sh
```

**Windows:**
```powershell
.\scripts\setup_env.ps1
```

### 2. Manual Setup (Alternative)

```bash
# Install dependencies
uv sync

# Setup pre-commit hooks
uv run pre-commit install

# Run tests to verify setup
make test
```

## Development Commands

```bash
make help                   # View all available commands
make setup                  # Setup development environment
make test                   # Run all tests
make lint                   # Run linting
make format                 # Format code
make pipeline              # Run ML pipeline
make train                 # Train model
make predict               # Generate predictions
```

## Project Structure

```
titanic-enterprise/
├── configs/               # Hydra configuration files
├── data/                 # Data directories
├── src/titanic_enterprise/ # Main package
├── tests/                # Test suite
├── scripts/              # Utility scripts
├── pyproject.toml        # Project configuration
├── Makefile             # Build automation
└── README.md            # This file
```

## Usage

### Training a Model

```bash
titanic-train
```

### Making Predictions

```bash
titanic-predict --model-path models/latest --input-file data/test.csv
```

### Running Complete Pipeline

```bash
titanic-pipeline --stage all
```

## Testing

```bash
make test                  # All tests
make test-unit            # Unit tests only
make test-integration     # Integration tests only
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes and add tests
4. Run the test suite: `make test`
5. Submit a Pull Request

## License

This project is licensed under the MIT License.