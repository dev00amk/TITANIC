#!/bin/bash
# Setup script for Titanic Enterprise ML Pipeline development environment

set -e  # Exit on any error

echo "🚀 Setting up Titanic Enterprise ML Pipeline development environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install uv first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "   or visit: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

echo "✅ uv found: $(uv --version)"

# Check Python version
echo "🐍 Checking Python version..."
if [ -f ".python-version" ]; then
    PYTHON_VERSION=$(cat .python-version)
    echo "📋 Required Python version: $PYTHON_VERSION"
else
    echo "⚠️  No .python-version file found, using system Python"
fi

# Create virtual environment and install dependencies
echo "📦 Installing dependencies with uv..."
uv sync

echo "🔧 Setting up pre-commit hooks..."
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/raw data/processed data/external
mkdir -p models outputs logs
mkdir -p mlruns dvc-storage

# Initialize git if not already initialized
if [ ! -d ".git" ]; then
    echo "🔄 Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit: Enterprise ML infrastructure setup"
fi

# Run initial tests to verify setup
echo "🧪 Running initial tests to verify setup..."
uv run pytest tests/unit/test_config.py -v

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "Next steps:"
echo "  1. Activate the environment: source .venv/bin/activate (Linux/Mac) or .venv\\Scripts\\activate (Windows)"
echo "  2. Run tests: make test"
echo "  3. Start development: make help"
echo "  4. View available commands: make help"
echo ""
echo "Happy coding! 🚀"