# Setup script for Titanic Enterprise ML Pipeline development environment (Windows)

$ErrorActionPreference = "Stop"

Write-Host "🚀 Setting up Titanic Enterprise ML Pipeline development environment..." -ForegroundColor Green

# Check if uv is installed
try {
    $uvVersion = uv --version
    Write-Host "✅ uv found: $uvVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ uv is not installed. Please install uv first:" -ForegroundColor Red
    Write-Host "   Visit: https://docs.astral.sh/uv/getting-started/installation/" -ForegroundColor Yellow
    exit 1
}

# Check Python version
Write-Host "🐍 Checking Python version..." -ForegroundColor Blue
if (Test-Path ".python-version") {
    $pythonVersion = Get-Content ".python-version"
    Write-Host "📋 Required Python version: $pythonVersion" -ForegroundColor Blue
} else {
    Write-Host "⚠️  No .python-version file found, using system Python" -ForegroundColor Yellow
}

# Create virtual environment and install dependencies
Write-Host "📦 Installing dependencies with uv..." -ForegroundColor Blue
uv sync

Write-Host "🔧 Setting up pre-commit hooks..." -ForegroundColor Blue
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg

# Create necessary directories
Write-Host "📁 Creating project directories..." -ForegroundColor Blue
$directories = @(
    "data/raw", "data/processed", "data/external",
    "models", "outputs", "logs",
    "mlruns", "dvc-storage"
)

foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

# Initialize git if not already initialized
if (!(Test-Path ".git")) {
    Write-Host "🔄 Initializing git repository..." -ForegroundColor Blue
    git init
    git add .
    git commit -m "Initial commit: Enterprise ML infrastructure setup"
}

# Run initial tests to verify setup
Write-Host "🧪 Running initial tests to verify setup..." -ForegroundColor Blue
uv run pytest tests/unit/test_config.py -v

Write-Host ""
Write-Host "🎉 Development environment setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Activate the environment: .venv\Scripts\activate" -ForegroundColor White
Write-Host "  2. Run tests: make test" -ForegroundColor White
Write-Host "  3. Start development: make help" -ForegroundColor White
Write-Host "  4. View available commands: make help" -ForegroundColor White
Write-Host ""
Write-Host "Happy coding! 🚀" -ForegroundColor Green