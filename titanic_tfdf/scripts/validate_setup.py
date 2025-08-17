#!/usr/bin/env python3
"""
Validation script to test the basic setup of the Titanic Enterprise ML Infrastructure.
This script can be run without uv to validate the project structure.
"""

import os
import sys
from pathlib import Path


def validate_project_structure():
    """Validate that all required directories and files exist."""
    print("🔍 Validating project structure...")
    
    required_files = [
        "pyproject.toml",
        "README.md",
        "Makefile",
        ".gitignore",
        ".pre-commit-config.yaml",
        ".gitleaks.toml",
        ".python-version",
    ]
    
    required_dirs = [
        "src/titanic_enterprise",
        "configs",
        "tests/unit",
        "tests/integration", 
        "data/raw",
        "data/processed",
        "scripts",
        "logs",
    ]
    
    required_config_files = [
        "configs/config.yaml",
        "configs/model/tfdf.yaml",
        "configs/data/titanic.yaml",
        "configs/experiment/baseline.yaml",
    ]
    
    required_src_files = [
        "src/titanic_enterprise/__init__.py",
        "src/titanic_enterprise/cli.py",
        "src/titanic_enterprise/models/base.py",
        "src/titanic_enterprise/utils/config.py",
        "src/titanic_enterprise/utils/logging.py",
    ]
    
    required_test_files = [
        "tests/conftest.py",
        "tests/unit/test_config.py",
        "tests/unit/test_base_model.py",
    ]
    
    all_required = (
        required_files + 
        required_config_files + 
        required_src_files + 
        required_test_files
    )
    
    missing_files = []
    missing_dirs = []
    
    # Check files
    for file_path in all_required:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    # Check directories
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
    
    if missing_dirs:
        print("❌ Missing required directories:")
        for dir_path in missing_dirs:
            print(f"   - {dir_path}")
    
    if not missing_files and not missing_dirs:
        print("✅ All required files and directories are present!")
        return True
    
    return False


def validate_python_imports():
    """Test that basic Python imports work."""
    print("\n🐍 Validating Python imports...")
    
    try:
        # Test standard library imports
        import json
        import os
        import sys
        from pathlib import Path
        print("✅ Standard library imports work")
        
        # Test that our package structure is importable
        sys.path.insert(0, str(Path("src").absolute()))
        
        try:
            import titanic_enterprise
            print(f"✅ titanic_enterprise package imported (version: {titanic_enterprise.__version__})")
        except ImportError as e:
            print(f"❌ Failed to import titanic_enterprise: {e}")
            return False
        
        try:
            from titanic_enterprise.utils.config import load_config
            print("✅ Configuration utilities imported")
        except ImportError as e:
            print(f"❌ Failed to import config utilities: {e}")
            return False
        
        try:
            from titanic_enterprise.models.base import BaseModel
            print("✅ Base model imported")
        except ImportError as e:
            print(f"❌ Failed to import base model: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Python import validation failed: {e}")
        return False


def validate_configuration():
    """Test that configuration files are valid YAML."""
    print("\n⚙️  Validating configuration files...")
    
    try:
        import yaml
    except ImportError:
        print("⚠️  PyYAML not available, skipping YAML validation")
        return True
    
    config_files = [
        "configs/config.yaml",
        "configs/model/tfdf.yaml", 
        "configs/data/titanic.yaml",
        "configs/experiment/baseline.yaml",
    ]
    
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                yaml.safe_load(f)
            print(f"✅ {config_file} is valid YAML")
        except Exception as e:
            print(f"❌ {config_file} is invalid: {e}")
            return False
    
    return True


def main():
    """Run all validation checks."""
    print("🚀 Titanic Enterprise ML Infrastructure - Setup Validation")
    print("=" * 60)
    
    # Change to project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    print(f"📁 Project root: {project_root.absolute()}")
    
    # Run validation checks
    checks = [
        ("Project Structure", validate_project_structure),
        ("Python Imports", validate_python_imports),
        ("Configuration Files", validate_configuration),
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name} validation failed with error: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Validation Summary:")
    
    all_passed = True
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {check_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All validation checks passed!")
        print("Your Titanic Enterprise ML Infrastructure is ready for development!")
        print("\nNext steps:")
        print("1. Install uv: https://docs.astral.sh/uv/getting-started/installation/")
        print("2. Run: uv sync")
        print("3. Run: make test")
        return 0
    else:
        print("\n⚠️  Some validation checks failed.")
        print("Please review the errors above and fix any issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())