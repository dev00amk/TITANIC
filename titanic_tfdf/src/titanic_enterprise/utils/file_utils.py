"""File and path utilities for the Titanic Enterprise ML infrastructure."""

import json
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import yaml
from omegaconf import DictConfig, OmegaConf

from titanic_enterprise.utils.exceptions import TitanicEnterpriseError
from titanic_enterprise.utils.logging import get_logger

logger = get_logger(__name__)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent_dir(file_path: Union[str, Path]) -> Path:
    """
    Ensure parent directory of a file exists.
    
    Args:
        file_path: File path
        
    Returns:
        Path object of the file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def save_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Output file path
        indent: JSON indentation
    """
    file_path = ensure_parent_dir(file_path)
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent, default=str)
        logger.debug(f"Saved JSON to: {file_path}")
    except Exception as e:
        raise TitanicEnterpriseError(f"Failed to save JSON file: {e}")


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from JSON file.
    
    Args:
        file_path: Input file path
        
    Returns:
        Loaded data
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.debug(f"Loaded JSON from: {file_path}")
        return data
    except Exception as e:
        raise TitanicEnterpriseError(f"Failed to load JSON file: {e}")


def save_yaml(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Save data to YAML file.
    
    Args:
        data: Data to save
        file_path: Output file path
    """
    file_path = ensure_parent_dir(file_path)
    try:
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)
        logger.debug(f"Saved YAML to: {file_path}")
    except Exception as e:
        raise TitanicEnterpriseError(f"Failed to save YAML file: {e}")


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from YAML file.
    
    Args:
        file_path: Input file path
        
    Returns:
        Loaded data
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        logger.debug(f"Loaded YAML from: {file_path}")
        return data
    except Exception as e:
        raise TitanicEnterpriseError(f"Failed to load YAML file: {e}")


def save_pickle(obj: Any, file_path: Union[str, Path]) -> None:
    """
    Save object to pickle file.
    
    Args:
        obj: Object to save
        file_path: Output file path
    """
    file_path = ensure_parent_dir(file_path)
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        logger.debug(f"Saved pickle to: {file_path}")
    except Exception as e:
        raise TitanicEnterpriseError(f"Failed to save pickle file: {e}")


def load_pickle(file_path: Union[str, Path]) -> Any:
    """
    Load object from pickle file.
    
    Args:
        file_path: Input file path
        
    Returns:
        Loaded object
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        logger.debug(f"Loaded pickle from: {file_path}")
        return obj
    except Exception as e:
        raise TitanicEnterpriseError(f"Failed to load pickle file: {e}")


def save_dataframe(df: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> None:
    """
    Save DataFrame to file (CSV, Parquet, etc.).
    
    Args:
        df: DataFrame to save
        file_path: Output file path
        **kwargs: Additional arguments for pandas save methods
    """
    file_path = ensure_parent_dir(file_path)
    file_path = Path(file_path)
    
    try:
        if file_path.suffix.lower() == '.csv':
            df.to_csv(file_path, index=False, **kwargs)
        elif file_path.suffix.lower() == '.parquet':
            df.to_parquet(file_path, index=False, **kwargs)
        elif file_path.suffix.lower() == '.json':
            df.to_json(file_path, orient='records', **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logger.debug(f"Saved DataFrame to: {file_path}")
    except Exception as e:
        raise TitanicEnterpriseError(f"Failed to save DataFrame: {e}")


def load_dataframe(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Load DataFrame from file (CSV, Parquet, etc.).
    
    Args:
        file_path: Input file path
        **kwargs: Additional arguments for pandas load methods
        
    Returns:
        Loaded DataFrame
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    try:
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path, **kwargs)
        elif file_path.suffix.lower() == '.parquet':
            df = pd.read_parquet(file_path, **kwargs)
        elif file_path.suffix.lower() == '.json':
            df = pd.read_json(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logger.debug(f"Loaded DataFrame from: {file_path} (shape: {df.shape})")
        return df
    except Exception as e:
        raise TitanicEnterpriseError(f"Failed to load DataFrame: {e}")


def copy_file(src: Union[str, Path], dst: Union[str, Path]) -> None:
    """
    Copy file from source to destination.
    
    Args:
        src: Source file path
        dst: Destination file path
    """
    src = Path(src)
    dst = ensure_parent_dir(dst)
    
    try:
        shutil.copy2(src, dst)
        logger.debug(f"Copied file: {src} -> {dst}")
    except Exception as e:
        raise TitanicEnterpriseError(f"Failed to copy file: {e}")


def copy_directory(src: Union[str, Path], dst: Union[str, Path]) -> None:
    """
    Copy directory from source to destination.
    
    Args:
        src: Source directory path
        dst: Destination directory path
    """
    src = Path(src)
    dst = Path(dst)
    
    try:
        shutil.copytree(src, dst, dirs_exist_ok=True)
        logger.debug(f"Copied directory: {src} -> {dst}")
    except Exception as e:
        raise TitanicEnterpriseError(f"Failed to copy directory: {e}")


def remove_file(file_path: Union[str, Path]) -> None:
    """
    Remove file if it exists.
    
    Args:
        file_path: File path to remove
    """
    file_path = Path(file_path)
    if file_path.exists():
        try:
            file_path.unlink()
            logger.debug(f"Removed file: {file_path}")
        except Exception as e:
            raise TitanicEnterpriseError(f"Failed to remove file: {e}")


def remove_directory(dir_path: Union[str, Path]) -> None:
    """
    Remove directory if it exists.
    
    Args:
        dir_path: Directory path to remove
    """
    dir_path = Path(dir_path)
    if dir_path.exists():
        try:
            shutil.rmtree(dir_path)
            logger.debug(f"Removed directory: {dir_path}")
        except Exception as e:
            raise TitanicEnterpriseError(f"Failed to remove directory: {e}")


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: File path
        
    Returns:
        File size in bytes
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return file_path.stat().st_size


def list_files(
    directory: Union[str, Path],
    pattern: str = "*",
    recursive: bool = False
) -> List[Path]:
    """
    List files in directory matching pattern.
    
    Args:
        directory: Directory to search
        pattern: File pattern (glob)
        recursive: Whether to search recursively
        
    Returns:
        List of matching file paths
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if recursive:
        files = list(directory.rglob(pattern))
    else:
        files = list(directory.glob(pattern))
    
    # Filter to only files (not directories)
    files = [f for f in files if f.is_file()]
    
    logger.debug(f"Found {len(files)} files matching '{pattern}' in {directory}")
    return files


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    # Start from current file and go up until we find pyproject.toml
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    
    # Fallback to current working directory
    return Path.cwd()


def resolve_path(path: Union[str, Path], base_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolve path relative to base path or project root.
    
    Args:
        path: Path to resolve
        base_path: Base path (defaults to project root)
        
    Returns:
        Resolved absolute path
    """
    path = Path(path)
    
    if path.is_absolute():
        return path
    
    if base_path is None:
        base_path = get_project_root()
    else:
        base_path = Path(base_path)
    
    return (base_path / path).resolve()