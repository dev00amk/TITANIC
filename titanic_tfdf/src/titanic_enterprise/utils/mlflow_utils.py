"""MLflow utilities for experiment tracking and model registry."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mlflow
import mlflow.sklearn
import mlflow.tensorflow
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig

from titanic_enterprise.utils.exceptions import MLflowError
from titanic_enterprise.utils.logging import get_logger

logger = get_logger(__name__)


class MLflowManager:
    """Manager class for MLflow operations."""
    
    def __init__(self, config: DictConfig):
        """
        Initialize MLflow manager.
        
        Args:
            config: Configuration containing MLflow settings
        """
        self.config = config
        self.tracking_uri = config.mlflow.tracking_uri
        self.experiment_name = config.mlflow.experiment_name
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(
                name=self.experiment_name,
                artifact_location=config.mlflow.get("artifact_location")
            )
        except mlflow.exceptions.MlflowException:
            # Experiment already exists
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            self.experiment_id = experiment.experiment_id
        
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        logger.info(f"MLflow initialized with experiment: {self.experiment_name}")
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> mlflow.ActiveRun:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
            tags: Optional tags for the run
            
        Returns:
            Active MLflow run
        """
        try:
            run = mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=run_name,
                tags=tags
            )
            logger.info(f"Started MLflow run: {run.info.run_id}")
            return run
        except Exception as e:
            raise MLflowError(f"Failed to start MLflow run: {e}")
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to MLflow.
        
        Args:
            params: Dictionary of parameters to log
        """
        try:
            mlflow.log_params(params)
            logger.debug(f"Logged {len(params)} parameters to MLflow")
        except Exception as e:
            raise MLflowError(f"Failed to log parameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        try:
            for name, value in metrics.items():
                mlflow.log_metric(name, value, step=step)
            logger.debug(f"Logged {len(metrics)} metrics to MLflow")
        except Exception as e:
            raise MLflowError(f"Failed to log metrics: {e}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log an artifact to MLflow.
        
        Args:
            local_path: Local path to the artifact
            artifact_path: Optional path within the artifact store
        """
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.debug(f"Logged artifact: {local_path}")
        except Exception as e:
            raise MLflowError(f"Failed to log artifact: {e}")
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        """
        Log multiple artifacts from a directory.
        
        Args:
            local_dir: Local directory containing artifacts
            artifact_path: Optional path within the artifact store
        """
        try:
            mlflow.log_artifacts(local_dir, artifact_path)
            logger.debug(f"Logged artifacts from: {local_dir}")
        except Exception as e:
            raise MLflowError(f"Failed to log artifacts: {e}")
    
    def log_model(
        self,
        model: Any,
        artifact_path: str,
        model_type: str = "sklearn",
        **kwargs
    ) -> None:
        """
        Log a model to MLflow.
        
        Args:
            model: Model object to log
            artifact_path: Path within the artifact store
            model_type: Type of model (sklearn, tensorflow, etc.)
            **kwargs: Additional arguments for model logging
        """
        try:
            if model_type == "sklearn":
                mlflow.sklearn.log_model(model, artifact_path, **kwargs)
            elif model_type == "tensorflow":
                mlflow.tensorflow.log_model(model, artifact_path, **kwargs)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            logger.info(f"Logged {model_type} model to: {artifact_path}")
        except Exception as e:
            raise MLflowError(f"Failed to log model: {e}")
    
    def register_model(
        self,
        model_uri: str,
        model_name: str,
        tags: Optional[Dict[str, str]] = None
    ) -> Any:
        """
        Register a model in MLflow Model Registry.
        
        Args:
            model_uri: URI of the model to register
            model_name: Name for the registered model
            tags: Optional tags for the model
            
        Returns:
            Model version object
        """
        try:
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags=tags
            )
            logger.info(f"Registered model: {model_name} version {model_version.version}")
            return model_version
        except Exception as e:
            raise MLflowError(f"Failed to register model: {e}")
    
    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing_versions: bool = False
    ) -> None:
        """
        Transition a model to a different stage.
        
        Args:
            model_name: Name of the registered model
            version: Version of the model
            stage: Target stage (Staging, Production, Archived)
            archive_existing_versions: Whether to archive existing versions
        """
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing_versions
            )
            logger.info(f"Transitioned {model_name} v{version} to {stage}")
        except Exception as e:
            raise MLflowError(f"Failed to transition model stage: {e}")
    
    def get_model_version(self, model_name: str, version: str) -> Any:
        """
        Get a specific model version.
        
        Args:
            model_name: Name of the registered model
            version: Version of the model
            
        Returns:
            Model version object
        """
        try:
            return self.client.get_model_version(model_name, version)
        except Exception as e:
            raise MLflowError(f"Failed to get model version: {e}")
    
    def search_runs(
        self,
        filter_string: str = "",
        order_by: Optional[List[str]] = None,
        max_results: int = 1000
    ) -> List[Any]:
        """
        Search for runs in the experiment.
        
        Args:
            filter_string: Filter string for the search
            order_by: List of columns to order by
            max_results: Maximum number of results
            
        Returns:
            List of run objects
        """
        try:
            return self.client.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=filter_string,
                order_by=order_by,
                max_results=max_results
            )
        except Exception as e:
            raise MLflowError(f"Failed to search runs: {e}")
    
    def end_run(self, status: str = "FINISHED") -> None:
        """
        End the current MLflow run.
        
        Args:
            status: Status of the run (FINISHED, FAILED, KILLED)
        """
        try:
            mlflow.end_run(status=status)
            logger.info(f"Ended MLflow run with status: {status}")
        except Exception as e:
            logger.warning(f"Failed to end MLflow run: {e}")


def setup_mlflow(config: DictConfig) -> MLflowManager:
    """
    Set up MLflow with the given configuration.
    
    Args:
        config: Configuration containing MLflow settings
        
    Returns:
        MLflowManager instance
    """
    return MLflowManager(config)


def log_config_as_params(config: DictConfig, prefix: str = "") -> None:
    """
    Log configuration as MLflow parameters.
    
    Args:
        config: Configuration to log
        prefix: Optional prefix for parameter names
    """
    def flatten_config(cfg, parent_key="", sep="."):
        """Flatten nested configuration."""
        items = []
        for k, v in cfg.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_config(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    try:
        flat_config = flatten_config(config)
        if prefix:
            flat_config = {f"{prefix}.{k}": v for k, v in flat_config.items()}
        
        # MLflow has a limit on parameter value length
        params = {}
        for k, v in flat_config.items():
            str_value = str(v)
            if len(str_value) > 250:  # MLflow limit is 250 characters
                str_value = str_value[:247] + "..."
            params[k] = str_value
        
        mlflow.log_params(params)
        logger.debug(f"Logged {len(params)} configuration parameters")
    except Exception as e:
        logger.warning(f"Failed to log configuration as parameters: {e}")