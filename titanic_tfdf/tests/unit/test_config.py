"""Unit tests for configuration utilities."""

import pytest
from omegaconf import DictConfig, OmegaConf

from titanic_enterprise.utils.config import (
    get_config_value,
    merge_configs,
    update_config,
    validate_config,
)


class TestConfigUtils:
    """Test configuration utility functions."""
    
    def test_get_config_value_existing_key(self, sample_config):
        """Test getting an existing configuration value."""
        value = get_config_value(sample_config, "project.name")
        assert value == "titanic-enterprise-test"
    
    def test_get_config_value_missing_key(self, sample_config):
        """Test getting a missing configuration value with default."""
        value = get_config_value(sample_config, "missing.key", default="default_value")
        assert value == "default_value"
    
    def test_get_config_value_missing_key_no_default(self, sample_config):
        """Test getting a missing configuration value without default."""
        value = get_config_value(sample_config, "missing.key")
        assert value is None
    
    def test_merge_configs(self):
        """Test merging two configurations."""
        base_config = OmegaConf.create({
            "a": 1,
            "b": {"c": 2, "d": 3}
        })
        
        override_config = OmegaConf.create({
            "b": {"c": 4},
            "e": 5
        })
        
        merged = merge_configs(base_config, override_config)
        
        assert merged.a == 1
        assert merged.b.c == 4  # Overridden
        assert merged.b.d == 3  # Preserved
        assert merged.e == 5    # Added
    
    def test_update_config(self, sample_config):
        """Test updating configuration with new values."""
        updates = {
            "project.name": "updated-name",
            "new.key": "new_value"
        }
        
        updated_config = update_config(sample_config, updates)
        
        assert get_config_value(updated_config, "project.name") == "updated-name"
        assert get_config_value(updated_config, "new.key") == "new_value"
        # Original should be unchanged
        assert get_config_value(sample_config, "project.name") == "titanic-enterprise-test"
    
    def test_validate_config_success(self, sample_config):
        """Test successful configuration validation."""
        required_keys = ["project.name", "project.version", "paths.data_dir"]
        
        # Should not raise any exception
        validate_config(sample_config, required_keys)
    
    def test_validate_config_missing_keys(self, sample_config):
        """Test configuration validation with missing keys."""
        required_keys = ["project.name", "missing.key", "another.missing.key"]
        
        with pytest.raises(ValueError) as exc_info:
            validate_config(sample_config, required_keys)
        
        error_message = str(exc_info.value)
        assert "missing.key" in error_message
        assert "another.missing.key" in error_message
    
    def test_validate_config_empty_required_keys(self, sample_config):
        """Test configuration validation with empty required keys."""
        # Should not raise any exception
        validate_config(sample_config, [])