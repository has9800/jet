"""
Tests for the SDK training functionality
"""

import pytest
import tempfile
import os
from pathlib import Path
from jet.sdk.training import JetTrainer, quick_train, list_available_models, list_available_datasets

def test_jet_trainer_initialization():
    """Test JetTrainer initialization"""
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = JetTrainer(
            model_name="sshleifer/tiny-gpt2",  # Use tiny model for testing
            dataset_name="wikitext",
            output_dir=temp_dir
        )
        
        assert trainer.model_name == "sshleifer/tiny-gpt2"
        assert trainer.dataset_name == "wikitext"
        assert trainer.output_dir == Path(temp_dir)
        assert trainer.engine == "auto"

def test_jet_trainer_model_info():
    """Test getting model information"""
    trainer = JetTrainer("microsoft/DialoGPT-small", "wikitext")
    
    info = trainer.get_model_info()
    assert "model_name" in info
    assert "dataset_name" in info
    assert "engine" in info
    assert "trained" in info
    assert info["model_name"] == "microsoft/DialoGPT-small"
    assert info["trained"] == False

def test_list_available_models():
    """Test listing available models"""
    models = list_available_models()
    assert isinstance(models, dict)
    assert len(models) > 0
    
    # Check structure
    for name, info in models.items():
        assert "params" in info
        assert "gpu_memory_gb" in info
        assert "description" in info
        assert "tags" in info

def test_list_available_datasets():
    """Test listing available datasets"""
    datasets = list_available_datasets()
    assert isinstance(datasets, dict)
    assert len(datasets) > 0
    
    # Check structure
    for category, category_datasets in datasets.items():
        assert isinstance(category_datasets, dict)
        for name, info in category_datasets.items():
            assert "name" in info
            assert "description" in info
            assert "tags" in info

def test_jet_trainer_with_curated_model():
    """Test JetTrainer with curated model"""
    trainer = JetTrainer("microsoft/Phi-3-mini-4k-instruct", "wikitext")
    
    # Should have model info for curated model
    assert trainer.model_info is not None
    assert trainer.model_info.name == "microsoft/Phi-3-mini-4k-instruct"
    assert trainer.model_info.params == "3.8B"

def test_jet_trainer_with_non_curated_model():
    """Test JetTrainer with non-curated model"""
    trainer = JetTrainer("non-curated-model", "wikitext")
    
    # Should not have model info for non-curated model
    assert trainer.model_info is None

# Note: We don't test actual training here as it requires significant resources
# and time. Training tests should be run separately in CI with proper GPU resources.
