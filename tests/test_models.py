"""
Tests for the curated models catalog
"""

import pytest
from jet.models import (
    get_model_info, 
    validate_model_for_gpu, 
    list_models_by_category,
    search_models,
    CURATED_MODELS
)

def test_get_model_info():
    """Test getting model information"""
    # Test existing model
    info = get_model_info("microsoft/Phi-3-mini-4k-instruct")
    assert info is not None
    assert info.name == "microsoft/Phi-3-mini-4k-instruct"
    assert info.params == "3.8B"
    assert info.gpu_memory_gb == 4
    
    # Test non-existing model
    info = get_model_info("non-existing-model")
    assert info is None

def test_validate_model_for_gpu():
    """Test GPU memory validation"""
    # Test model that fits
    assert validate_model_for_gpu("microsoft/Phi-3-mini-4k-instruct", 6) == True
    assert validate_model_for_gpu("microsoft/Phi-3-mini-4k-instruct", 4) == True
    
    # Test model that doesn't fit
    assert validate_model_for_gpu("microsoft/Phi-3-mini-4k-instruct", 2) == False
    
    # Test non-existing model
    assert validate_model_for_gpu("non-existing-model", 8) == False

def test_list_models_by_category():
    """Test listing models by category"""
    # Test specific category
    small_models = list_models_by_category("small")
    assert len(small_models) > 0
    assert "microsoft/Phi-3-mini-4k-instruct" in small_models
    
    # Test all models
    all_models = list_models_by_category()
    assert len(all_models) > len(small_models)
    assert "microsoft/Phi-3-mini-4k-instruct" in all_models
    assert "microsoft/Phi-4" in all_models

def test_search_models():
    """Test model search functionality"""
    # Search by name
    results = search_models(query="Phi-3")
    assert len(results) > 0
    assert any("Phi-3" in model.name for model in results)
    
    # Search by tags
    results = search_models(tags=["instruction"])
    assert len(results) > 0
    assert all("instruction" in model.tags for model in results)
    
    # Search by both
    results = search_models(query="mini", tags=["instruction"])
    assert len(results) > 0
    assert any("mini" in model.name.lower() and "instruction" in model.tags for model in results)

def test_curated_models_structure():
    """Test that curated models have proper structure"""
    for category, models in CURATED_MODELS.items():
        assert category in ["small", "medium", "large"]
        assert len(models) > 0
        
        for model_name, model_info in models.items():
            assert hasattr(model_info, 'name')
            assert hasattr(model_info, 'params')
            assert hasattr(model_info, 'gpu_memory_gb')
            assert hasattr(model_info, 'max_seq_length')
            assert hasattr(model_info, 'description')
            assert hasattr(model_info, 'tags')
            
            assert isinstance(model_info.params, str)
            assert isinstance(model_info.gpu_memory_gb, int)
            assert isinstance(model_info.max_seq_length, int)
            assert isinstance(model_info.tags, list)
