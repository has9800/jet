"""
Curated model catalog for 1-GPU fine-tuning (≤70B parameters)
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class ModelInfo:
    """Information about a model for fine-tuning"""
    name: str
    params: str
    gpu_memory_gb: int
    max_seq_length: int
    description: str
    tags: List[str]
    recommended_batch_size: int = 1
    recommended_lr: float = 2e-4

# Curated models that work well on 1 GPU
CURATED_MODELS: Dict[str, Dict[str, ModelInfo]] = {
    "small": {
        "microsoft/Phi-3-mini-4k-instruct": ModelInfo(
            name="microsoft/Phi-3-mini-4k-instruct",
            params="3.8B",
            gpu_memory_gb=4,
            max_seq_length=4096,
            description="Microsoft's Phi-3 mini model, excellent for instruction following and reasoning",
            tags=["phi3", "instruction", "reasoning", "microsoft"],
            recommended_batch_size=2,
            recommended_lr=2e-4
        ),
        "microsoft/Phi-3-mini-128k-instruct": ModelInfo(
            name="microsoft/Phi-3-mini-128k-instruct",
            params="3.8B",
            gpu_memory_gb=6,
            max_seq_length=128000,
            description="Phi-3 mini with 128k context length, perfect for long documents",
            tags=["phi3", "long-context", "instruction", "microsoft"],
            recommended_batch_size=1,
            recommended_lr=1e-4
        ),
        "microsoft/Phi-3-small-8k-instruct": ModelInfo(
            name="microsoft/Phi-3-small-8k-instruct",
            params="7B",
            gpu_memory_gb=6,
            max_seq_length=8192,
            description="Phi-3 small model with 8k context, great balance of size and performance",
            tags=["phi3", "instruction", "balanced", "microsoft"],
            recommended_batch_size=1,
            recommended_lr=1e-4
        )
    },
    "medium": {
        "microsoft/Phi-3-medium-4k-instruct": ModelInfo(
            name="microsoft/Phi-3-medium-4k-instruct",
            params="14B",
            gpu_memory_gb=8,
            max_seq_length=4096,
            description="Phi-3 medium model with excellent reasoning capabilities",
            tags=["phi3", "instruction", "reasoning", "microsoft", "high-quality"],
            recommended_batch_size=1,
            recommended_lr=8e-5
        ),
        "microsoft/Phi-4-mini": ModelInfo(
            name="microsoft/Phi-4-mini",
            params="12B",
            gpu_memory_gb=8,
            max_seq_length=8192,
            description="Microsoft's latest Phi-4 mini model with state-of-the-art performance",
            tags=["phi4", "latest", "instruction", "microsoft", "sota"],
            recommended_batch_size=1,
            recommended_lr=1e-4
        ),
        "facebook/opt-1.3b": ModelInfo(
            name="facebook/opt-1.3b",
            params="1.3B",
            gpu_memory_gb=4,
            max_seq_length=2048,
            description="Meta's OPT model, good for instruction following",
            tags=["instruction", "opt", "meta", "lightweight"],
            recommended_batch_size=2,
            recommended_lr=2e-4
        )
    },
    "large": {
        "microsoft/Phi-3-large-4k-instruct": ModelInfo(
            name="microsoft/Phi-3-large-4k-instruct",
            params="28B",
            gpu_memory_gb=12,
            max_seq_length=4096,
            description="Phi-3 large model with exceptional reasoning and instruction following",
            tags=["phi3", "instruction", "reasoning", "microsoft", "premium"],
            recommended_batch_size=1,
            recommended_lr=5e-5
        ),
        "microsoft/Phi-4": ModelInfo(
            name="microsoft/Phi-4",
            params="20B",
            gpu_memory_gb=16,
            max_seq_length=16384,
            description="Microsoft's flagship Phi-4 model with cutting-edge capabilities",
            tags=["phi4", "flagship", "instruction", "microsoft", "premium"],
            recommended_batch_size=1,
            recommended_lr=5e-5
        ),
        "gpt-oss:20b": ModelInfo(
            name="gpt-oss:20b",
            params="20B",
            gpu_memory_gb=16,
            max_seq_length=4096,
            description="Open-source GPT model with 20B parameters, excellent for fine-tuning",
            tags=["gpt-oss", "open-source", "fine-tuning", "balanced"],
            recommended_batch_size=1,
            recommended_lr=5e-5
        )
    }
}

# Popular datasets for fine-tuning
CURATED_DATASETS = {
    "conversational": {
        "databricks/databricks-dolly-15k": {
            "name": "Databricks Dolly 15k",
            "description": "High-quality instruction-following dataset",
            "size": "15k examples",
            "tags": ["instruction", "high-quality", "dolly"],
            "text_field": "instruction",  # Primary text field
            "format": "instruction-response"  # Dataset format type
        },
        "OpenAssistant/oasst1": {
            "name": "OpenAssistant Conversations",
            "description": "Human-AI conversation dataset",
            "size": "161k examples",
            "tags": ["conversation", "assistant", "openassistant"],
            "text_field": "text",
            "format": "conversation"
        }
    },
    "code": {
        "bigcode/the-stack-dedup": {
            "name": "The Stack (Code)",
            "description": "Large collection of source code",
            "size": "3TB+",
            "tags": ["code", "programming", "large"],
            "text_field": "content",
            "format": "code"
        },
        "HuggingFaceH4/CodeAlpaca_20K": {
            "name": "Code Alpaca 20K",
            "description": "Code instruction dataset",
            "size": "20k examples",
            "tags": ["code", "instruction", "alpaca"],
            "text_field": "instruction",
            "format": "instruction-response"
        }
    },
    "general": {
        "wikitext-2-raw-v1": {
            "name": "WikiText-2 Raw",
            "description": "Wikipedia text for language modeling",
            "size": "4MB",
            "tags": ["wikipedia", "general", "small"],
            "text_field": "text",
            "format": "text"
        },
        "bookcorpus": {
            "name": "BookCorpus",
            "description": "Collection of books for training",
            "size": "11GB",
            "tags": ["books", "general", "large"],
            "text_field": "text",
            "format": "text"
        }
    }
}

def get_model_info(model_name: str) -> Optional[ModelInfo]:
    """Get information about a specific model"""
    for category in CURATED_MODELS.values():
        if model_name in category:
            return category[model_name]
    return None

def list_models_by_category(category: str = None) -> Dict[str, ModelInfo]:
    """List models by category (small, medium, large) or all models"""
    if category:
        return CURATED_MODELS.get(category, {})
    
    all_models = {}
    for models in CURATED_MODELS.values():
        all_models.update(models)
    return all_models

def validate_model_for_gpu(model_name: str, available_gpu_memory_gb: int = 8) -> bool:
    """Check if a model can run on the available GPU memory"""
    model_info = get_model_info(model_name)
    if not model_info:
        return False
    return model_info.gpu_memory_gb <= available_gpu_memory_gb

def get_recommended_models(gpu_memory_gb: int = 8) -> List[ModelInfo]:
    """Get models recommended for the available GPU memory"""
    recommended = []
    for category in CURATED_MODELS.values():
        for model in category.values():
            if model.gpu_memory_gb <= gpu_memory_gb:
                recommended.append(model)
    return sorted(recommended, key=lambda x: x.gpu_memory_gb)

def search_models(query: str = None, tags: List[str] = None) -> List[ModelInfo]:
    """Search models by name or tags"""
    results = []
    all_models = list_models_by_category()
    
    for model in all_models.values():
        match = True
        
        if query:
            query_lower = query.lower()
            if (query_lower not in model.name.lower() and 
                query_lower not in model.description.lower()):
                match = False
        
        if tags:
            if not any(tag in model.tags for tag in tags):
                match = False
        
        if match:
            results.append(model)
    
    return results

def get_dataset_info(dataset_name: str) -> Optional[Dict[str, Any]]:
    """Get information about a specific dataset"""
    for category in CURATED_DATASETS.values():
        if dataset_name in category:
            return category[dataset_name]
    return None

def detect_text_field(dataset) -> str:
    """
    Automatically detect the best text field in a dataset
    
    Args:
        dataset: HuggingFace dataset object
        
    Returns:
        Name of the best text field to use
    """
    # Common text field names in order of preference
    preferred_fields = [
        "text", "instruction", "content", "prompt", "input", 
        "question", "context", "response", "answer", "completion"
    ]
    
    available_fields = list(dataset.features.keys())
    
    # First, try to find a field from our curated datasets
    dataset_info = get_dataset_info(dataset.info.dataset_name if hasattr(dataset, 'info') else None)
    if dataset_info and 'text_field' in dataset_info:
        if dataset_info['text_field'] in available_fields:
            return dataset_info['text_field']
    
    # Then try preferred fields in order
    for field in preferred_fields:
        if field in available_fields:
            return field
    
    # If no preferred field found, return the first available field
    if available_fields:
        return available_fields[0]
    
    # Fallback
    return "text"

def format_dataset_for_training(dataset, text_field: str = None) -> Any:
    """
    Format a dataset for training by creating a unified text field
    
    Args:
        dataset: HuggingFace dataset
        text_field: Specific text field to use (auto-detected if None)
        
    Returns:
        Dataset with unified text field
    """
    if text_field is None:
        text_field = detect_text_field(dataset)
    
    # For instruction-response datasets, always combine fields into 'text'
    if "instruction" in dataset.features and "response" in dataset.features:
        def combine_instruction_response(example):
            instruction = example.get("instruction", "")
            response = example.get("response", "")
            context = example.get("context", "")
            
            if context:
                text = f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n{response}"
            else:
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
            
            return {"text": text}
        
        return dataset.map(combine_instruction_response)
    
    # For conversation datasets, combine messages
    if "messages" in dataset.features:
        def combine_messages(example):
            messages = example["messages"]
            text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            return {"text": text}
        
        return dataset.map(combine_messages)
    
    # If the dataset already has a 'text' field, return as-is
    if "text" in dataset.features:
        return dataset
    
    # Fallback: use the detected field as 'text'
    if text_field in dataset.features:
        def rename_field(example):
            return {"text": example[text_field]}
        return dataset.map(rename_field)
    
    # Last resort: use the first available field
    available_fields = list(dataset.features.keys())
    if available_fields:
        def use_first_field(example):
            return {"text": example[available_fields[0]]}
        return dataset.map(use_first_field)
    
    return dataset
