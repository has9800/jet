from .eval import Evaluator
from .metrics import compute_rouge, compute_perplexity
from .merge import merge_lora
from .sdk.client import JetClient
from .sdk.training import JetTrainer, quick_train, list_available_models, list_available_datasets
from .models import CURATED_MODELS, CURATED_DATASETS, get_model_info, validate_model_for_gpu

# API imports (optional)
try:
    from .api import app as api_app
    _API_AVAILABLE = True
except ImportError:
    _API_AVAILABLE = False

__all__ = [
    "Evaluator", 
    "compute_rouge", 
    "compute_perplexity", 
    "merge_lora",
    "JetClient",
    "JetTrainer", 
    "quick_train",
    "list_available_models",
    "list_available_datasets",
    "CURATED_MODELS",
    "CURATED_DATASETS",
    "get_model_info",
    "validate_model_for_gpu"
]

# Add API app if available
if _API_AVAILABLE:
    __all__.append("api_app")
