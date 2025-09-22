import os, requests
from typing import Optional, Dict, Any, List

class JetClient:
    """
    Jet AI SDK Client for both local training and remote API access
    """
    
    def __init__(self, base_url: str = None, api_key: str = None, timeout: int = 60):
        """
        Initialize Jet client
        
        Args:
            base_url: API base URL (None for local-only mode)
            api_key: API key for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/") if base_url else None
        self.api_key = api_key or os.getenv("JET_API_KEY", "")
        self.timeout = timeout
        
        # Import training functionality
        from .training import JetTrainer, quick_train, list_available_models, list_available_datasets
        self.JetTrainer = JetTrainer
        self.quick_train = quick_train
        self.list_available_models = list_available_models
        self.list_available_datasets = list_available_datasets

    def chat_completions(self, model: str, messages, **kwargs):
        """
        Send chat completion request to remote API
        
        Args:
            model: Model name to use
            messages: List of message dictionaries
            **kwargs: Additional parameters
            
        Returns:
            API response
        """
        if not self.base_url:
            raise ValueError("base_url required for API calls")
            
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": messages}
        payload.update(kwargs)
        r = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json()
    
    def create_trainer(self, model_name: str, dataset_name: str, **kwargs) -> 'JetTrainer':
        """
        Create a local trainer instance
        
        Args:
            model_name: Model to fine-tune
            dataset_name: Dataset to use
            **kwargs: Additional trainer parameters
            
        Returns:
            JetTrainer instance
        """
        return self.JetTrainer(model_name, dataset_name, **kwargs)
    
    def train_model(self, model_name: str, dataset_name: str, **kwargs) -> 'JetTrainer':
        """
        Quick model training
        
        Args:
            model_name: Model to fine-tune
            dataset_name: Dataset to use
            **kwargs: Additional training parameters
            
        Returns:
            Trained JetTrainer instance
        """
        return self.quick_train(model_name, dataset_name, **kwargs)
