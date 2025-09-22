"""
High-level training SDK for Jet AI

This module provides a user-friendly interface for fine-tuning language models
with automatic dataset formatting, model validation, and training optimization.
"""

import os
import time
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import mlflow
from datasets import load_dataset, Dataset

from ..train import train_with_options
from ..eval import Evaluator
from ..merge import merge_lora
from ..models import get_model_info, validate_model_for_gpu, CURATED_DATASETS, detect_text_field, format_dataset_for_training


class JetTrainer:
    """
    High-level interface for fine-tuning models with Jet AI
    """
    
    def __init__(
        self, 
        model_name: str, 
        dataset_name: str,
        output_dir: str = "./jet_outputs",
        engine: str = "auto"
    ):
        """
        Initialize a Jet trainer
        
        Args:
            model_name: HuggingFace model name or curated model ID
            dataset_name: HuggingFace dataset name
            output_dir: Directory to save outputs
            engine: Training engine ("auto", "hf", "unsloth")
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.engine = engine
        
        # Validate model
        self.model_info = get_model_info(model_name)
        if not self.model_info:
            print(f"⚠️  Model '{model_name}' not in curated catalog. Using as-is.")
        else:
            print(f"✅ Using curated model: {self.model_info.description}")
            if not validate_model_for_gpu(model_name):
                print(f"⚠️  Warning: Model may require more GPU memory than available")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.training_job = None
        self.evaluator = None
        
    def train(
        self,
        epochs: int = 1,
        learning_rate: float = None,
        batch_size: int = None,
        max_seq_length: int = None,
        text_field: str = "text",
        merge_weights: bool = False,
        **kwargs
    ) -> 'JetTrainer':
        """
        Fine-tune the model
        
        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate (uses model recommendation if None)
            batch_size: Batch size per device (uses model recommendation if None)
            max_seq_length: Maximum sequence length (uses model recommendation if None)
            text_field: Dataset field containing text
            merge_weights: Whether to merge LoRA weights after training
            **kwargs: Additional training parameters
        """
        print(f"🚀 Starting fine-tuning of {self.model_name}")
        print(f"📊 Dataset: {self.dataset_name}")
        print(f"🔧 Engine: {self.engine}")
        
        # Use model recommendations if available
        if self.model_info:
            learning_rate = learning_rate or self.model_info.recommended_lr
            batch_size = batch_size or self.model_info.recommended_batch_size
            max_seq_length = max_seq_length or self.model_info.max_seq_length
        else:
            learning_rate = learning_rate or 2e-4
            batch_size = batch_size or 1
            max_seq_length = max_seq_length or 2048
        
        # Load dataset
        print("📥 Loading dataset...")
        try:
            dataset = load_dataset(self.dataset_name, split="train", streaming=False)
            print(f"✅ Loaded {len(dataset)} examples")
            
            # Auto-detect and format the dataset for training
            print("🔍 Detecting text field...")
            detected_field = detect_text_field(dataset)
            print(f"📝 Using field: '{detected_field}'")
            
            # Format dataset to have a unified 'text' field
            dataset = format_dataset_for_training(dataset, detected_field)
            print(f"✅ Dataset formatted for training")
            
        except Exception as e:
            error_msg = f"Failed to load dataset '{self.dataset_name}': {e}"
            print(f"❌ {error_msg}")
            print("💡 Tip: Make sure the dataset name is correct and accessible on HuggingFace Hub")
            raise ValueError(error_msg)
        
        # Create training options
        class TrainingOptions:
            def __init__(self, model_name, output_dir, batch_size, learning_rate, epochs, engine, text_field, max_seq_length, merge_weights):
                self.model = model_name
                self.output_dir = str(output_dir)
                self.per_device_batch = batch_size
                self.grad_accum = 8
                self.lr = learning_rate
                self.epochs = epochs
                self.seed = 42
                self.engine = engine
                self.text_field = "text"  # Always use 'text' after formatting
                self.max_seq = max_seq_length
                self.merge_weights = merge_weights
                self.do_sample = True
                self.temperature = 0.7
                self.top_p = 0.9
                self.top_k = 50
        
        # Start training
        print("🏋️  Training started...")
        start_time = time.time()
        
        try:
            self.training_job = train_with_options(
                TrainingOptions(
                    self.model_name, 
                    self.output_dir, 
                    batch_size, 
                    learning_rate, 
                    epochs, 
                    self.engine, 
                    text_field, 
                    max_seq_length, 
                    merge_weights
                ), 
                train_ds=dataset, 
                eval_ds=None
            )
            
            training_time = time.time() - start_time
            print(f"✅ Training completed in {training_time:.1f} seconds")
            print(f"📁 Model saved to: {self.training_job.model_dir}")
            
            if hasattr(self.training_job, 'merged_dir') and self.training_job.merged_dir:
                print(f"🔗 Merged model saved to: {self.training_job.merged_dir}")
            
        except Exception as e:
            error_msg = f"Training failed: {e}"
            print(f"❌ {error_msg}")
            print("💡 Common solutions:")
            print("   - Check if you have enough GPU memory")
            print("   - Try a smaller model or dataset")
            print("   - Ensure all dependencies are installed")
            raise RuntimeError(error_msg)
        
        return self
    
    def evaluate(
        self, 
        test_prompts: List[str], 
        references: Optional[List[str]] = None,
        perplexity_texts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the fine-tuned model
        
        Args:
            test_prompts: List of prompts to test
            references: Optional reference answers for ROUGE scoring
            perplexity_texts: Optional texts for perplexity calculation
            
        Returns:
            Dictionary with evaluation results
        """
        if not self.training_job:
            raise ValueError("Must train model before evaluating")
        
        print("📊 Evaluating model...")
        
        # Use merged model if available, otherwise use adapter
        model_path = getattr(self.training_job, 'merged_dir', None) or self.training_job.model_dir
        
        try:
            self.evaluator = Evaluator(
                model_path,
                do_sample=True,
                max_new_tokens=64
            )
            
            results = self.evaluator.evaluate(
                test_prompts, 
                references=references, 
                perplexity_texts=perplexity_texts
            )
            
            print("✅ Evaluation completed")
            print(f"📈 Generated {results['count']} responses")
            
            if 'metrics' in results and results['metrics']:
                print("📊 Metrics:")
                for metric, value in results['metrics'].items():
                    if isinstance(value, float):
                        print(f"  {metric}: {value:.4f}")
                    else:
                        print(f"  {metric}: {value}")
            
            return results
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            raise
    
    def chat(self, prompt: str, max_new_tokens: int = 64) -> str:
        """
        Chat with the fine-tuned model
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Model response
        """
        if not self.evaluator:
            # Initialize evaluator if not already done
            if not self.training_job:
                raise ValueError("Must train model before chatting")
            
            model_path = getattr(self.training_job, 'merged_dir', None) or self.training_job.model_dir
            self.evaluator = Evaluator(model_path, do_sample=True, max_new_tokens=max_new_tokens)
        
        results = self.evaluator.evaluate([prompt])
        return results['preds'][0]
    
    def save_model(self, path: str = None) -> str:
        """
        Save the trained model to a specific path
        
        Args:
            path: Directory to save model (uses output_dir if None)
            
        Returns:
            Path where model was saved
        """
        if not self.training_job:
            raise ValueError("No trained model to save")
        
        if path:
            save_path = Path(path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Copy model files
            import shutil
            model_path = getattr(self.training_job, 'merged_dir', None) or self.training_job.model_dir
            shutil.copytree(model_path, save_path, dirs_exist_ok=True)
            
            print(f"💾 Model saved to: {save_path}")
            return str(save_path)
        
        return self.training_job.model_dir
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        info = {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "engine": self.engine,
            "output_dir": str(self.output_dir),
            "trained": self.training_job is not None
        }
        
        if self.model_info:
            info.update({
                "parameters": self.model_info.params,
                "gpu_memory_gb": self.model_info.gpu_memory_gb,
                "max_seq_length": self.model_info.max_seq_length,
                "description": self.model_info.description,
                "tags": self.model_info.tags
            })
        
        return info


def quick_train(
    model_name: str,
    dataset_name: str,
    test_prompts: List[str] = None,
    output_dir: str = "./jet_outputs"
) -> JetTrainer:
    """
    Quick training function for simple use cases
    
    Args:
        model_name: Model to fine-tune
        dataset_name: Dataset to use
        test_prompts: Optional test prompts for evaluation
        output_dir: Output directory
        
    Returns:
        Trained JetTrainer instance
    """
    trainer = JetTrainer(model_name, dataset_name, output_dir)
    trainer.train(epochs=1)  # Use default epochs
    
    if test_prompts:
        trainer.evaluate(test_prompts)
    
    return trainer


def list_available_models(category: str = None) -> Dict[str, Any]:
    """List available models for fine-tuning"""
    from ..models import list_models_by_category
    
    models = list_models_by_category(category)
    return {name: {
        "params": info.params,
        "gpu_memory_gb": info.gpu_memory_gb,
        "description": info.description,
        "tags": info.tags
    } for name, info in models.items()}


def list_available_datasets() -> Dict[str, Any]:
    """List available datasets for fine-tuning"""
    return CURATED_DATASETS
