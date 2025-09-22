#!/usr/bin/env python3
"""
Jet AI - Quick Start Example

This script demonstrates how to use Jet AI to fine-tune language models
with just a few lines of code.
"""

# Install Jet AI (if not already installed)
# pip install jet-ai-sdk

# Import Jet AI
import jet
from jet import JetTrainer, quick_train, list_available_models, list_available_datasets

def main():
    print("🚀 Jet AI Quick Start Example")
    print("=" * 50)
    
    # 1. Explore available models
    print("\n📚 Available Models:")
    models = list_available_models()
    for name, info in models.items():
        print(f"{name}: {info['params']} parameters, {info['gpu_memory_gb']}GB GPU memory")
        print(f"{info['description']}")
        print()
    
    # 2. Quick training example
    print("🏋️ Starting quick training...")
    print("This will fine-tune a small model on a dataset")
    
    try:
        trainer = quick_train(
            model_name="microsoft/Phi-3-mini-4k-instruct",  # Phi-3 mini for quick training
            dataset_name="databricks/databricks-dolly-15k",
            test_prompts=["Hello, how are you?", "What is machine learning?"],
            output_dir="./my_fine_tuned_model"
        )
        print("✅ Training completed!")
        
        # 3. Chat with the model
        print("\n💬 Chat with your model:")
        test_prompts = [
            "Hello, how are you today?",
            "What is the capital of France?",
            "Explain machine learning in simple terms."
        ]
        
        for prompt in test_prompts:
            response = trainer.chat(prompt, max_new_tokens=50)
            print(f"You: {prompt}")
            print(f"Model: {response}")
            print()
        
        # 4. Show model information
        print("📋 Model Information:")
        model_info = trainer.get_model_info()
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
        print("\n🎉 Quick start completed successfully!")
        print("You can now use your fine-tuned model for inference!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        print("Make sure you have the required dependencies installed:")
        print("pip install jet-ai-sdk transformers datasets torch")

if __name__ == "__main__":
    main()
