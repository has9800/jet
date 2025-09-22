#!/usr/bin/env python3
"""
Jet AI - Production Demo Script

This script demonstrates the complete Jet AI pipeline for fine-tuning
language models with just a few lines of code.
"""

import time
from jet import JetTrainer, quick_train, list_available_models, list_available_datasets

def print_header():
    """Print a beautiful header for the demo"""
    print("=" * 60)
    print("🚀 JET AI - DEMOCRATIZING AI MODEL TRAINING")
    print("=" * 60)
    print("💡 Build your own custom AI model for the price of a meal")
    print("🔧 No more API dependencies - own your models")
    print("⚡ Fast, efficient, and cost-effective fine-tuning")
    print("=" * 60)
    print()

def demo_model_catalog():
    """Demonstrate the curated model catalog"""
    print("📚 AVAILABLE MODELS")
    print("-" * 30)
    
    models = list_available_models()
    categories = {"small": [], "medium": [], "large": []}
    
    for name, info in models.items():
        if "mini" in name.lower() or "small" in name.lower():
            categories["small"].append((name, info))
        elif "medium" in name.lower() or "4k" in name.lower():
            categories["medium"].append((name, info))
        else:
            categories["large"].append((name, info))
    
    for category, model_list in categories.items():
        if model_list:
            print(f"\n{category.upper()} MODELS:")
            for name, info in model_list[:3]:  # Show top 3
                print(f"  • {name}: {info['params']} parameters, {info['gpu_memory_gb']}GB GPU")
                print(f"    {info['description']}")
    
    print(f"\n✅ Total: {len(models)} curated models available")
    print()

def demo_dataset_catalog():
    """Demonstrate the dataset catalog"""
    print("📊 AVAILABLE DATASETS")
    print("-" * 30)
    
    datasets = list_available_datasets()
    
    for category, category_datasets in datasets.items():
        print(f"\n{category.upper()}:")
        for name, info in list(category_datasets.items())[:2]:  # Show top 2
            print(f"  • {info['name']}: {info['description']}")
            print(f"    Size: {info['size']}")
    
    print(f"\n✅ Multiple dataset categories available")
    print()

def demo_quick_training():
    """Demonstrate quick training with a small model"""
    print("🏋️ QUICK TRAINING DEMO")
    print("-" * 30)
    print("Training a small model for demonstration...")
    print("(This will use a tiny model for speed)")
    print()
    
    try:
        # Use a very small model for demo
        print("🚀 Starting training...")
        start_time = time.time()
        
        trainer = quick_train(
            model_name="sshleifer/tiny-gpt2",  # Very small for demo
            dataset_name="wikitext",  # Valid dataset
            test_prompts=["Hello, how are you?", "What is AI?"],
            output_dir="./demo_output"
        )
        
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.1f} seconds!")
        print()
        
        # Demo the trained model
        print("💬 TESTING THE TRAINED MODEL")
        print("-" * 30)
        
        test_prompts = [
            "Hello, how are you?",
            "What is artificial intelligence?",
            "Tell me about machine learning."
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n{i}. You: {prompt}")
            response = trainer.chat(prompt, max_new_tokens=30)
            print(f"   Model: {response}")
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"📁 Model saved to: {trainer.save_model()}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("💡 This is expected on CPU - GPU training is much faster!")

def demo_advanced_usage():
    """Demonstrate advanced usage patterns"""
    print("\n🔧 ADVANCED USAGE PATTERNS")
    print("-" * 30)
    
    print("1. Custom Training Parameters:")
    print("   trainer = JetTrainer('microsoft/Phi-3-mini-4k-instruct', 'dataset')")
    print("   trainer.train(epochs=3, learning_rate=1e-4, batch_size=2)")
    print()
    
    print("2. Model Evaluation:")
    print("   results = trainer.evaluate(['prompt1', 'prompt2'], references=['ref1', 'ref2'])")
    print("   print(results['metrics'])  # ROUGE scores, perplexity")
    print()
    
    print("3. Model Information:")
    print("   info = trainer.get_model_info()")
    print("   print(f'Model: {info[\"parameters\"]} parameters')")
    print()

def demo_pricing():
    """Demonstrate the pricing model"""
    print("💰 PRICING MODEL")
    print("-" * 30)
    print("🎯 Our Mission: Democratize AI model ownership")
    print()
    print("💵 Pricing Tiers:")
    print("   • Starter ($30): 3B-20B models, small datasets")
    print("   • Pro ($50): 20B-70B models, medium datasets") 
    print("   • Enterprise ($100): 70B+ models, large datasets")
    print()
    print("⚡ Cost Comparison:")
    print("   • OpenAI API: $1000s/month for heavy usage")
    print("   • Jet AI: $30 one-time for your own model")
    print("   • ROI: Break even in days, not months")
    print()

def main():
    """Run the complete demo"""
    print_header()
    
    # Demo sections
    demo_model_catalog()
    demo_dataset_catalog()
    demo_quick_training()
    demo_advanced_usage()
    demo_pricing()
    
    print("\n" + "=" * 60)
    print("🎯 READY TO DEMOCRATIZE AI?")
    print("=" * 60)
    print("📦 Install: pip install jet-ai-sdk")
    print("🚀 Start: from jet import quick_train")
    print("💡 Docs: https://github.com/your-org/jet-ai")
    print("=" * 60)

if __name__ == "__main__":
    main()
