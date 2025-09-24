#!/usr/bin/env python3
"""
Jet AI - Google Colab Demo Script

Copy and paste this code into a Google Colab notebook to test Jet AI.
Perfect for demos and experimentation.
"""

# Cell 1: Installation
print("📦 Installing Jet AI...")
print("!pip install jet-ai-sdk")
print("!pip install accelerate bitsandbytes")
print()

# Cell 2: Import and Setup
print("🔧 Import and Setup")
print("=" * 30)
print("""
from jet import quick_train, list_available_models, list_available_datasets
import torch

# Check if GPU is available
print(f"🖥️  Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

print("\\n📚 Available Models:")
models = list_available_models()
for name, info in list(models.items())[:5]:  # Show first 5
    print(f"  • {name}: {info['params']} parameters, {info['gpu_memory_gb']}GB GPU")
""")

# Cell 3: Quick Training Demo
print("⚡ Quick Training Demo")
print("=" * 30)
print("""
# Quick training with a small model
print("🚀 Starting Jet AI training...")
print("This will train a small model for demonstration")
print()

try:
    # Use a very small model for Colab demo
    trainer = quick_train(
        model_name="sshleifer/tiny-gpt2",  # Very small for demo
        dataset_name="wikitext",  # Small dataset
        test_prompts=[
            "Hello, how are you?",
            "What is artificial intelligence?",
            "Tell me about machine learning."
        ],
        output_dir="./colab_demo_output"
    )
    
    print("\\n✅ Training completed successfully!")
    print(f"📁 Model saved to: {trainer.save_model()}")
    
except Exception as e:
    print(f"❌ Training failed: {e}")
    print("💡 This is expected on CPU - GPU training is much faster!")
""")

# Cell 4: Chat with Model
print("💬 Chat with Your Model")
print("=" * 30)
print("""
# Test the trained model
print("💬 Testing the trained model...")
print("=" * 50)

test_prompts = [
    "Hello, how are you?",
    "What is artificial intelligence?",
    "Tell me about machine learning.",
    "Explain deep learning.",
    "What is a neural network?"
]

for i, prompt in enumerate(test_prompts, 1):
    print(f"\\n{i}. You: {prompt}")
    try:
        response = trainer.chat(prompt, max_new_tokens=50)
        print(f"   Model: {response}")
    except Exception as e:
        print(f"   Model: [Error: {e}]")

print("\\n🎉 Demo completed!")
""")

# Cell 5: Advanced Features
print("🔧 Advanced Features")
print("=" * 30)
print("""
from jet import JetTrainer

print("🔧 Advanced Jet AI Features")
print("=" * 40)

# Show available datasets
print("\\n📊 Available Datasets:")
datasets = list_available_datasets()
for category, category_datasets in datasets.items():
    print(f"\\n{category.upper()}:")
    for name, info in list(category_datasets.items())[:2]:
        print(f"  • {info['name']}: {info['description']}")

# Show model information
print("\\n🔍 Model Information:")
info = trainer.get_model_info()
print(f"  • Model: {info['parameters']} parameters")
print(f"  • GPU Memory: {info['gpu_memory_gb']}GB")
print(f"  • Max Sequence Length: {info['max_seq_length']}")
print(f"  • Description: {info['description']}")
""")

# Cell 6: Pricing Demo
print("💰 Pricing Model")
print("=" * 30)
print("""
print("💰 Jet AI Pricing Model")
print("=" * 30)
print()
print("🎯 Our Mission: Democratize AI model ownership")
print()
print("💵 Pricing Tiers:")
print("  • Starter ($30): 3B-20B models, small datasets")
print("  • Pro ($50): 20B-70B models, medium datasets")
print("  • Enterprise ($100): 70B+ models, large datasets")
print()
print("⚡ Cost Comparison:")
print("  • OpenAI API: $1000s/month for heavy usage")
print("  • Jet AI: $30-100 one-time for your own model")
print("  • ROI: Break even in days, not months")
print()
print("🚀 Ready to democratize AI?")
print("   pip install jet-ai-sdk")
""")

print("\n" + "=" * 60)
print("🎯 COPY THESE CELLS INTO GOOGLE COLAB")
print("=" * 60)
print("1. Go to colab.research.google.com")
print("2. Create a new notebook")
print("3. Copy each cell above into separate cells")
print("4. Run each cell to see Jet AI in action!")
print("=" * 60)
