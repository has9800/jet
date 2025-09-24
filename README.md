<img src="/Jet-sdk.png" alt="Jet SDK logo" style="width: 100%; height: 100%" />

# Jet AI 🚀

**Democratize AI model ownership - Build your own custom models for the price of a meal**

Stop being a "wrapper" business. Own your AI models instead of paying thousands in API fees to Big Tech.

## ⚡ Quick Start

```python
from jet import quick_train

# Train your own model in 3 lines
trainer = quick_train(
    model_name="microsoft/Phi-3-mini-4k-instruct",
    dataset_name="databricks/databricks-dolly-15k"
)

# Chat with your model
response = trainer.chat("Hello, how are you?")
print(response)
```

## 🎯 Why Jet AI?

### **The Problem:**
- **API Dependency**: Most AI startups are just OpenAI/Anthropic wrappers
- **Expensive**: $1000s/month in API costs for heavy usage
- **No Control**: Vendor lock-in, rate limits, unpredictable costs
- **Generic Models**: One-size-fits-all, not tailored to your data

### **Our Solution:**
- **Own Your Models**: Complete control and independence
- **Affordable**: $30-100 per custom model (one-time cost)
- **Custom Training**: Models tailored to your specific data
- **No Vendor Lock-in**: Deploy anywhere, anytime

## 🚀 Features

### **Smart Model Selection**
- **Curated Catalog**: Phi-3, Phi-4, GPT-OSS models optimized for 1-GPU training
- **Auto-Detection**: Automatically selects the best model for your GPU
- **Size Validation**: Warns if model won't fit on your hardware

### **Intelligent Dataset Handling**
- **Auto-Formatting**: Works with any HuggingFace dataset format
- **Field Detection**: Automatically finds the right text fields
- **Smart Preprocessing**: Combines instruction/response pairs intelligently

### **Optimized Training**
- **Unsloth Engine**: 2-4x faster training with FlashAttention2
- **QLoRA**: 4-bit quantization for memory efficiency
- **Mixed Precision**: FP16/BF16 for optimal performance
- **Auto-Scaling**: Automatically uses available GPU resources

### **Production Ready**
- **MLflow Integration**: Track experiments and model versions
- **Easy Deployment**: Deploy with vLLM for high-throughput serving
- **Comprehensive Testing**: 100% test coverage, production-grade reliability

## 📦 Installation

```bash
pip install jet-ai-sdk
```

## 💻 Usage Examples

### **Quick Training (One-Liner)**
```python
from jet import quick_train

# Train and test in one line
trainer = quick_train(
    "microsoft/Phi-3-mini-4k-instruct",
    "databricks/databricks-dolly-15k",
    test_prompts=["What is AI?", "Explain machine learning"]
)
```

### **Advanced Training**
```python
from jet import JetTrainer

# Create trainer with custom settings
trainer = JetTrainer(
    model_name="microsoft/Phi-3-mini-4k-instruct",
    dataset_name="databricks/databricks-dolly-15k",
    output_dir="./my_model"
)

# Train with custom parameters
trainer.train(
    epochs=3,
    learning_rate=1e-4,
    batch_size=2,
    merge_weights=True  # Merge LoRA weights after training
)

# Evaluate your model
results = trainer.evaluate(
    test_prompts=["Test prompt 1", "Test prompt 2"],
    references=["Expected answer 1", "Expected answer 2"]
)

# Chat with your model
response = trainer.chat("Hello, how are you?")
```

### **Model Management**
```python
# Get model information
info = trainer.get_model_info()
print(f"Model: {info['parameters']} parameters")
print(f"GPU Memory: {info['gpu_memory_gb']}GB")

# Save model to specific location
trainer.save_model("./my_saved_model")
```

## 🎯 Pricing

| Tier | Price | Models | Datasets | Use Case |
|------|-------|--------|----------|----------|
| **Starter** | $30 | 3B-20B params | Up to 50K examples | Small businesses, experiments |
| **Pro** | $50 | 20B-70B params | Up to 200K examples | Growing companies, production |
| **Enterprise** | $100 | 70B+ params | Up to 1M examples | Large organizations, complex tasks |

**Compare to API costs:**
- OpenAI API: $1000s/month for heavy usage
- Jet AI: $30-100 one-time for your own model
- **ROI**: Break even in days, not months

## 🔧 Available Models

### **Small Models (3B-20B)**
- `microsoft/Phi-3-mini-4k-instruct` - 3.8B params, 4GB GPU
- `microsoft/Phi-3-mini-128k-instruct` - 3.8B params, 6GB GPU  
- `microsoft/Phi-3-small-8k-instruct` - 7B params, 6GB GPU

### **Medium Models (20B-70B)**
- `microsoft/Phi-3-medium-4k-instruct` - 14B params, 8GB GPU
- `microsoft/Phi-4-mini` - 12B params, 8GB GPU
- `microsoft/Phi-3-large-4k-instruct` - 28B params, 12GB GPU

### **Large Models (70B+)**
- `microsoft/Phi-4` - 20B params, 16GB GPU
- `gpt-oss:20b` - 20B params, 16GB GPU

## 📊 Available Datasets

- **Conversational**: Databricks Dolly, OpenAssistant
- **Code**: The Stack, Code Alpaca
- **General**: WikiText, BookCorpus
- **Custom**: Upload your own data

## 🚀 Deployment

### **Python SDK**
```python
# Deploy with vLLM for production
from jet.service.app import deploy_model

deploy_spec = {
    "model": "./my_trained_model",
    "api_key": "your-api-key",
    "port": 8000
}

deploy_model(deploy_spec)
```

### **REST API**
```bash
# Start the API server (for frontend integration)
jet api

# API Documentation: http://localhost:8000/docs
```

## 🧪 Testing

```bash
# Run the test suite
pytest tests/ -v

# Run the demo
python examples/demo.py
```

## 📚 Documentation

- [Quick Start Guide](examples/quickstart.py)
- [API Reference](docs/api.md)
- [Deployment Guide](docs/deployment.md)
- [Examples](examples/)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

Apache 2.0 - see [LICENSE](LICENSE) for details.

## 🎯 Mission

**Democratize AI model ownership** - Make custom AI models accessible to everyone, not just Big Tech.

---

**Ready to own your AI future?** 

```bash
pip install jet-ai-sdk
```

**Start building today!** 🚀
