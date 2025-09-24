# Jet AI API

FastAPI microservice for Jet AI platform integration.

## Features

- 🚀 **Model Training** - REST endpoints for fine-tuning
- 📊 **Real-time Progress** - WebSocket updates
- 🔧 **Model Management** - List models and datasets
- 🚀 **Model Deployment** - Deploy trained models

## Installation

```bash
# Install with API dependencies
pip install jet-ai-sdk[api]

# Or install all dependencies
pip install jet-ai-sdk
```

## Quick Start

### Start the API Server

```bash
# Using the CLI
jet api

# Or directly with Python
python -m jet.api.app

# With custom settings
jet api --host 0.0.0.0 --port 8000 --reload
```

### API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Health Check
```http
GET /health
```

### Models & Datasets
```http
GET /api/v1/models          # List available models
GET /api/v1/datasets        # List available datasets
GET /api/v1/models/{name}   # Get model details
```

### Training
```http
POST /api/v1/train          # Start training job
GET  /api/v1/train/{job_id} # Get training status
DELETE /api/v1/train/{job_id} # Cancel training job
```

### Deployment
```http
POST /api/v1/deploy         # Deploy trained model
```

### WebSocket
```http
WS /ws/{job_id}             # Real-time training updates
```

## Usage Examples

### Start Training

```python
import requests

# Start a training job
response = requests.post("http://localhost:8000/api/v1/train", json={
    "model_name": "microsoft/Phi-3-mini-4k-instruct",
    "dataset_name": "wikitext-2-raw-v1",
    "epochs": 1,
    "use_gpu": False,
    "test_prompts": [
        "Hello, how are you?",
        "What is artificial intelligence?"
    ]
})

job_data = response.json()
job_id = job_data["job_id"]
print(f"Training started: {job_id}")
```

### Monitor Progress

```python
# Check training status
response = requests.get(f"http://localhost:8000/api/v1/train/{job_id}")
status = response.json()
print(f"Status: {status['status']}")
print(f"Progress: {status['progress']:.1%}")
```

### WebSocket Updates

```python
import websocket
import json

def on_message(ws, message):
    data = json.loads(message)
    print(f"Update: {data}")

ws = websocket.WebSocketApp(f"ws://localhost:8000/ws/{job_id}")
ws.on_message = on_message
ws.run_forever()
```

## Configuration

### Environment Variables

- `JET_API_HOST` - Host to bind to (default: 0.0.0.0)
- `JET_API_PORT` - Port to bind to (default: 8000)
- `JET_API_WORKERS` - Number of worker processes (default: 1)
- `JET_API_RELOAD` - Enable auto-reload for development (default: false)

### CORS Configuration

The API includes CORS middleware configured for development. For production, update the CORS settings in `app.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.com"],  # Production domains
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
```

## Development

### Running in Development Mode

```bash
# With auto-reload
jet api --reload

# Or with uvicorn directly
uvicorn jet.api.app:app --reload --host 0.0.0.0 --port 8000
```

### Testing the API

```bash
# Run the demo script
python examples/api_demo.py

# Or test individual endpoints
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/models
```

## Production Deployment

### Using Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .
RUN pip install jet-ai-sdk[api]

EXPOSE 8000
CMD ["jet", "api", "--host", "0.0.0.0", "--port", "8000"]
```

### Using Gunicorn

```bash
pip install gunicorn
gunicorn jet.api.app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Error Handling

The API includes comprehensive error handling:

- **400 Bad Request** - Invalid request parameters
- **404 Not Found** - Resource not found
- **500 Internal Server Error** - Server-side errors

All errors include detailed error messages and suggestions for resolution.

## Rate Limiting

For production deployment, consider adding rate limiting middleware:

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/api/v1/train")
@limiter.limit("5/minute")  # 5 requests per minute
async def start_training(request: Request, training_request: TrainingRequest):
    # ... training logic
```

## Monitoring

### Health Checks

The `/health` endpoint provides basic health information:

```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### Logging

The API uses Python's standard logging module. Configure logging levels and handlers as needed for your deployment.

## Security

### Authentication

The API currently runs without authentication for development. For production:

1. Add authentication middleware
2. Implement API key validation
3. Add rate limiting
4. Use HTTPS
5. Validate all inputs

### Input Validation

All inputs are validated using Pydantic models. Invalid inputs will return detailed error messages.

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure to install with `pip install jet-ai-sdk[api]`
2. **Port Already in Use**: Change the port with `--port 8001`
3. **GPU Not Available**: Set `use_gpu: false` in training requests
4. **Memory Issues**: Use smaller models or reduce batch size

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

Apache 2.0 - see the main project LICENSE file for details.
