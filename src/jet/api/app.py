"""
Jet AI FastAPI Application

A microservice API for the Jet AI platform, providing endpoints for:
- Model training and fine-tuning
- Model deployment and inference
- GPU resource management via Vast.ai
- Real-time progress tracking
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import asyncio
import uuid
import json
import logging
from datetime import datetime
import traceback

# Import existing Jet AI modules
from ..train import train_with_options
from ..models import CURATED_MODELS, CURATED_DATASETS, get_model_info, validate_model_for_gpu
from ..sdk.training import JetTrainer
from ..eval import Evaluator

# Configure logging
logger = logging.getLogger("jet.api")

# Create FastAPI app
app = FastAPI(
    title="Jet AI API",
    description="Fine-tune and deploy open-weight AI models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for jobs (use Redis in production)
active_jobs: Dict[str, Dict[str, Any]] = {}
websocket_connections: List[WebSocket] = []

# Pydantic models for API
class TrainingRequest(BaseModel):
    model_name: str = Field(..., description="Name of the model to train")
    dataset_name: str = Field(..., description="Name of the dataset to use")
    epochs: int = Field(1, description="Number of training epochs")
    learning_rate: Optional[float] = Field(None, description="Learning rate (auto if not specified)")
    batch_size: Optional[int] = Field(None, description="Batch size (auto if not specified)")
    max_seq_length: Optional[int] = Field(None, description="Max sequence length (auto if not specified)")
    output_dir: Optional[str] = Field(None, description="Output directory for the trained model")
    use_gpu: bool = Field(True, description="Whether to use GPU for training")
    test_prompts: Optional[List[str]] = Field(None, description="Test prompts for evaluation")

class TrainingResponse(BaseModel):
    job_id: str
    status: str
    message: str
    estimated_duration: Optional[str] = None

class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, running, completed, failed
    progress: float  # 0.0 to 1.0
    message: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    model_path: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    error: Optional[str] = None

class DeployRequest(BaseModel):
    model_path: str = Field(..., description="Path to the trained model")
    api_key: str = Field(..., description="API key for the deployed model")
    port: int = Field(8000, description="Port for the deployed model")
    lora_adapters: Optional[Dict[str, str]] = Field(None, description="LoRA adapters to use")

class DeployResponse(BaseModel):
    deployment_id: str
    status: str
    endpoint: Optional[str] = None
    command: Optional[str] = None

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Model and dataset listing endpoints
@app.get("/api/v1/models")
async def list_models():
    """List all available models"""
    return {
        "models": CURATED_MODELS,
        "total": len(CURATED_MODELS)
    }

@app.get("/api/v1/datasets")
async def list_datasets():
    """List all available datasets"""
    return {
        "datasets": CURATED_DATASETS,
        "total": len(CURATED_DATASETS)
    }

@app.get("/api/v1/models/{model_name}")
async def get_model_info_endpoint(model_name: str):
    """Get detailed information about a specific model"""
    try:
        model_info = get_model_info(model_name)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        return model_info
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

# Training endpoints
@app.post("/api/v1/train", response_model=TrainingResponse)
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start a new training job"""
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Validate model
        model_found = False
        for category, models in CURATED_MODELS.items():
            if request.model_name in models:
                model_found = True
                break
        
        if not model_found:
            available_models = []
            for category, models in CURATED_MODELS.items():
                available_models.extend(models.keys())
            raise HTTPException(
                status_code=400, 
                detail=f"Model '{request.model_name}' not in curated models. Available: {available_models[:10]}..."
            )
        
        # Validate dataset
        dataset_found = False
        for category, datasets in CURATED_DATASETS.items():
            if request.dataset_name in datasets:
                dataset_found = True
                break
        
        if not dataset_found:
            available_datasets = []
            for category, datasets in CURATED_DATASETS.items():
                available_datasets.extend(datasets.keys())
            raise HTTPException(
                status_code=400,
                detail=f"Dataset '{request.dataset_name}' not in curated datasets. Available: {available_datasets[:10]}..."
            )
        
        # Check GPU availability if requested
        if request.use_gpu:
            model_info = get_model_info(request.model_name)
            if not validate_model_for_gpu(request.model_name):
                raise HTTPException(
                    status_code=400,
                    detail=f"Model '{request.model_name}' requires {model_info.gpu_memory_gb}GB GPU memory. Consider using a smaller model or CPU training."
                )
        
        # Create job entry
        active_jobs[job_id] = {
            "status": "pending",
            "progress": 0.0,
            "message": "Job queued",
            "start_time": datetime.now(),
            "end_time": None,
            "model_path": None,
            "metrics": None,
            "error": None,
            "request": request.dict()
        }
        
        # Start training in background
        background_tasks.add_task(run_training_job, job_id, request)
        
        # Estimate duration (rough calculation)
        estimated_duration = "5-15 minutes" if request.use_gpu else "30-60 minutes"
        
        return TrainingResponse(
            job_id=job_id,
            status="pending",
            message="Training job started",
            estimated_duration=estimated_duration
        )
        
    except Exception as e:
        logger.error(f"Error starting training job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/train/{job_id}", response_model=JobStatus)
async def get_training_status(job_id: str):
    """Get the status of a training job"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = active_jobs[job_id]
    return JobStatus(**job)

@app.delete("/api/v1/train/{job_id}")
async def cancel_training_job(job_id: str):
    """Cancel a training job"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = active_jobs[job_id]
    if job["status"] in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Cannot cancel completed or failed job")
    
    job["status"] = "cancelled"
    job["message"] = "Job cancelled by user"
    job["end_time"] = datetime.now()
    
    return {"message": "Job cancelled successfully"}

# Deployment endpoints
@app.post("/api/v1/deploy", response_model=DeployResponse)
async def deploy_model(request: DeployRequest):
    """Deploy a trained model"""
    try:
        # Generate deployment ID
        deployment_id = str(uuid.uuid4())
        
        # For now, return the deployment command
        # In production, this would integrate with Vast.ai or your GPU infrastructure
        lora_args = ""
        if request.lora_adapters:
            lora_args = " " + " ".join([f"--lora-modules {k}={v}" for k, v in request.lora_adapters.items()])
        
        command = (
            f"docker run --rm --gpus all -p {request.port}:8000 server-vllm:latest "
            f"python3 -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8000 "
            f"--api-key {request.api_key} --model {request.model_path}{lora_args}"
        )
        
        return DeployResponse(
            deployment_id=deployment_id,
            status="ready",
            command=command
        )
        
    except Exception as e:
        logger.error(f"Error deploying model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time updates
@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time training progress"""
    await websocket.accept()
    websocket_connections.append(websocket)
    
    try:
        while True:
            # Send current job status
            if job_id in active_jobs:
                job = active_jobs[job_id]
                await websocket.send_text(json.dumps({
                    "job_id": job_id,
                    "status": job["status"],
                    "progress": job["progress"],
                    "message": job["message"],
                    "timestamp": datetime.now().isoformat()
                }))
            
            # Wait for next update
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)

# Background task for training
async def run_training_job(job_id: str, request: TrainingRequest):
    """Run the actual training job in the background"""
    try:
        # Update job status
        active_jobs[job_id]["status"] = "running"
        active_jobs[job_id]["message"] = "Starting training..."
        active_jobs[job_id]["progress"] = 0.1
        
        # Create trainer
        trainer = JetTrainer(
            model_name=request.model_name,
            dataset_name=request.dataset_name,
            output_dir=request.output_dir or f"./jet_outputs/{job_id}"
        )
        
        # Update progress
        active_jobs[job_id]["progress"] = 0.2
        active_jobs[job_id]["message"] = "Loading dataset and model..."
        
        # Start training
        active_jobs[job_id]["progress"] = 0.3
        active_jobs[job_id]["message"] = "Training in progress..."
        
        # Run training
        trainer.train(
            epochs=request.epochs,
            learning_rate=request.learning_rate,
            batch_size=request.batch_size,
            max_seq_length=request.max_seq_length
        )
        
        # Update progress
        active_jobs[job_id]["progress"] = 0.8
        active_jobs[job_id]["message"] = "Training completed, saving model..."
        
        # Save model
        model_path = trainer.save_model()
        active_jobs[job_id]["model_path"] = model_path
        
        # Run evaluation if test prompts provided
        if request.test_prompts:
            active_jobs[job_id]["message"] = "Running evaluation..."
            evaluator = Evaluator(model_path)
            metrics = evaluator.evaluate(request.test_prompts)
            active_jobs[job_id]["metrics"] = metrics
        
        # Complete job
        active_jobs[job_id]["status"] = "completed"
        active_jobs[job_id]["progress"] = 1.0
        active_jobs[job_id]["message"] = "Training completed successfully!"
        active_jobs[job_id]["end_time"] = datetime.now()
        
        logger.info(f"Training job {job_id} completed successfully")
        
    except Exception as e:
        # Handle training errors
        error_msg = f"Training failed: {str(e)}"
        logger.error(f"Training job {job_id} failed: {e}")
        logger.error(traceback.format_exc())
        
        active_jobs[job_id]["status"] = "failed"
        active_jobs[job_id]["message"] = error_msg
        active_jobs[job_id]["error"] = str(e)
        active_jobs[job_id]["end_time"] = datetime.now()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
