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
from ..vast_ai import VastAIClient, GPUInstance, InstanceStatus
from ..credits import CreditManager, CreditType

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

# Initialize services
try:
    vast_client = VastAIClient()
except ValueError:
    # VastAIClient will be None if no API key is provided
    vast_client = None
    print("Warning: VAST_API_KEY not found. Vast.ai features will be disabled.")

credit_manager = CreditManager()
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

# New Pydantic models for Vast.ai and Credits
class GPUInstanceRequest(BaseModel):
    gpu_type: str = Field(default="RTX 4090", description="Type of GPU to request")
    min_memory: int = Field(default=16, description="Minimum GPU memory in GB")
    max_price: float = Field(default=3.0, description="Maximum price per hour")
    min_reliability: float = Field(default=0.9, description="Minimum reliability score")

class CreditPurchaseRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    amount: float = Field(..., description="Amount of credits to purchase")
    description: str = Field(default="Credit purchase", description="Purchase description")

class VastTrainingRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    model_name: str = Field(..., description="Name of the model to train")
    dataset_name: str = Field(..., description="Name of the dataset")
    gpu_type: str = Field(default="RTX 4090", description="Preferred GPU type")
    epochs: int = Field(default=1, description="Number of training epochs")
    learning_rate: float = Field(default=2e-4, description="Learning rate")
    batch_size: int = Field(default=4, description="Batch size")
    max_seq_length: int = Field(default=512, description="Maximum sequence length")
    test_prompts: Optional[List[str]] = Field(default=None, description="Test prompts for evaluation")

# Vast.ai GPU Management Endpoints
@app.get("/api/v1/gpu/offers")
async def search_gpu_offers(
    gpu_type: str = "RTX 4090",
    min_memory: int = 16,
    max_price: float = 3.0,
    min_reliability: float = 0.9
):
    """Search for available GPU offers"""
    if not vast_client:
        raise HTTPException(status_code=503, detail="Vast.ai service not available. Please set VAST_API_KEY.")
    
    try:
        offers = vast_client.search_offers(
            gpu_type=gpu_type,
            min_memory=min_memory,
            max_price=max_price,
            min_reliability=min_reliability
        )
        return {"offers": offers, "count": len(offers)}
    except Exception as e:
        logger.error(f"Error searching GPU offers: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching offers: {e}")

@app.post("/api/v1/gpu/instances")
async def create_gpu_instance(request: GPUInstanceRequest):
    """Create a new GPU instance"""
    if not vast_client:
        raise HTTPException(status_code=503, detail="Vast.ai service not available. Please set VAST_API_KEY.")
    
    try:
        # Search for best offer
        offers = vast_client.search_offers(
            gpu_type=request.gpu_type,
            min_memory=request.min_memory,
            max_price=request.max_price,
            min_reliability=request.min_reliability
        )
        
        if not offers:
            raise HTTPException(status_code=404, detail="No suitable GPU offers found")
        
        # Create instance with best offer
        best_offer = offers[0]
        instance = vast_client.create_instance(best_offer["id"])
        
        if not instance:
            raise HTTPException(status_code=500, detail="Failed to create GPU instance")
        
        return {
            "instance_id": instance.id,
            "status": instance.status.value,
            "gpu_type": instance.gpu_type,
            "price_per_hour": instance.price_per_hour,
            "message": "GPU instance created successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating GPU instance: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating instance: {e}")

@app.get("/api/v1/gpu/instances")
async def list_gpu_instances():
    """List all GPU instances"""
    if not vast_client:
        raise HTTPException(status_code=503, detail="Vast.ai service not available. Please set VAST_API_KEY.")
    
    try:
        instances = vast_client.list_instances()
        return {
            "instances": [
                {
                    "id": instance.id,
                    "status": instance.status.value,
                    "gpu_type": instance.gpu_type,
                    "price_per_hour": instance.price_per_hour,
                    "ssh_host": instance.ssh_host,
                    "ssh_port": instance.ssh_port,
                    "created_at": instance.created_at
                }
                for instance in instances
            ],
            "count": len(instances)
        }
    except Exception as e:
        logger.error(f"Error listing GPU instances: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing instances: {e}")

@app.get("/api/v1/gpu/instances/{instance_id}")
async def get_gpu_instance(instance_id: str):
    """Get GPU instance details"""
    if not vast_client:
        raise HTTPException(status_code=503, detail="Vast.ai service not available. Please set VAST_API_KEY.")
    
    try:
        instance = vast_client.get_instance(instance_id)
        if not instance:
            raise HTTPException(status_code=404, detail="Instance not found")
        
        return {
            "id": instance.id,
            "status": instance.status.value,
            "gpu_type": instance.gpu_type,
            "price_per_hour": instance.price_per_hour,
            "ssh_host": instance.ssh_host,
            "ssh_port": instance.ssh_port,
            "created_at": instance.created_at,
            "error_message": instance.error_message
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting GPU instance: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting instance: {e}")

@app.delete("/api/v1/gpu/instances/{instance_id}")
async def terminate_gpu_instance(instance_id: str):
    """Terminate a GPU instance"""
    if not vast_client:
        raise HTTPException(status_code=503, detail="Vast.ai service not available. Please set VAST_API_KEY.")
    
    try:
        success = vast_client.terminate_instance(instance_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to terminate instance")
        
        return {"message": f"Instance {instance_id} terminated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error terminating GPU instance: {e}")
        raise HTTPException(status_code=500, detail=f"Error terminating instance: {e}")

# Credit Management Endpoints
@app.get("/api/v1/credits/{user_id}")
async def get_user_credits(user_id: str):
    """Get user credit balance"""
    try:
        balance = credit_manager.check_balance(user_id)
        return balance
    except Exception as e:
        logger.error(f"Error getting user credits: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting credits: {e}")

@app.post("/api/v1/credits/purchase")
async def purchase_credits(request: CreditPurchaseRequest):
    """Purchase credits for user"""
    try:
        success = credit_manager.add_credits(
            user_id=request.user_id,
            amount=request.amount,
            description=request.description
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to add credits")
        
        balance = credit_manager.check_balance(request.user_id)
        return {
            "message": f"Successfully added {request.amount} credits",
            "balance": balance
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error purchasing credits: {e}")
        raise HTTPException(status_code=500, detail=f"Error purchasing credits: {e}")

@app.get("/api/v1/credits/{user_id}/history")
async def get_credit_history(user_id: str, limit: int = 10):
    """Get user credit history"""
    try:
        history = credit_manager.get_credit_history(user_id, limit)
        return {"history": history, "count": len(history)}
    except Exception as e:
        logger.error(f"Error getting credit history: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting history: {e}")

# Enhanced Training with Vast.ai and Credits
@app.post("/api/v1/train/vast", response_model=TrainingResponse)
async def start_vast_training(request: VastTrainingRequest, background_tasks: BackgroundTasks):
    """Start training on Vast.ai GPU with credit checking"""
    if not vast_client:
        raise HTTPException(status_code=503, detail="Vast.ai service not available. Please set VAST_API_KEY.")
    
    try:
        # Check user credits first
        user_credits = credit_manager.get_user_credits(request.user_id)
        estimated_hours = request.epochs * 0.5  # Rough estimate
        estimated_cost = credit_manager.calculate_training_cost(request.gpu_type, estimated_hours)
        
        if not user_credits.can_afford(estimated_cost):
            raise HTTPException(
                status_code=402, 
                detail=f"Insufficient credits. Need {estimated_cost:.2f}, have {user_credits.available_credits:.2f}"
            )
        
        # Create GPU instance
        gpu_request = GPUInstanceRequest(
            gpu_type=request.gpu_type,
            min_memory=16,
            max_price=3.0
        )
        
        offers = vast_client.search_offers(
            gpu_type=request.gpu_type,
            min_memory=16,
            max_price=3.0
        )
        
        if not offers:
            raise HTTPException(status_code=404, detail="No suitable GPU offers found")
        
        # Create instance
        instance = vast_client.create_instance(offers[0]["id"])
        if not instance:
            raise HTTPException(status_code=500, detail="Failed to create GPU instance")
        
        # Create job
        job_id = str(uuid.uuid4())
        active_jobs[job_id] = {
            "id": job_id,
            "user_id": request.user_id,
            "status": "creating",
            "message": "Creating GPU instance...",
            "progress": 0.0,
            "start_time": datetime.now(),
            "instance_id": instance.id,
            "gpu_type": instance.gpu_type,
            "estimated_cost": estimated_cost
        }
        
        # Start training in background
        background_tasks.add_task(
            run_vast_training,
            job_id,
            request,
            instance.id
        )
        
        return TrainingResponse(
            job_id=job_id,
            status="creating",
            message="GPU instance created, starting training...",
            progress=0.0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting Vast training: {e}")
        raise HTTPException(status_code=500, detail=f"Error starting training: {e}")

async def run_vast_training(job_id: str, request: VastTrainingRequest, instance_id: str):
    """Run training on Vast.ai GPU instance"""
    try:
        # Wait for instance to be ready
        active_jobs[job_id]["message"] = "Waiting for GPU instance to be ready..."
        active_jobs[job_id]["progress"] = 0.1
        
        # Monitor instance status
        max_wait = 300  # 5 minutes
        wait_time = 0
        while wait_time < max_wait:
            instance = vast_client.get_instance(instance_id)
            if instance and instance.status == InstanceStatus.RUNNING:
                break
            elif instance and instance.status == InstanceStatus.ERROR:
                raise Exception(f"GPU instance failed: {instance.error_message}")
            
            await asyncio.sleep(10)
            wait_time += 10
        
        if wait_time >= max_wait:
            raise Exception("GPU instance took too long to start")
        
        # Update progress
        active_jobs[job_id]["status"] = "running"
        active_jobs[job_id]["message"] = "GPU instance ready, starting training..."
        active_jobs[job_id]["progress"] = 0.2
        
        # Run training job on instance
        success = vast_client.run_training_job(
            instance_id=instance_id,
            model_name=request.model_name,
            dataset_name=request.dataset_name,
            epochs=request.epochs,
            learning_rate=request.learning_rate
        )
        
        if not success:
            raise Exception("Failed to run training job on GPU instance")
        
        # Update progress
        active_jobs[job_id]["progress"] = 0.8
        active_jobs[job_id]["message"] = "Training completed on GPU..."
        
        # Calculate actual cost and deduct credits
        actual_hours = 1.0  # Placeholder - would track actual time
        actual_cost = credit_manager.calculate_training_cost(request.gpu_type, actual_hours)
        credit_manager.deduct_training_credits(request.user_id, request.gpu_type, actual_hours)
        
        # Complete job
        active_jobs[job_id]["status"] = "completed"
        active_jobs[job_id]["progress"] = 1.0
        active_jobs[job_id]["message"] = "Training completed successfully!"
        active_jobs[job_id]["actual_cost"] = actual_cost
        active_jobs[job_id]["end_time"] = datetime.now()
        
        # Terminate instance
        vast_client.terminate_instance(instance_id)
        
        logger.info(f"Vast training job {job_id} completed successfully")
        
    except Exception as e:
        error_msg = f"Vast training failed: {str(e)}"
        logger.error(f"Vast training job {job_id} failed: {e}")
        logger.error(traceback.format_exc())
        
        active_jobs[job_id]["status"] = "failed"
        active_jobs[job_id]["message"] = error_msg
        active_jobs[job_id]["error"] = str(e)
        active_jobs[job_id]["end_time"] = datetime.now()
        
        # Terminate instance on failure
        try:
            vast_client.terminate_instance(instance_id)
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
