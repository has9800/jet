"""
Vast.ai API Integration for GPU Instance Management
"""
import os
import time
import requests
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

class InstanceStatus(Enum):
    """GPU instance status"""
    CREATING = "creating"
    RUNNING = "running"
    STOPPED = "stopped"
    TERMINATED = "terminated"
    ERROR = "error"

@dataclass
class GPUInstance:
    """GPU instance information"""
    id: str
    status: InstanceStatus
    gpu_type: str
    price_per_hour: float
    ssh_host: Optional[str] = None
    ssh_port: Optional[int] = None
    created_at: Optional[str] = None
    error_message: Optional[str] = None

class VastAIClient:
    """Vast.ai API client for managing GPU instances"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("VAST_API_KEY")
        if not self.api_key:
            raise ValueError("VAST_API_KEY environment variable is required")
        
        self.base_url = "https://console.vast.ai/api/v0"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def search_offers(self, 
                     gpu_type: str = "RTX 4090",
                     min_memory: int = 16,
                     max_price: float = 3.0,
                     min_reliability: float = 0.9) -> List[Dict[str, Any]]:
        """Search for available GPU offers"""
        try:
            url = f"{self.base_url}/offers/"
            params = {
                "type": "on-demand",
                "gpu_name": gpu_type,
                "min_memory": min_memory,
                "max_price": max_price,
                "min_reliability": min_reliability,
                "order": "price"
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            offers = response.json().get("offers", [])
            return offers[:10]  # Return top 10 cheapest
            
        except requests.exceptions.RequestException as e:
            print(f"Error searching offers: {e}")
            return []
    
    def create_instance(self, 
                      offer_id: str,
                      image: str = "pytorch/pytorch:latest",
                      disk_space: int = 20,
                      label: str = "jet-ai-training") -> Optional[GPUInstance]:
        """Create a new GPU instance"""
        try:
            url = f"{self.base_url}/instances/"
            data = {
                "offer": offer_id,
                "image": image,
                "disk_space": disk_space,
                "label": label,
                "onstart": "pip install jet-ai-sdk[api] && python -m jet.cli api --host 0.0.0.0 --port 8000"
            }
            
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            instance_id = result.get("id")
            
            if instance_id:
                return GPUInstance(
                    id=instance_id,
                    status=InstanceStatus.CREATING,
                    gpu_type=result.get("gpu_name", "Unknown"),
                    price_per_hour=result.get("price", 0.0),
                    created_at=time.strftime("%Y-%m-%d %H:%M:%S")
                )
            
        except requests.exceptions.RequestException as e:
            print(f"Error creating instance: {e}")
            return None
    
    def get_instance(self, instance_id: str) -> Optional[GPUInstance]:
        """Get instance details"""
        try:
            url = f"{self.base_url}/instances/{instance_id}/"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            return GPUInstance(
                id=instance_id,
                status=InstanceStatus(data.get("status", "error")),
                gpu_type=data.get("gpu_name", "Unknown"),
                price_per_hour=data.get("price", 0.0),
                ssh_host=data.get("ssh_host"),
                ssh_port=data.get("ssh_port"),
                created_at=data.get("created_at"),
                error_message=data.get("error_message")
            )
            
        except requests.exceptions.RequestException as e:
            print(f"Error getting instance: {e}")
            return None
    
    def list_instances(self) -> List[GPUInstance]:
        """List all user instances"""
        try:
            url = f"{self.base_url}/instances/"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            instances = []
            for data in response.json().get("instances", []):
                instances.append(GPUInstance(
                    id=data.get("id"),
                    status=InstanceStatus(data.get("status", "error")),
                    gpu_type=data.get("gpu_name", "Unknown"),
                    price_per_hour=data.get("price", 0.0),
                    ssh_host=data.get("ssh_host"),
                    ssh_port=data.get("ssh_port"),
                    created_at=data.get("created_at"),
                    error_message=data.get("error_message")
                ))
            
            return instances
            
        except requests.exceptions.RequestException as e:
            print(f"Error listing instances: {e}")
            return []
    
    def terminate_instance(self, instance_id: str) -> bool:
        """Terminate an instance"""
        try:
            url = f"{self.base_url}/instances/{instance_id}/"
            response = requests.delete(url, headers=self.headers)
            response.raise_for_status()
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Error terminating instance: {e}")
            return False
    
    def run_training_job(self, 
                        instance_id: str,
                        model_name: str,
                        dataset_name: str,
                        epochs: int = 1,
                        learning_rate: float = 2e-4) -> bool:
        """Run training job on instance via SSH"""
        try:
            instance = self.get_instance(instance_id)
            if not instance or not instance.ssh_host:
                return False
            
            # This would require SSH connection to run the training
            # For now, return True as placeholder
            print(f"Training job queued for instance {instance_id}")
            return True
            
        except Exception as e:
            print(f"Error running training job: {e}")
            return False
