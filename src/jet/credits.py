"""
Credit System for Jet AI Platform
"""
import os
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

class CreditType(Enum):
    """Types of credits"""
    TRAINING = "training"
    INFERENCE = "inference"
    STORAGE = "storage"

@dataclass
class UserCredits:
    """User credit information"""
    user_id: str
    total_credits: float
    used_credits: float
    last_updated: str
    credit_history: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.credit_history is None:
            self.credit_history = []
    
    @property
    def available_credits(self) -> float:
        """Available credits for use"""
        return self.total_credits - self.used_credits
    
    def can_afford(self, cost: float) -> bool:
        """Check if user can afford the cost"""
        return self.available_credits >= cost
    
    def deduct_credits(self, amount: float, description: str = "") -> bool:
        """Deduct credits from user account"""
        if not self.can_afford(amount):
            return False
        
        self.used_credits += amount
        self.last_updated = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Add to history
        self.credit_history.append({
            "timestamp": self.last_updated,
            "type": "deduction",
            "amount": amount,
            "description": description,
            "balance": self.available_credits
        })
        
        return True
    
    def add_credits(self, amount: float, description: str = "") -> None:
        """Add credits to user account"""
        self.total_credits += amount
        self.last_updated = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Add to history
        self.credit_history.append({
            "timestamp": self.last_updated,
            "type": "addition",
            "amount": amount,
            "description": description,
            "balance": self.available_credits
        })

class CreditManager:
    """Manages user credits and billing"""
    
    def __init__(self, storage_file: str = "credits.json"):
        self.storage_file = storage_file
        self.credits_db = self._load_credits()
        
        # Pricing configuration
        self.pricing = {
            "RTX 4090": 2.50,  # per hour
            "RTX 4080": 2.00,
            "RTX 3090": 1.80,
            "A100": 3.50,
            "H100": 4.00,
            "storage": 0.10,  # per GB per month
            "inference": 0.01  # per 1000 tokens
        }
    
    def _load_credits(self) -> Dict[str, UserCredits]:
        """Load credits from storage"""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)
                    credits_db = {}
                    for user_id, user_data in data.items():
                        credits_db[user_id] = UserCredits(**user_data)
                    return credits_db
            except Exception as e:
                print(f"Error loading credits: {e}")
        
        return {}
    
    def _save_credits(self) -> None:
        """Save credits to storage"""
        try:
            data = {}
            for user_id, user_credits in self.credits_db.items():
                data[user_id] = asdict(user_credits)
            
            with open(self.storage_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving credits: {e}")
    
    def get_user_credits(self, user_id: str) -> UserCredits:
        """Get user credits, create if doesn't exist"""
        if user_id not in self.credits_db:
            # New user gets 10 free credits
            self.credits_db[user_id] = UserCredits(
                user_id=user_id,
                total_credits=10.0,
                used_credits=0.0,
                last_updated=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            self._save_credits()
        
        return self.credits_db[user_id]
    
    def add_credits(self, user_id: str, amount: float, description: str = "Credit purchase") -> bool:
        """Add credits to user account"""
        user_credits = self.get_user_credits(user_id)
        user_credits.add_credits(amount, description)
        self._save_credits()
        return True
    
    def calculate_training_cost(self, gpu_type: str, hours: float) -> float:
        """Calculate training cost based on GPU type and hours"""
        base_price = self.pricing.get(gpu_type, 2.0)  # Default price
        return base_price * hours
    
    def calculate_storage_cost(self, gb_size: float, days: float) -> float:
        """Calculate storage cost"""
        daily_rate = self.pricing["storage"] / 30  # Convert monthly to daily
        return daily_rate * gb_size * days
    
    def deduct_training_credits(self, user_id: str, gpu_type: str, hours: float) -> bool:
        """Deduct credits for training"""
        cost = self.calculate_training_cost(gpu_type, hours)
        user_credits = self.get_user_credits(user_id)
        
        if user_credits.deduct_credits(cost, f"Training on {gpu_type} for {hours:.2f} hours"):
            self._save_credits()
            return True
        return False
    
    def deduct_storage_credits(self, user_id: str, gb_size: float, days: float) -> bool:
        """Deduct credits for storage"""
        cost = self.calculate_storage_cost(gb_size, days)
        user_credits = self.get_user_credits(user_id)
        
        if user_credits.deduct_credits(cost, f"Storage {gb_size:.2f}GB for {days:.2f} days"):
            self._save_credits()
            return True
        return False
    
    def check_balance(self, user_id: str) -> Dict[str, Any]:
        """Check user credit balance"""
        user_credits = self.get_user_credits(user_id)
        return {
            "user_id": user_id,
            "total_credits": user_credits.total_credits,
            "used_credits": user_credits.used_credits,
            "available_credits": user_credits.available_credits,
            "last_updated": user_credits.last_updated
        }
    
    def get_credit_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get user credit history"""
        user_credits = self.get_user_credits(user_id)
        return user_credits.credit_history[-limit:]
