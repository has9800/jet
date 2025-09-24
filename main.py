#!/usr/bin/env python3
"""
Jet AI API Server - Render Deployment Entry Point
"""
import sys
import os
from pathlib import Path

# Add src to Python path so we can import jet
sys.path.insert(0, str(Path(__file__).parent / "src"))

from jet.api.app import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
