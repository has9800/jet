"""
Jet AI API CLI

Command-line interface for running the Jet AI API server.
"""

import uvicorn
import argparse
import sys
from .app import app

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Jet AI API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    print(f"🚀 Jet AI API Server starting on http://{args.host}:{args.port}")
    print(f"📚 Documentation: http://{args.host}:{args.port}/docs")
    
    try:
        uvicorn.run(
            "jet.api.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down Jet AI API Server...")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
