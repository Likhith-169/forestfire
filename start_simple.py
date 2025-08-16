#!/usr/bin/env python3
"""
Simple Startup Script for Forest Fire Detection API

This script starts just the FastAPI backend server.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

if __name__ == "__main__":
    try:
        import uvicorn
        from api.simple_main import app
        
        print("🚀 Starting Forest Fire Detection API Server")
        print("=" * 50)
        print("🌐 Server will be available at: http://127.0.0.1:8000")
        print("📱 Web interface: Open web/index.html in your browser")
        print("🔧 API docs: http://127.0.0.1:8000/docs")
        print("\nPress Ctrl+C to stop the server")
        
        uvicorn.run(
            "api.simple_main:app",
            host="127.0.0.1",
            port=8000,
            reload=False,
            log_level="info"
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install dependencies: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        print("Check that all dependencies are installed")
