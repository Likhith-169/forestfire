# Quick Start Guide - Forest Fire Detection System

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
# Install minimal dependencies (recommended for first run)
pip install -r requirements-minimal.txt

# OR install full dependencies (includes ML libraries)
pip install -r requirements.txt
```

### Step 2: Start the API Server
```bash
# Start the simple API server
python start_simple.py
```

You should see:
```
🚀 Starting Forest Fire Detection API Server
==================================================
🌐 Server will be available at: http://127.0.0.1:8000
📱 Web interface: Open web/index.html in your browser
🔧 API docs: http://127.0.0.1:8000/docs
```

### Step 3: Open the Web Interface
- Open `web/index.html` in your web browser
- OR navigate to `http://127.0.0.1:8000` if using the development server

## 🔧 Troubleshooting

### "Connection Refused" Error
- Make sure the API server is running (`python start_simple.py`)
- Check that port 8000 is not blocked by firewall
- Verify the server started without errors

### Import Errors
Run the dependency checker:
```bash
python test_deps.py
```

### Missing Dependencies
Install missing packages:
```bash
pip install fastapi uvicorn pydantic numpy pandas
```

## 📁 Project Structure
```
forestfire/
├── src/api/main.py          # FastAPI backend
├── web/index.html           # Web interface
├── start_simple.py          # Simple startup script
├── requirements-minimal.txt  # Minimal dependencies
└── config/                  # Configuration files
```

## 🌐 API Endpoints
- **Health Check**: `GET /health`
- **API Info**: `GET /`
- **Fire Detection**: `POST /api/v1/detect`
- **API Docs**: `GET /docs` (Swagger UI)

## 🎯 Next Steps
1. **Test the system** with the web interface
2. **Configure settings** in `config/config.yaml`
3. **Add ML models** for improved detection
4. **Deploy to production** with proper configuration

## 📞 Need Help?
- Check the logs in the terminal where you started the server
- Run `python test_deps.py` to diagnose dependency issues
- Review the full README.md for detailed documentation
