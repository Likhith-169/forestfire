# Forest Fire Detection API

A FastAPI-based service for detecting forest fires using satellite imagery analysis.

## Setup
```bash
pip install -r requirements.txt
```

## Development
```bash
uvicorn api.index:app --reload
```

## Production
Deployed on Vercel
```bash
python start_simple.py
```

### 3. Open the Web Interface
- Open `web/index.html` in your web browser
- OR navigate to `http://127.0.0.1:8000` if using the development server

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Ingest   │───▶│   Processing    │───▶│   Alert System  │
│                 │    │                 │    │                 │
│ • VIIRS/MODIS   │    │ • Cloud Masking │    │ • Dashboard     │
│ • Sentinel-2    │    │ • Index Calc    │    │ • Notifications │
│ • Landsat       │    │ • ML Refinement │    │ • API Endpoints │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 Core Detection Logic

### A) Hotspot Detection (Active Fire)
- Thermal anomaly detection from VIIRS/MODIS
- Brightness Temperature analysis
- MIR/NIR ratio computation

### B) Optical Confirmation
- **NBR**: Normalized Burn Ratio
- **dNBR**: Differenced NBR (pre vs post-fire)
- **BAI**: Burn Area Index
- Cloud and haze masking

### C) ML Refinement (Optional)
- U-Net for binary segmentation
- XGBoost for pixel classification
- Feature engineering with spectral indices

## 📁 Clean Project Structure

```
forestfire/
├── src/
│   ├── api/
│   │   └── simple_main.py      # FastAPI backend (working)
│   ├── detection/
│   │   ├── fire_detector.py    # Core detection algorithms
│   │   └── spectral_indices.py # Spectral index calculations
│   └── utils/
│       └── config_loader.py    # Advanced configuration system
├── web/
│   └── index.html              # Modern web interface
├── config/
│   └── config.example.yaml     # Comprehensive configuration
├── tests/
│   └── test_spectral_indices.py # Core functionality tests
├── start_simple.py             # Simple startup script
├── requirements-minimal.txt    # Minimal dependencies
├── requirements.txt            # Full dependencies
└── QUICKSTART.md              # Quick start guide
```

## 🌐 API Endpoints

- **Health Check**: `GET /health`
- **API Info**: `GET /`
- **Fire Detection**: `POST /api/v1/detect`
- **Status Check**: `GET /api/v1/status/{request_id}`
- **Results**: `GET /api/v1/results/{request_id}`
- **Hotspots**: `GET /api/v1/hotspots`
- **Regions**: `GET /api/v1/regions`
- **API Documentation**: `GET /docs` (Swagger UI)

## 🎯 Key Features

### ✅ Currently Working
- **Real-time API**: FastAPI backend with async processing
- **Interactive Web Interface**: Modern dashboard with Leaflet maps
- **Mock Detection System**: Realistic fire detection simulation
- **Progress Tracking**: Real-time detection progress updates
- **Multiple Regions**: California, Australia, and Global monitoring
- **Satellite Selection**: Sentinel-2 and Landsat support

### 🚧 Ready for Enhancement
- **Machine Learning Models**: U-Net, Transformers, XGBoost
- **Real Satellite Data**: NASA FIRMS, Copernicus, Google Earth Engine
- **Advanced Processing**: Cloud masking, atmospheric correction
- **Production Features**: Database, caching, monitoring

## 🔧 Configuration

Copy and customize the configuration:
```bash
cp config/config.example.yaml config/config.yaml
# Edit config.yaml with your settings
```

## 📊 Evaluation Metrics

- **Pixel-level**: IoU, F1-score, Precision, Recall
- **Event-level**: Time-to-first-alert, False alarm rate
- **System-level**: Latency, throughput, accuracy

## 🚀 Deployment

### Development
```bash
python start_simple.py
```

### Production
```bash
# Use gunicorn for production
gunicorn src.api.simple_main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## 🔮 Future Extensions

- Risk nowcasting with fuel and weather data
- On-device inference for edge deployment
- Temporal analysis for fire progression tracking
- Integration with emergency response systems

## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For questions or issues, please open a GitHub issue or contact the development team.

---

**🎉 The system is now clean, functional, and ready for advanced development!**
