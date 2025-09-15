# MLOps Project

## Setup

### 1. Start MLflow server
```bash
mlflow ui --port 8080 --host 127.0.0.1
```

### 2. Train and register model (new terminal)
```bash
python3 level1.py
```

### 3. Start API service (new terminal)
```bash
docker-compose up --build
```

### 4. Test API endpoints (new terminal)
```bash
python3 test_api.py
```

## Files

- `level1.py` - Model training with MLflow tracking
- `level2.py` - FastAPI service
- `test_api.py` - API testing script
- `docker-compose.yml` - Container orchestration
- `Dockerfile` - Service containerization
