# ML Model Deployment Pipeline

End-to-end MLOps pipeline for deploying ML models to production using containerized microservices.

## Features

- **Model Versioning** — MLflow experiment tracking and model registry
- **Artifact Storage** — S3-compatible storage (MinIO) for model artifacts
- **Model Serving** — FastAPI containers with health checks and metrics
- **Kubernetes Deployment** — k3s cluster with auto-scaling
- **Automated CI/CD** — GitHub Actions pipeline (test → build → deploy)

## Tech Stack

| Component | Technology |
|-----------|------------|
| ML Tracking | MLflow |
| Object Storage | MinIO |
| API Framework | FastAPI |
| Containerization | Docker |
| Orchestration | Kubernetes (k3s) |
| CI/CD | GitHub Actions |
| Monitoring | Prometheus + Grafana |

## Quick Start

```bash
# Start local development stack
docker-compose up -d

# Train a model
python src/training/train.py

# Run API locally
uvicorn src.serving.main:app --reload

# Run tests
pytest tests/
```

## Documentation

- [Planning Document](PLANNING.md) — Architecture, phases, and implementation details

## License

MIT
