.PHONY: setup up down logs train serve test build push deploy-staging deploy-prod example-iris clean help

# Default target
help:
	@echo "ML Pipeline Bootstrap - Available Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make setup           Run interactive setup wizard"
	@echo ""
	@echo "Infrastructure:"
	@echo "  make up              Start MLflow + MinIO + Postgres + Monitoring"
	@echo "  make down            Stop all infrastructure"
	@echo "  make logs            View infrastructure logs"
	@echo ""
	@echo "Development:"
	@echo "  make train           Train model (reads config.yaml)"
	@echo "  make serve           Run API locally"
	@echo "  make test            Run all tests"
	@echo ""
	@echo "Docker:"
	@echo "  make build           Build serving container"
	@echo "  make push            Push to container registry"
	@echo ""
	@echo "Kubernetes:"
	@echo "  make deploy-staging  Deploy to staging"
	@echo "  make deploy-prod     Deploy to production"
	@echo ""
	@echo "Example:"
	@echo "  make example-iris    Run full Iris example"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean           Remove generated files"

# Setup
setup:
	python setup.py

# Infrastructure
up:
	docker-compose up -d
	@echo ""
	@echo "Services started:"
	@echo "  MLflow:     http://localhost:5000"
	@echo "  MinIO:      http://localhost:9001 (admin/minioadmin)"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Grafana:    http://localhost:3000 (admin/admin)"

down:
	docker-compose down

logs:
	docker-compose logs -f

# Development
train:
	python -m src.training.train

serve:
	uvicorn src.serving.main:app --reload --port 8000

test:
	pytest tests/ -v

# Docker
IMAGE_NAME ?= ml-pipeline
IMAGE_TAG ?= latest
REGISTRY ?= ghcr.io/vineethkumar7

build:
	docker build -f docker/Dockerfile.serve -t $(IMAGE_NAME):$(IMAGE_TAG) .

push: build
	docker tag $(IMAGE_NAME):$(IMAGE_TAG) $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)
	docker push $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)

# Kubernetes
deploy-staging:
	kubectl apply -k k8s/overlays/staging

deploy-prod:
	@echo "⚠️  Deploying to production requires approval"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ]
	kubectl apply -k k8s/overlays/production

# Example
example-iris:
	@echo "Setting up Iris example..."
	cp examples/iris/config.yaml config.yaml
	@echo ""
	@echo "Starting infrastructure..."
	$(MAKE) up
	@echo ""
	@echo "Waiting for services to be ready..."
	sleep 10
	@echo ""
	@echo "Training model..."
	$(MAKE) train
	@echo ""
	@echo "Starting API server..."
	@echo "API will be available at http://localhost:8000"
	@echo "Try: curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{\"features\": [5.1, 3.5, 1.4, 0.2]}'"
	$(MAKE) serve

# Cleanup
clean:
	rm -rf models/
	rm -rf mlruns/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
