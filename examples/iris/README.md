# Iris Classifier Example

This example demonstrates the complete ML pipeline using the classic Iris dataset.

## Overview

- **Model**: Random Forest Classifier
- **Framework**: scikit-learn
- **Dataset**: Iris (150 samples, 4 features, 3 classes)
- **Features**: sepal_length, sepal_width, petal_length, petal_width
- **Classes**: setosa, versicolor, virginica

## Quick Start

```bash
# From project root
make example-iris
```

Or manually:

```bash
# 1. Update config to use iris example
# Edit config.yaml:
#   model.module: examples.iris.model
#   model.name: iris-classifier

# 2. Start infrastructure
make up

# 3. Train
python -m src.training.train

# 4. Serve
make serve

# 5. Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

## Expected Output

```json
{
  "prediction": 0,
  "class_name": "setosa",
  "probability": 0.97,
  "probabilities": {
    "setosa": 0.97,
    "versicolor": 0.02,
    "virginica": 0.01
  },
  "model_version": "Production",
  "latency_ms": 5.2
}
```

## Files

| File | Description |
|------|-------------|
| `model.py` | ModelWrapper implementation |
| `data.py` | Dataset loader |
| `README.md` | This file |

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | 100 | Number of trees |
| `max_depth` | 5 | Maximum tree depth |
| `random_state` | 42 | Random seed |

Override via config.yaml:

```yaml
training:
  params:
    n_estimators: 200
    max_depth: 10
```

Or command line:

```bash
python -m src.training.train --params '{"n_estimators": 200}'
```
