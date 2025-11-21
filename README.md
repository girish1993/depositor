# Depositor

A lightweight e2e ML project that trains a model to predict whether a bank customer will subscribe to a term deposit. This project includes a complete training pipeline, FastAPI inference service, and Dockerized setup.


# Prerequisites
Make sure you have these installed:
- Python 3.11+
- pip
- make
- Docker and Docker Compose

If `make` or `Docker` aren’t installed, follow your OS-specific instructions:

```
sudo apt install make docker.io docker-compose -y
```

# How to run?

### Steps to run locally

Make use of the make commands to run the application. With the following commands

```
# Install dependencies
make install

# Train model and save artifacts
make train

# Run FastAPI server
make api
# → http://localhost:8000
```

The api has a GET `/health` and a POST `/predict` endpoints.


### Steps to run with Docker

1. Build and run trainer

`make build-trainer`

2. Build and run Api(serving/inference)

`make build-api`

3. Build and run both services

`make build-both`

4. Tear down all container

`make down`

That's it. Use the `/predict` end point to run predictions. Example payload schema

```
{
  "customers": [
    {
      "age": 35,
      "job": "blue-collar",
      "marital": "married",
      "education": "secondary",
      "default": "no",
      "balance": 1200.50,
      "housing": "yes",
      "loan": "no",
      "contact": "cellular",
      "day": 5,
      "month": "may",
      "duration": 180,
      "campaign": 2,
      "pdays": 999,
      "previous": 0,
      "poutcome": "unknown"
    }
  ]
}

```

You can use the cURL command 

```
curl -X POST "http://127.0.0.1:8000/predict?threshold=0.5" \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      {
        "age": 35,
        "job": "blue-collar",
        "marital": "married",
        "education": "secondary",
        "default": "no",
        "balance": 1200.50,
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "day": 5,
        "month": "may",
        "duration": 180,
        "campaign": 2,
        "pdays": 999,
        "previous": 0,
        "poutcome": "unknown"
      }
    ]
  }'

```

Or simply use Postman or other API testing clients.


