# MLOps Project Documentation

## Introduction

This documentation provides comprehensive information about the MLOps project for Innovate Analytics Inc. This project demonstrates a complete end-to-end MLOps workflow using the Iris dataset, implementing industry best practices for machine learning operations.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Setup](#project-setup)
3. [Development Process](#development-process)
4. [CI/CD Pipeline](#cicd-pipeline)
5. [Deployment](#deployment)
6. [Monitoring](#monitoring)
7. [MLOps Best Practices](#mlops-best-practices)

## Project Overview

The project uses the classic Iris dataset to demonstrate a complete MLOps pipeline. The Iris dataset contains measurements of three different species of iris flowers (Setosa, Versicolor, and Virginica) and is used to train a classification model.

### Key Features

- End-to-end MLOps pipeline from data processing to production deployment
- Automated data processing and model training
- Version control for data and models using DVC
- Experiment tracking with MLflow
- CI/CD pipeline using GitHub Actions and Jenkins
- Containerization with Docker
- Kubernetes deployment for scalability
- Automated testing and quality assurance

## Project Setup

### Prerequisites

- Python 3.8+
- Git
- Docker and Docker Compose
- Minikube (for local Kubernetes deployment)
- kubectl
- Jenkins (for CD pipeline)
- Airflow (for workflow automation)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/mlops-project.git
cd mlops-project
```

2. **Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Initialize DVC**

```bash
dvc init
```

5. **Start MLflow server**

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```

6. **Setup Airflow**

```bash
export AIRFLOW_HOME=$(pwd)/airflow
airflow db init
airflow scheduler -D
airflow webserver -D -p 8080
```

## Development Process

### Branching Strategy

The project follows a standardized Git branching strategy:

- `main` - Production code
- `test` - Integration testing code
- `dev` - Development code
- Feature branches - Individual features created from and merged into `dev`

### Workflow

1. **Create a feature branch**

   ```bash
   git checkout dev
   git pull
   git checkout -b feature/your-feature-name
   ```

2. **Implement your changes**

   - Write code
   - Add tests
   - Run local tests

3. **Commit changes**

   ```bash
   git add .
   git commit -m "Add your feature description"
   ```

4. **Push and create PR**

   ```bash
   git push origin feature/your-feature-name
   # Create PR to dev branch using GitHub interface
   ```

5. **Code review and merge**

   - Another team member reviews the code
   - Automated tests run via GitHub Actions
   - Merge to `dev` after approval

6. **Testing and production**
   - Code from `dev` is merged to `test` for integration testing
   - After passing all tests, code from `test` is merged to `main`
   - Jenkins pipeline deploys to production

## CI/CD Pipeline

### Continuous Integration (GitHub Actions)

The CI pipeline runs automatically when:

- Code is pushed to the `dev` branch
- A pull request is created targeting `dev` or `test` branches

CI pipeline stages:

1. **Linting** - Check code style using flake8, black, and isort
2. **Unit Testing** - Run unit tests with pytest
3. **Build** - Process data and train model
4. **Artifact Creation** - Build Docker image

### Continuous Deployment (Jenkins)

The CD pipeline runs when:

- Changes are merged to the `main` branch

CD pipeline stages:

1. **Build** - Build the Docker image with latest code
2. **Test** - Run tests on the built image
3. **Push** - Push the image to Docker Hub
4. **Deploy** - Deploy to Kubernetes cluster

## Deployment

### Docker

The application is containerized using Docker:

```Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files for the API service
COPY models/ /app/models/
COPY src/models/predict_api.py /app/src/models/

# Set environment variables
ENV MODEL_PATH=/app/models/iris_classifier.joblib
ENV PYTHONUNBUFFERED=1

# Expose the port the application will run on
EXPOSE 8000

# Run the API server
CMD ["uvicorn", "src.models.predict_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Deployment

Deployment configuration (`kubernetes/deployment.yaml`):

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: iris-classifier
  labels:
    app: iris-classifier
spec:
  replicas: 3
  selector:
    matchLabels:
      app: iris-classifier
  template:
    metadata:
      labels:
        app: iris-classifier
    spec:
      containers:
        - name: iris-classifier
          image: ${DOCKER_HUB_USERNAME}/iris-classifier:latest
          ports:
            - containerPort: 8000
          resources:
            limits:
              cpu: "0.5"
              memory: "512Mi"
            requests:
              cpu: "0.2"
              memory: "256Mi"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
```

Service configuration (`kubernetes/service.yaml`):

```yaml
apiVersion: v1
kind: Service
metadata:
  name: iris-classifier-service
spec:
  selector:
    app: iris-classifier
  ports:
    - port: 80
      targetPort: 8000
      protocol: TCP
  type: NodePort
```

## Monitoring

The project includes several monitoring mechanisms:

1. **API Health Checks** - The API exposes a `/health` endpoint for monitoring
2. **Kubernetes Probes** - Liveness and readiness probes monitor container health
3. **MLflow Metrics Tracking** - Model performance metrics are tracked over time
4. **Jenkins Build Reports** - CI/CD pipeline execution reports

## MLOps Best Practices

This project implements several MLOps best practices:

1. **Version Control**

   - Code is versioned using Git
   - Data and models are versioned using DVC

2. **Continuous Integration/Continuous Deployment**

   - Automated testing with GitHub Actions
   - Automated deployment with Jenkins

3. **Infrastructure as Code**

   - Kubernetes deployments defined as YAML
   - CI/CD pipelines defined as code

4. **Experiment Tracking**

   - ML experiments tracked using MLflow
   - Parameters, metrics, and artifacts are logged

5. **Containerization**

   - Application packaged as Docker container
   - Consistent environments across development and production

6. **Workflow Automation**

   - Data processing and model training automated with Airflow

7. **Reproducibility**
   - Fixed random seeds
   - Environment dependencies pinned to specific versions
   - Data versioning
