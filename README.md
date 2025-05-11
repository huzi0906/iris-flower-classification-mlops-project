# MLOps Project: Iris Flower Classification System

## Project Overview

This is an end-to-end MLOps project for Innovate Analytics Inc., implementing a scalable machine learning system for Iris flower classification. The project demonstrates a complete MLOps workflow including data engineering, model development, and deployment.

## Dataset

The project uses the Iris dataset, a classic and well-known dataset in machine learning. It contains 150 samples of iris flowers from three different species (Setosa, Versicolor, and Virginica) with four features measured for each sample: sepal length, sepal width, petal length, and petal width.

The Iris dataset is:

- Small and manageable (~150 records)
- Easy to train models on (clear class separation)
- Readily available (part of scikit-learn)
- Perfect for demonstrating ML workflow concepts

## Project Structure

```
mlops-project/
├── data/                      # Data directory
│   ├── raw/                   # Raw data
│   └── processed/             # Processed data ready for modeling
├── models/                    # Saved model files
├── src/                       # Source code
│   ├── data/                  # Data processing scripts
│   ├── models/                # Model training scripts
│   ├── visualization/         # Visualization scripts
│   └── pipeline/              # Pipeline orchestration
├── notebooks/                 # Jupyter notebooks for exploration
├── kubernetes/                # Kubernetes deployment files
├── airflow/                   # Airflow DAGs for workflow automation
├── jenkins/                   # Jenkins pipeline configuration
├── docs/                      # Project documentation
├── Dockerfile                 # Docker configuration
├── .github/                   # GitHub Actions workflows
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## MLOps Components

This project implements a complete MLOps workflow with the following components:

1. **Project Management**

   - GitHub Issues for sprint planning and task tracking
   - Milestones and user stories to organize work

2. **Environment Management**

   - Git branching strategy (dev, test, prod)
   - Environment-specific configurations

3. **Quality Assurance**

   - Automated linting and unit testing with GitHub Actions
   - Code reviews through Pull Requests

4. **Data Version Control**

   - DVC for tracking datasets and model versions

5. **Experiment Tracking**

   - MLflow for logging experiments, parameters, and metrics

6. **CI/CD Pipeline**

   - GitHub Actions for continuous integration
   - Jenkins for continuous deployment
   - Docker for containerization
   - Kubernetes (Minikube) for orchestration

7. **ETL Pipeline**
   - Airflow for data processing workflow automation

## Getting Started

### Prerequisites

- Python 3.8+
- Docker and Docker Compose
- Minikube
- kubectl
- Git

### Installation

1. Clone the repository

```bash
git clone https://github.com/yourusername/mlops-project.git
cd mlops-project
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Set up DVC

```bash
dvc init
dvc add data/raw/iris.csv
```

4. Run the data processing pipeline

```bash
python src/data/process_data.py
```

5. Train the model

```bash
python src/models/train_model.py
```

6. Build and run Docker container

```bash
docker build -t iris-classifier .
docker run -p 8000:8000 iris-classifier
```

7. Deploy to Kubernetes

```bash
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
```

## Development Workflow

1. Create a feature branch from dev

```bash
git checkout -b feature/your-feature-name
```

2. Implement your changes and commit

```bash
git add .
git commit -m "Add your feature"
```

3. Push your branch and create a Pull Request to dev

```bash
git push origin feature/your-feature-name
```

4. After review and approval, merge to dev
5. Run integration tests on test branch
6. Deploy to production via Jenkins pipeline

## License

This project is licensed under the MIT License - see the LICENSE file for details.
