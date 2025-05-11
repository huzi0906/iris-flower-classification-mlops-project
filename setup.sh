#!/bin/bash

# MLOps Project Initialization Script
# This script sets up the MLOps project environment

# Set colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}====================================${NC}"
echo -e "${GREEN}  MLOps Project Initialization     ${NC}"
echo -e "${GREEN}====================================${NC}"

# Create necessary directories if they don't exist
echo -e "\n${YELLOW}Creating project directories...${NC}"
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models
mkdir -p logs
echo -e "${GREEN}✓ Project directories created${NC}"

# Check Python version
echo -e "\n${YELLOW}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1)
if [[ $python_version == *"Python 3."* ]]; then
    echo -e "${GREEN}✓ $python_version detected${NC}"
else
    echo -e "${RED}❌ Python 3.8+ is required. Please install Python 3.8 or newer.${NC}"
    exit 1
fi

# Create virtual environment
echo -e "\n${YELLOW}Setting up virtual environment...${NC}"
python3 -m venv venv
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment created and activated${NC}"

# Install dependencies
echo -e "\n${YELLOW}Installing dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Initialize Git if not already initialized
echo -e "\n${YELLOW}Checking Git initialization...${NC}"
if [ -d .git ]; then
    echo -e "${GREEN}✓ Git repository already initialized${NC}"
else
    git init
    echo -e "${GREEN}✓ Git repository initialized${NC}"
    
    # Add .gitkeep files to empty directories
    touch data/raw/.gitkeep
    touch data/processed/.gitkeep
    touch models/.gitkeep
    
    # Initial commit
    git add .
    git commit -m "Initial project setup"
    echo -e "${GREEN}✓ Initial commit created${NC}"
fi

# Initialize DVC
echo -e "\n${YELLOW}Initializing DVC...${NC}"
dvc init
echo -e "${GREEN}✓ DVC initialized${NC}"

# Download Iris dataset
echo -e "\n${YELLOW}Downloading Iris dataset...${NC}"
python src/data/process_data.py
echo -e "${GREEN}✓ Iris dataset downloaded and processed${NC}"

# Train initial model
echo -e "\n${YELLOW}Training initial model...${NC}"
python src/models/train_model.py
echo -e "${GREEN}✓ Initial model trained${NC}"

# Set up MLflow
echo -e "\n${YELLOW}Setting up MLflow...${NC}"
mkdir -p mlruns
echo -e "${GREEN}✓ MLflow directory created${NC}"
echo -e "${YELLOW}You can start the MLflow server with:${NC}"
echo -e "  mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000"

# Final message
echo -e "\n${GREEN}====================================${NC}"
echo -e "${GREEN}  MLOps Project Setup Complete!    ${NC}"
echo -e "${GREEN}====================================${NC}"
echo -e "\n${YELLOW}Next steps:${NC}"
echo -e "1. Start developing using the created structure"
echo -e "2. Run 'git branch -b dev' to create and switch to the dev branch"
echo -e "3. Create feature branches for your work with 'git checkout -b feature/your-feature-name'"
echo -e "4. Refer to the documentation in the docs/ directory for more information"
echo -e "\n${YELLOW}To start the FastAPI service:${NC}"
echo -e "  uvicorn src.models.predict_api:app --reload --host 0.0.0.0 --port 8000"
echo -e "\n${GREEN}Happy coding!${NC}\n"
