pipeline {
    agent {
        docker {
            image 'python:3.9-slim'
            args '-v /var/run/docker.sock:/var/run/docker.sock'
        }
    }
    
    environment {
        DOCKER_HUB_CREDS = credentials('docker-hub-credentials')
        IMAGE_NAME = 'iris-classifier'
        IMAGE_TAG = "${env.BUILD_NUMBER}"
        DOCKER_HUB_USERNAME = "${env.DOCKER_HUB_CREDS_USR}"
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Setup') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }
        
        stage('Prepare Data & Train Model') {
            steps {
                sh 'python src/data/process_data.py'
                sh 'python src/models/train_model.py'
            }
        }
        
        stage('Test') {
            steps {
                sh 'python -m pytest tests/'
            }
        }
        
        stage('Build Docker Image') {
            steps {
                sh 'docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .'
                sh 'docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${DOCKER_HUB_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}'
                sh 'docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${DOCKER_HUB_USERNAME}/${IMAGE_NAME}:latest'
            }
        }
        
        stage('Push to Docker Hub') {
            steps {
                sh 'echo $DOCKER_HUB_CREDS_PSW | docker login -u $DOCKER_HUB_USERNAME --password-stdin'
                sh 'docker push ${DOCKER_HUB_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}'
                sh 'docker push ${DOCKER_HUB_USERNAME}/${IMAGE_NAME}:latest'
            }
        }
        
        stage('Deploy to Kubernetes') {
            steps {
                sh "sed -i 's|image:.*|image: ${DOCKER_HUB_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}|' kubernetes/deployment.yaml"
                sh 'kubectl apply -f kubernetes/deployment.yaml'
                sh 'kubectl apply -f kubernetes/service.yaml'
            }
        }
    }
    
    post {
        always {
            sh 'docker logout'
            cleanWs()
        }
        success {
            echo 'Pipeline executed successfully!'
        }
        failure {
            echo 'Pipeline execution failed!'
        }
    }
}
