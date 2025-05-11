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
