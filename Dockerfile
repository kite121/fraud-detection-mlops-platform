# Python image
FROM python:3.11-slim

# Select working directory
WORKDIR /app

# Copy requirements to avoid reinstallations in cale of changes not in requirements.txt
COPY requirements.txt .

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole project to the working directory
COPY . .

# Open FastAPI port
EXPOSE 8000

# Run Fast API. The main is the file where FastAPI client is created
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]