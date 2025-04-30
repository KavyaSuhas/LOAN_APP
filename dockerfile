# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy everything into the container
COPY . .

# Install build tools for scikit-surprise and other packages that need compilation
RUN apt-get update && apt-get install -y build-essential gcc

# Install dependencies
RUN pip ins tall --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask's port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]