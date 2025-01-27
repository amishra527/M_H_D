# Use the official Python image as a base image
FROM python:3.10-slim

# Set a working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app/

# Expose the port FastAPI will run on
EXPOSE 8000

# Run the application with Uvicorn
CMD ["uvicorn", "main_12_asset:app", "--host", "0.0.0.0", "--port", "8000"]