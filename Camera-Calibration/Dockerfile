# Use official Python 3.11 image
FROM python:3.11

# Set working directory inside container
WORKDIR /app

# Install system libraries needed by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Run your script
CMD ["python", "calibrate.py"]