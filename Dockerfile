# Base Image
FROM python:3.10-slim

# Working Directory
WORKDIR /app

# Install System Dependencies
# (build-essential potentially needed for some python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy Requirements
COPY requirements.txt .

# Install Python Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy Application Code
COPY . .

# Expose Streamlit Port
EXPOSE 8501

# Run Command
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
