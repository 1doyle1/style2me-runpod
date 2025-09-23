FROM python:3.11-slim

WORKDIR /app

# Optional: installs git so the build log warning disappears
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY handler.py .

# RunPod serverless expects to execute handler.py
CMD ["python", "-u", "handler.py"]
