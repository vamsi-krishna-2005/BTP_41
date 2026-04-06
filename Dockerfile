# Use a standard Python image
FROM python:3.9-slim

# Install system dependencies (added wget!)
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 git wget && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download SAM into the container so the end-user doesn't have to
RUN wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Copy your Swin-Unet weights and API code
COPY . .

# Expose the port
EXPOSE 8000

# Start the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]