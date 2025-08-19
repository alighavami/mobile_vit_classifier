
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace

# Install dependencies first to leverage cache
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Default entrypoint
CMD ["python", "train.py"]
