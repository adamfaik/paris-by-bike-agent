# Use a slim Python image
FROM python:3.11-slim

# System deps (optional but handy for scientific libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

# Create app dir
WORKDIR /app

# Copy and install deps first (better caching)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest
COPY . /app

# Expose the port Spaces expects
ENV PORT=7860
# Make Chainlit listen on all interfaces
ENV HOST=0.0.0.0

# Start Chainlit
# - no autoreload/watch (not needed on Spaces)
# - headless so it doesn't try to open a browser
CMD chainlit run app.py --host $HOST --port $PORT --headless