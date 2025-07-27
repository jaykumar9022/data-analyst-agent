# Dockerfile

# Stage 1: Build Stage (for installing dependencies)
# Use a slim Python base image. This keeps the final image size smaller.
FROM python:3.9-slim-buster

# Set the working directory inside the container.
WORKDIR /app

# Copy the requirements.txt file into the container's /app directory.
COPY requirements.txt .

# Install Python dependencies.
# --no-cache-dir: Prevents pip from storing cached wheels, reducing image size.
# -r requirements.txt: Installs all packages listed in requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Application Stage (copying actual code)
# Copy the rest of your application code from your local machine's current directory (.)
# into the container's working directory (/app).
COPY . .

# Expose port 80. This tells Docker that the container will listen on port 80.
EXPOSE 80

# Command to run your FastAPI application when the container starts.
# uvicorn: The ASGI server to run your FastAPI app.
# main:app: Refers to the 'app' object inside the 'main.py' file.
# --host 0.0.0.0: Makes the application accessible from outside the container.
# --port 80: Tells uvicorn to listen on port 80 inside the container.
CMD uvicorn main:app --host 0.0.0.0 --port $PORT