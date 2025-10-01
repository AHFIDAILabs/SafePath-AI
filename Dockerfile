# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# The files are now copied from their location *relative to the project root*
# Requirements is inside the backend/ folder:
COPY ./backend/requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy all application directories:
COPY ./backend/app /app/app
COPY ./backend/artifacts /app/artifacts
COPY ./frontend/templates /app/templates
COPY ./frontend/static /app/static
COPY ./src /app/src  

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable for Python path
ENV PYTHONPATH=/app

# Run main.py when the container launches
# Uvicorn entry point is still 'app.main:app' within the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]