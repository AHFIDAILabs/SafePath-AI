# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY ./requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app, artifacts, and src directories into the container
COPY ./app /app/app
COPY ./artifacts /app/artifacts
COPY ./src /app/src 

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable for Python path
ENV PYTHONPATH=/app

# Run main.py when the container launches
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]