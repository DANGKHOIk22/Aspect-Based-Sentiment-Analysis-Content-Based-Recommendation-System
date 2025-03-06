# Use the official Airflow image as the base
FROM apache/airflow:2.10.4

# Switch to root user to install dependencies
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to the airflow user
USER airflow

# Copy the requirements.txt file into the container
COPY requirements.txt /requirements.txt

# Install the Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt