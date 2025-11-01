# Spotify Analytics Pipeline - Dockerized
# Base: Apache Spark 3.5.3 with Python 3.11 + Delta Lake

FROM apache/spark:3.5.3-python3

# Switch to root for installations
USER root

# Set working directory
WORKDIR /app

# Install system dependencies, update CA certificates, and create python symlink
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    vim \
    ca-certificates \
    && apt-get upgrade -y ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Download Delta Lake JARs for Spark 3.5
RUN wget -q https://repo1.maven.org/maven2/io/delta/delta-spark_2.12/3.2.1/delta-spark_2.12-3.2.1.jar -P /opt/spark/jars/ && \
    wget -q https://repo1.maven.org/maven2/io/delta/delta-storage/3.2.1/delta-storage-3.2.1.jar -P /opt/spark/jars/

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (including delta-spark for Python bindings)
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir delta-spark==3.2.1

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p \
    data/bronze \
    data/kaggle \
    data/logs \
    data/.tokens \
    && chmod -R 777 data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

# Expose port for OAuth callback
EXPOSE 8888

# Set the entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "run_ingestion.py"]
