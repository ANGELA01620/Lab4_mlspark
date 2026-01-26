# Proyecto Final MLSpark - Imagen Base
FROM python:3.11-slim

# Instalar Java (Requerido por Spark)
RUN apt-get update && \
    apt-get install -y default-jdk procps curl wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Variables de Entorno
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV SPARK_VERSION=3.5.0
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
ENV PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9.7-src.zip:$PYTHONPATH

# Descargar e instalar Apache Spark
RUN curl -O https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.tgz && \
    tar xvf spark-${SPARK_VERSION}-bin-hadoop3.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop3 /opt/spark && \
    rm spark-${SPARK_VERSION}-bin-hadoop3.tgz

# Copiar requirements.txt e instalar dependencias
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Crear directorios de trabajo
RUN mkdir -p /opt/spark-data /opt/spark-notebooks /opt/spark-apps /opt/mlflow/mlruns

WORKDIR /opt/spark-notebooks

# Comando por defecto (Jupyter Lab)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token=''"]
