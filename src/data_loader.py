"""
Utilidades para descarga y carga de datos
"""

import requests
import json
from typing import Optional


def download_secop_data(
    limit: int = 100000,
    output_path: str = "/opt/spark-data/raw/secop_contratos.json"
) -> str:
    """
    Descarga datos de SECOP II desde la API de Datos Abiertos Colombia

    Args:
        limit: Número máximo de registros a descargar
        output_path: Ruta donde guardar el archivo JSON

    Returns:
        Ruta del archivo guardado
    """
    api_url = f"https://www.datos.gov.co/resource/jbjy-vk9h.json?$limit={limit}"

    print(f"Descargando {limit} registros desde API...")
    response = requests.get(api_url)
    response.raise_for_status()

    data = response.json()
    print(f"✓ {len(data)} registros descargados")

    # Guardar JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✓ Datos guardados en: {output_path}")
    return output_path


def create_spark_session(
    app_name: str,
    master: str = "spark://spark-master:7077",
    executor_memory: str = "2g"
):
    """
    Crea una SparkSession configurada

    Args:
        app_name: Nombre de la aplicación
        master: URL del Spark Master
        executor_memory: Memoria para executors

    Returns:
        SparkSession configurada
    """
    from pyspark.sql import SparkSession

    spark = SparkSession.builder \
        .appName(app_name) \
        .master(master) \
        .config("spark.executor.memory", executor_memory) \
        .getOrCreate()

    return spark
