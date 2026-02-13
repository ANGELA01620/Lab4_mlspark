# %% [markdown]
# # Notebook 01: Ingesta de Datos
#
# **Objetivo**: Descargar y cargar datos filtrados de SECOP II
#
# **Filtro aplicado**:
# - Departamento: Distrito Capital de Bogotá
# - Fecha de firma: 2025-01-01 a 2025-03-31
# - Límite: 150,000 registros
#
# Fuente:
# https://www.datos.gov.co/Gastos-Gubernamentales/SECOP-II-Contratos-Electr-nicos/jbjy-vk9h

# %%
# Importar librerías
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, year, month
from sodapy import Socrata
import json
import os

# %%
# Configurar SparkSession (ACTUALIZADO)
spark = SparkSession.builder \
    .appName("SECOP_Ingesta") \
    .master("local[*]") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

print(f"Spark Version: {spark.version}")
print(f"Spark Master: {spark.sparkContext.master}")

# %%
# Descargar datos desde API Socrata con filtro
print("Descargando datos filtrados desde API Socrata...")

client = Socrata("www.datos.gov.co", None)

results = client.get(
    "jbjy-vk9h",
    query="""
        SELECT *
        WHERE 
            departamento = "Distrito Capital de Bogotá"
        AND
            fecha_de_firma >= '2025-01-01T00:00:00'
        AND
            fecha_de_firma < '2025-04-01T00:00:00'
        LIMIT 150000
    """
)

print(f"Registros descargados: {len(results)}")

# %%
# Guardar JSON localmente
json_path = "/opt/spark-data/raw/secop_bogota_2025_q1.json"
os.makedirs(os.path.dirname(json_path), exist_ok=True)

with open(json_path, "w", encoding="utf-8") as f:
    for record in results:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Datos guardados en: {json_path}")

# %%
# Leer JSON con Spark
print("Leyendo datos con Spark...")
df_raw = spark.read.json(json_path)

print(f"Total de registros: {df_raw.count():,}")
print(f"Total de columnas: {len(df_raw.columns)}")

# %%
# Explorar esquema
print("\n=== ESQUEMA DEL DATASET ===")
df_raw.printSchema()

# %%
# Mostrar primeras filas
print("\n=== PRIMERAS 5 FILAS ===")
df_raw.show(5, truncate=False)

# %%
# Columnas clave
columnas_clave = [
    "referencia_del_contrato",
    "nit_entidad",
    "nombre_entidad",
    "departamento",
    "ciudad",
    "tipo_de_contrato",
    "valor_del_contrato",
    "fecha_de_firma",
    "plazo",
    "plazo_de_ejec_del_contrato",
    "nombre_del_proveedor",
    "estado_contrato"
]

columnas_disponibles = [c for c in columnas_clave if c in df_raw.columns]

print(f"\nColumnas seleccionadas ({len(columnas_disponibles)}):")
for c in columnas_disponibles:
    print(f"- {c}")

# %%
# Seleccionar columnas disponibles
if columnas_disponibles:
    df_clean = df_raw.select(*columnas_disponibles)
else:
    print("No se encontraron columnas esperadas. Usando todas.")
    df_clean = df_raw

# %%
# Guardar en Parquet
output_path = "/opt/spark-data/raw/secop_bogota_2025_q1.parquet"

df_clean.write \
    .mode("overwrite") \
    .parquet(output_path)

print("Datos guardados en formato Parquet")

# %%
# Verificación
df_verificacion = spark.read.parquet(output_path)

print("\n=== VERIFICACIÓN ===")
print(f"Registros en Parquet: {df_verificacion.count():,}")
print(f"Columnas en Parquet: {len(df_verificacion.columns)}")

# %%
# Resumen final
print("\n" + "="*60)
print("RESUMEN DE INGESTA")
print("="*60)
print("✓ Fuente: API Socrata")
print("✓ Filtro: Bogotá 2025 Q1")
print(f"✓ Registros procesados: {df_clean.count():,}")
print("✓ Formato salida: Parquet")
print(f"✓ Ubicación: {output_path}")
print("="*60)

# %%
# Cerrar conexión
client.close()
spark.stop()
print("Proceso finalizado correctamente")
