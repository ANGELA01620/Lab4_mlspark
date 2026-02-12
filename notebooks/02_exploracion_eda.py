# %% [markdown]
# # Notebook 02: Análisis Exploratorio de Datos (EDA)
#
# Objetivo:
# - Entender la distribución de variables
# - Identificar valores nulos
# - Detectar outliers
# - Analizar contratos por ENTIDAD

# ==========================================================
# 1. IMPORTAR LIBRERÍAS
# ==========================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, sum as spark_sum, avg,
    min as spark_min, max as spark_max,
    stddev, isnan, when, isnull, desc,
    to_date, year, month
)

from pyspark.sql.types import DoubleType, FloatType, StructType
import os

# ==========================================================
# 2. CREAR SPARK SESSION
# ==========================================================

spark = SparkSession.builder \
    .appName("SECOP_EDA") \
    .master("local[*]") \
    .getOrCreate()

print("Spark Version:", spark.version)

# ==========================================================
# 3. CARGAR DATOS DESDE BRONZE
# ==========================================================

parquet_path = "/opt/spark-data/bronze/secop_contratos"

print(f"Cargando datos desde: {parquet_path}")

df = spark.read.parquet(parquet_path)

total_registros = df.count()

print(f"Registros cargados: {total_registros:,}")
print(f"Columnas: {len(df.columns)}")

df.printSchema()

# ==========================================================
# 4. ESTADÍSTICAS GENERALES
# ==========================================================

print("\n=== ESTADÍSTICAS DESCRIPTIVAS GENERALES ===")
df.describe().show()

# ==========================================================
# 5. ANÁLISIS DE VALORES NULOS
# ==========================================================

print("\n=== ANÁLISIS DE VALORES NULOS ===")

null_expressions = []

for field in df.schema.fields:
    
    column_name = field.name
    data_type = field.dataType

    if isinstance(data_type, (DoubleType, FloatType)):
        expr = spark_sum(
            when(col(column_name).isNull() | isnan(col(column_name)), 1)
            .otherwise(0)
        ).alias(column_name)
    else:
        expr = spark_sum(
            when(col(column_name).isNull(), 1)
            .otherwise(0)
        ).alias(column_name)

    null_expressions.append(expr)

null_counts = df.select(null_expressions)
null_counts.show(vertical=True)

# ==========================================================
# 6. IDENTIFICAR COLUMNA DE VALOR
# ==========================================================

valor_cols = [c for c in df.columns if 'valor' in c.lower()]

if not valor_cols:
    raise ValueError("No se encontró columna de valor en el dataset.")

valor_col = valor_cols[0]
print(f"\nColumna de valor detectada: {valor_col}")

df = df.withColumn(valor_col + "_num", col(valor_col).cast("double"))

# ==========================================================
# 7. ESTADÍSTICAS DE VALOR DEL CONTRATO
# ==========================================================

print(f"\n=== ESTADÍSTICAS DE {valor_col} ===")

df.select(
    spark_min(col(valor_col + "_num")).alias("Min"),
    spark_max(col(valor_col + "_num")).alias("Max"),
    avg(col(valor_col + "_num")).alias("Promedio"),
    stddev(col(valor_col + "_num")).alias("Desv_Std")
).show()

# ==========================================================
# 8. DETECCIÓN DE OUTLIERS (IQR)
# ==========================================================

print("\n=== DETECCIÓN DE OUTLIERS (IQR) ===")

percentiles = df.approxQuantile(
    valor_col + "_num",
    [0.25, 0.50, 0.75, 0.95, 0.99],
    0.01
)

q1, q2, q3, p95, p99 = percentiles

print(f"Q1 (25%): {q1:,.2f}")
print(f"Mediana (50%): {q2:,.2f}")
print(f"Q3 (75%): {q3:,.2f}")
print(f"P95: {p95:,.2f}")
print(f"P99: {p99:,.2f}")

iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

num_outliers = df.filter(
    (col(valor_col + "_num") < lower_bound) |
    (col(valor_col + "_num") > upper_bound)
).count()

print(f"Outliers detectados: {num_outliers:,} ({(num_outliers/total_registros)*100:.2f}%)")

# ==========================================================
# 9. DISTRIBUCIÓN POR ENTIDAD (CORREGIDO)
# ==========================================================

entidad_cols = [c for c in df.columns if 'entidad' in c.lower()]

if entidad_cols:
    entidad_col = entidad_cols[0]
    print(f"\nColumna de entidad detectada: {entidad_col}")

    print("\n=== TOP 10 ENTIDADES POR NÚMERO DE CONTRATOS ===")

    df.groupBy(entidad_col) \
        .agg(count("*").alias("num_contratos")) \
        .orderBy(desc("num_contratos")) \
        .show(10, truncate=False)

    print("\n=== TOP 10 ENTIDADES POR VALOR TOTAL CONTRATADO ===")

    df.groupBy(entidad_col) \
        .agg(
            count("*").alias("num_contratos"),
            spark_sum(valor_col + "_num").alias("valor_total"),
            avg(valor_col + "_num").alias("valor_promedio")
        ) \
        .orderBy(desc("valor_total")) \
        .show(10, truncate=False)

else:
    print("No se encontró columna de entidad en el dataset.")

# ==========================================================
# 10. DISTRIBUCIÓN POR TIPO DE CONTRATO
# ==========================================================

tipo_cols = [c for c in df.columns if 'tipo' in c.lower() and 'contrato' in c.lower()]

if tipo_cols:
    tipo_col = tipo_cols[0]
    print(f"\n=== DISTRIBUCIÓN POR TIPO DE CONTRATO ===")

    df.groupBy(tipo_col) \
        .agg(count("*").alias("num_contratos")) \
        .orderBy(desc("num_contratos")) \
        .show(10, truncate=False)

# ==========================================================
# 11. ANÁLISIS TEMPORAL
# ==========================================================

fecha_cols = [c for c in df.columns if 'fecha' in c.lower()]

if fecha_cols:
    fecha_col = fecha_cols[0]
    print(f"\nColumna de fecha detectada: {fecha_col}")

    df = df.withColumn("fecha_parsed", to_date(col(fecha_col)))
    df = df.withColumn("anio", year(col("fecha_parsed")))
    df = df.withColumn("mes", month(col("fecha_parsed")))

    print("\n=== CONTRATOS POR AÑO ===")
    df.groupBy("anio") \
        .agg(count("*").alias("num_contratos")) \
        .orderBy("anio") \
        .show()

# ==========================================================
# 12. GUARDAR DATASET PROCESADO
# ==========================================================

processed_path = "/opt/spark-data/processed/secop_eda"

os.makedirs("/opt/spark-data/processed", exist_ok=True)

df.write.mode("overwrite").parquet(processed_path)

print(f"\nDataset EDA guardado en: {processed_path}")

# ==========================================================
# 13. CERRAR SESIÓN
# ==========================================================

spark.stop()
print("SparkSession finalizada correctamente")
