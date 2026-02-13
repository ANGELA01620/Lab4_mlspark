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
    stddev, isnan, when, desc,
    to_date, year, month, regexp_replace
)

from pyspark.sql.types import DoubleType, FloatType
import os


# ==========================================================
# 2. CREAR SPARK SESSION
# ==========================================================

spark = SparkSession.builder \
    .appName("SECOP_EDA_CORREGIDO") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

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
# 4. IDENTIFICAR Y TIPAR CORRECTAMENTE VALOR DEL CONTRATO
# ==========================================================

valor_cols = [c for c in df.columns if 'valor' in c.lower()]

if not valor_cols:
    raise ValueError("No se encontró columna de valor en el dataset.")

valor_col = valor_cols[0]

print(f"\nColumna de valor detectada: {valor_col}")

# Limpieza preventiva (si existieran comas)
df = df.withColumn(
    valor_col,
    regexp_replace(col(valor_col), ",", "")
)

# Cast real a double (igual que tu bloque correcto)
df = df.withColumn(
    valor_col,
    col(valor_col).cast("double")
)

# Eliminar registros donde no se pudo convertir
df = df.filter(col(valor_col).isNotNull())

print("✓ Columna convertida correctamente a double")
df.select(valor_col).describe().show()


# ==========================================================
# 5. ESTADÍSTICAS DE VALOR DEL CONTRATO
# ==========================================================

print("\n=== ESTADÍSTICAS DE VALOR DEL CONTRATO ===")

df.select(
    spark_min(valor_col).alias("Min"),
    spark_max(valor_col).alias("Max"),
    avg(valor_col).alias("Promedio"),
    stddev(valor_col).alias("Desv_Std")
).show(truncate=False)


# ==========================================================
# 6. DETECCIÓN DE OUTLIERS (IQR)
# ==========================================================

print("\n=== DETECCIÓN DE OUTLIERS (IQR) ===")

percentiles = df.approxQuantile(
    valor_col,
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
    (col(valor_col) < lower_bound) |
    (col(valor_col) > upper_bound)
).count()

print(f"Outliers detectados: {num_outliers:,} ({(num_outliers/total_registros)*100:.2f}%)")


# ==========================================================
# 7. TOP 10 CONTRATOS MÁS ALTOS (VALIDACIÓN)
# ==========================================================

print("\n=== TOP 10 CONTRATOS POR VALOR ===")

df.orderBy(col(valor_col).desc()) \
  .select(
      "nombre_entidad",
      "departamento",
      valor_col
  ) \
  .show(10, truncate=False)


# ==========================================================
# 8. DISTRIBUCIÓN POR ENTIDAD
# ==========================================================

entidad_cols = [c for c in df.columns if 'entidad' in c.lower()]

if entidad_cols:
    entidad_col = entidad_cols[0]

    print("\n=== TOP 10 ENTIDADES POR VALOR TOTAL CONTRATADO ===")

    df.groupBy(entidad_col) \
        .agg(
            count("*").alias("num_contratos"),
            spark_sum(valor_col).alias("valor_total"),
            avg(valor_col).alias("valor_promedio")
        ) \
        .orderBy(desc("valor_total")) \
        .show(10, truncate=False)


# ==========================================================
# 9. ANÁLISIS TEMPORAL
# ==========================================================

fecha_cols = [c for c in df.columns if 'fecha' in c.lower()]

if fecha_cols:
    fecha_col = fecha_cols[0]

    df = df.withColumn("fecha_parsed", to_date(col(fecha_col)))
    df = df.withColumn("anio", year(col("fecha_parsed")))
    df = df.withColumn("mes", month(col("fecha_parsed")))

    print("\n=== CONTRATOS POR AÑO ===")

    df.groupBy("anio") \
        .agg(count("*").alias("num_contratos")) \
        .orderBy("anio") \
        .show()


# ==========================================================
# 10. GUARDAR DATASET EDA LIMPIO
# ==========================================================

processed_path = "/opt/spark-data/processed/secop_eda"

os.makedirs("/opt/spark-data/processed", exist_ok=True)

df.write.mode("overwrite").parquet(processed_path)

print(f"\n✓ Dataset EDA guardado en: {processed_path}")


# ==========================================================
# 11. CERRAR SESIÓN
# ==========================================================

spark.stop()
print("SparkSession finalizada correctamente")
