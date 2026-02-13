# %% [markdown]
# # Notebook 02: Análisis Exploratorio de Datos (EDA)
#
# **Objetivo**: Entender la distribución de las variables,
# identificar valores nulos y outliers.

# %%
# Importar librerías
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, sum as spark_sum, avg, min as spark_min,
    max as spark_max, stddev, isnan, when, isnull, desc,
    to_timestamp
)
from pyspark.sql.types import DoubleType, LongType, IntegerType
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Configurar Spark
spark = SparkSession.builder \
    .appName("SECOP_EDA") \
    .master("spark://spark-master:7077") \
    .config("spark.executor.memory", "2g") \
    .getOrCreate()

print(f"Spark Version: {spark.version}")

# %%
# Cargar datos
parquet_path = "/opt/spark-data/raw/secop_bogota_2025_q1.parquet"

df = spark.read.parquet(parquet_path)

print(f"Registros cargados: {df.count():,}")
print(f"Columnas: {len(df.columns)}")

print("\n=== ESQUEMA ORIGINAL ===")
df.printSchema()

# ============================================================
# CONVERSIÓN DE TIPOS
# ============================================================

fecha_cols = [
    "fecha_de_firma",
    "fecha_de_inicio_del_contrato",
    "fecha_de_fin_del_contrato",
    "ultima_actualizacion",
    "fecha_inicio_liquidacion",
    "fecha_fin_liquidacion",
    "fecha_de_notificaci_n_de_prorrogaci_n"
]

for c in fecha_cols:
    if c in df.columns:
        df = df.withColumn(c, to_timestamp(col(c)))

double_cols = [
    "valor_del_contrato",
    "valor_de_pago_adelantado",
    "valor_facturado",
    "valor_pendiente_de_pago",
    "valor_pagado",
    "valor_amortizado",
    "valor_pendiente_de",
    "valor_pendiente_de_ejecucion",
    "presupuesto_general_de_la_nacion_pgn",
    "sistema_general_de_participaciones",
    "recursos_propios_alcald_as_gobernaciones_y_resguardos_ind_genas_",
    "recursos_de_credito",
    "recursos_propios"
]

for c in double_cols:
    if c in df.columns:
        df = df.withColumn(c, col(c).cast(DoubleType()))

int_cols = [
    "dias_adicionados",
    "codigo_entidad",
    "codigo_proveedor",
    "sistema_general_de_regal_as"
]

for c in int_cols:
    if c in df.columns:
        df = df.withColumn(c, col(c).cast(IntegerType()))

long_cols = ["saldo_cdp", "saldo_vigencia"]

for c in long_cols:
    if c in df.columns:
        df = df.withColumn(c, col(c).cast(LongType()))

print("\n=== ESQUEMA DESPUÉS DE CAST ===")
df.printSchema()

# %%
# Primeras filas
df.show(10, truncate=True)

# %%
# Estadísticas generales
df.describe().show()

# %%
# Valores nulos
from pyspark.sql.types import DoubleType, FloatType

exprs = []

for c in df.columns:
    dtype = dict(df.dtypes)[c]

    # Si es numérico (double o float), usar isnan
    if dtype in ["double", "float"]:
        expr = count(when(isnull(col(c)) | isnan(col(c)), c)).alias(c)
    else:
        expr = count(when(isnull(col(c)), c)).alias(c)

    exprs.append(expr)

null_counts = df.select(exprs)

null_df = null_counts.toPandas().T
null_df.columns = ['null_count']
null_df['null_percentage'] = (null_df['null_count'] / df.count()) * 100
null_df = null_df.sort_values('null_count', ascending=False)

print(null_df[null_df['null_count'] > 0])

# %%



# ============================================================
# ANÁLISIS VALOR DEL CONTRATO
# ============================================================

valor_cols = [c for c in df.columns if 'valor' in c.lower() or 'precio' in c.lower()]

if valor_cols:
    valor_col = valor_cols[0]
    df = df.withColumn(valor_col + "_num", col(valor_col).cast("double"))

    df.select(
        spark_min(col(valor_col + "_num")).alias("Min"),
        spark_max(col(valor_col + "_num")).alias("Max"),
        avg(col(valor_col + "_num")).alias("Promedio"),
        stddev(col(valor_col + "_num")).alias("Desv_Std")
    ).show()

    df.select(
        count(when(col(valor_col + "_num") < 10000000, True)).alias("< 10M"),
        count(when((col(valor_col + "_num") >= 10000000) & (col(valor_col + "_num") < 100000000), True)).alias("10M-100M"),
        count(when((col(valor_col + "_num") >= 100000000) & (col(valor_col + "_num") < 1000000000), True)).alias("100M-1B"),
        count(when(col(valor_col + "_num") >= 1000000000, True)).alias("> 1B")
    ).show()

# %%
# ============================================================
# TOP 10 ENTIDADES
# ============================================================

entidad_cols = [c for c in df.columns if 'entidad' in c.lower()]

if entidad_cols and valor_cols:
    entidad_col = entidad_cols[0]

    df.groupBy(entidad_col) \
        .agg(
            count("*").alias("num_contratos"),
            spark_sum(col(valor_col + "_num")).alias("valor_total"),
            avg(col(valor_col + "_num")).alias("valor_promedio")
        ) \
        .orderBy(desc("valor_total")) \
        .show(10, truncate=False)

# %%
# ============================================================
# TIPO DE CONTRATO
# ============================================================

tipo_cols = [c for c in df.columns if 'tipo' in c.lower()]
if tipo_cols:
    df.groupBy(tipo_cols[0]) \
        .count() \
        .orderBy(desc("count")) \
        .show(20, truncate=False)

# %%
# ============================================================
# VISUALIZACIÓN TOP 10 ENTIDADES POR VALOR TOTAL
# ============================================================

from pyspark.sql.functions import sum as spark_sum, desc
import matplotlib.pyplot as plt

entidad_col = "nombre_entidad"
valor_col = "valor_del_contrato"

# Agrupar por entidad y ordenar por valor total
df_ent = df.groupBy(entidad_col) \
    .agg(
        spark_sum(valor_col).alias("valor_total")
    ) \
    .orderBy(desc("valor_total")) \
    .limit(10)

# Convertir a pandas para visualización
df_ent_pandas = df_ent.toPandas()

# Crear gráfica
plt.figure(figsize=(10, 6))

plt.barh(
    df_ent_pandas[entidad_col],
    df_ent_pandas['valor_total'] / 1e9  # convertir a miles de millones
)

plt.xlabel('Valor Total (Miles de Millones COP)')
plt.title('Top 10 Entidades por Valor Total Contratado')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

# Guardar imagen
plt.savefig(
    '/opt/spark-data/processed/eda_top10_entidades_valor.png',
    dpi=150,
    bbox_inches='tight'
)

print("\nGráfico guardado: /opt/spark-data/processed/eda_top10_entidades_valor.png")

# %%
# ============================================================
# ESTADO DEL CONTRATO
# ============================================================

estado_cols = [c for c in df.columns if 'estado' in c.lower()]
if estado_cols:
    df.groupBy(estado_cols[0]) \
        .count() \
        .orderBy(desc("count")) \
        .show(20, truncate=False)

# %%
# ============================================================
# TOP 10 PROVEEDORES
# ============================================================

proveedor_cols = [c for c in df.columns if 'proveedor' in c.lower()]
if proveedor_cols and valor_cols:
    df.groupBy(proveedor_cols[0]) \
        .agg(
            count("*").alias("num_contratos"),
            spark_sum(col(valor_col + "_num")).alias("valor_total")
        ) \
        .orderBy(desc("valor_total")) \
        .show(10, truncate=False)

# %%
# ============================================================
# OUTLIERS (IQR)
# ============================================================

if valor_cols:
    quantiles = df.approxQuantile(valor_col + "_num", [0.25, 0.75], 0.01)
    Q1, Q3 = quantiles
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df.filter(
        (col(valor_col + "_num") < lower) |
        (col(valor_col + "_num") > upper)
    ).count()

    print(f"Outliers detectados: {outliers:,}")

# %%
# ============================================================
# ANÁLISIS TEMPORAL
# ============================================================

fecha_cols_present = [c for c in fecha_cols if c in df.columns]

if fecha_cols_present:
    fecha_col = fecha_cols_present[0]

    df.groupBy(
        df[fecha_col].substr(1, 4).alias("anio")
    ).agg(
        count("*").alias("num_contratos"),
        spark_sum(col(valor_col + "_num")).alias("valor_total") if valor_cols else count("*")
    ).orderBy("anio").show()

# %%
# Guardar dataset procesado
output_path = "/opt/spark-data/processed/secop_eda.parquet"
df.write.mode("overwrite").parquet(output_path)

spark.stop()
print("EDA finalizado correctamente")
