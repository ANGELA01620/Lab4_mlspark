# %% [markdown]
# # Notebook 12: Inferencia en Producción
#
# **Sección 16 - MLOps**: Despliegue y predicciones batch
#
# **Objetivo**: Simular un pipeline de producción para generar predicciones
#
# ## Actividades:
# 1. Cargar modelo desde Model Registry (Production)
# 2. Aplicar transformaciones del pipeline
# 3. Generar predicciones batch sobre datos nuevos
# 4. Guardar resultados para consumo

# %%
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp
import mlflow
import mlflow.spark

# %%
spark = SparkSession.builder \
    .appName("SECOP_Produccion") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

# %%
# Configurar MLflow
mlflow.set_tracking_uri("http://mlflow:5000")

print("="*60)
print("PIPELINE DE INFERENCIA EN PRODUCCIÓN")
print("="*60)

# %%
# =====================================
# PASO 1: Cargar modelo en producción
# =====================================

print("\n=== PASO 1: Cargar Modelo en Producción ===")

model_name = "secop_prediccion_contratos"
model_uri = f"models:/{model_name}/Production"

print(f"Cargando modelo desde: {model_uri}")
production_model = mlflow.spark.load_model(model_uri)
print("✓ Modelo cargado exitosamente")

# %%
# =====================================
# PASO 2: Cargar datos nuevos para predicción
# =====================================

print("\n=== PASO 2: Cargar Datos Nuevos ===")

# En producción, estos serían contratos nuevos sin valor conocido
# Para el ejercicio, usamos el test set

df_new = spark.read.parquet("/opt/spark-data/processed/secop_ml_ready.parquet")

# Renombrar columnas
df_new = df_new.withColumnRenamed("features_pca", "features")

# Simular datos "nuevos" (sin label)
df_new_no_label = df_new.drop("valor_del_contrato_num")

print(f"Contratos para predecir: {df_new_no_label.count():,}")

# %%
# =====================================
# PASO 3: Generar predicciones batch
# =====================================

print("\n=== PASO 3: Generar Predicciones Batch ===")

print("Aplicando modelo a datos nuevos...")
predictions_batch = production_model.transform(df_new_no_label)

# Añadir timestamp de predicción
predictions_batch = predictions_batch.withColumn("prediction_timestamp", current_timestamp())

print("✓ Predicciones generadas")

# %%
# Ver muestra de predicciones
print("\n=== MUESTRA DE PREDICCIONES ===")
predictions_batch.select("features", "prediction", "prediction_timestamp").show(10, truncate=True)

# %%
# Estadísticas de las predicciones
from pyspark.sql.functions import min as spark_min, max as spark_max, avg, stddev, count

stats = predictions_batch.select(
    spark_min("prediction").alias("min_pred"),
    spark_max("prediction").alias("max_pred"),
    avg("prediction").alias("avg_pred"),
    stddev("prediction").alias("std_pred"),
    count("*").alias("total")
).collect()[0]

print("\n=== ESTADÍSTICAS DE PREDICCIONES ===")
print(f"Total predicciones: {stats['total']:,}")
print(f"Valor mínimo predicho: ${stats['min_pred']:,.2f}")
print(f"Valor máximo predicho: ${stats['max_pred']:,.2f}")
print(f"Valor promedio: ${stats['avg_pred']:,.2f}")
print(f"Desviación estándar: ${stats['std_pred']:,.2f}")

# %%
# =====================================
# PASO 4: Guardar resultados para consumo
# =====================================

print("\n=== PASO 4: Guardar Resultados ===")

# Directorio de predicciones
predictions_output = "/opt/spark-data/processed/predictions_produccion"

# Guardar en formato Parquet (optimizado para analytics)
predictions_batch.write.mode("overwrite").parquet(predictions_output + "/parquet")
print(f"✓ Predicciones guardadas (Parquet): {predictions_output}/parquet")

# Guardar también en CSV para consumo externo
predictions_batch.select("prediction", "prediction_timestamp") \
    .write.mode("overwrite") \
    .option("header", "true") \
    .csv(predictions_output + "/csv")
print(f"✓ Predicciones guardadas (CSV): {predictions_output}/csv")

# %%
# =====================================
# PASO 5: Monitoreo básico de predicciones
# =====================================

print("\n=== PASO 5: Monitoreo de Predicciones ===")

# Distribución por rangos de valor predicho
from pyspark.sql.functions import when

prediction_ranges = predictions_batch.select(
    count(when(col("prediction") < 10000000, True)).alias("< 10M"),
    count(when((col("prediction") >= 10000000) & (col("prediction") < 100000000), True)).alias("10M-100M"),
    count(when((col("prediction") >= 100000000) & (col("prediction") < 1000000000), True)).alias("100M-1B"),
    count(when(col("prediction") >= 1000000000, True)).alias("> 1B")
)

print("\n=== DISTRIBUCIÓN DE PREDICCIONES POR RANGO ===")
prediction_ranges.show()

# %%
# =====================================
# PASO 6: Simular scoring continuo
# =====================================

print("\n=== PASO 6: Scoring Continuo (Simulación) ===")

# En producción, este proceso se ejecutaría periódicamente (ej: cada hora)
# y procesaría solo los contratos nuevos desde la última ejecución

print("En producción, este notebook se ejecutaría como:")
print("  - Cron job diario/horario")
print("  - Trigger en Airflow/Luigi")
print("  - Stream processing con Spark Streaming")

# %%
# Resumen final
print("\n" + "="*60)
print("INFERENCIA EN PRODUCCIÓN COMPLETADA")
print("="*60)
print(f"✓ Modelo usado: {model_name} (Production)")
print(f"✓ Predicciones generadas: {stats['total']:,}")
print(f"✓ Valor promedio predicho: ${stats['avg_pred']:,.2f}")
print(f"✓ Resultados guardados en: {predictions_output}")
print("\nPróximos pasos:")
print("  1. Configurar monitoreo de data drift")
print("  2. Implementar A/B testing")
print("  3. Automatizar reentrenamiento periódico")
print("  4. Crear alertas de anomalías en predicciones")
print("="*60)

# %%
spark.stop()
