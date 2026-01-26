# %% [markdown]
# # Notebook 04: Transformaciones Avanzadas
#
# **Sección 13 - Spark ML**: Escalado y reducción de dimensionalidad
#
# **Objetivo**: Normalizar features y aplicar PCA
#
# ## Actividades:
# 1. Aplicar StandardScaler para normalizar
# 2. Reducir dimensionalidad con PCA
# 3. Crear pipeline completo optimizado

# %%
from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler, PCA, Imputer
from pyspark.ml import Pipeline, PipelineModel

# %%
spark = SparkSession.builder \
    .appName("SECOP_Transformaciones") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

# %%
# Cargar datos con features
df = spark.read.parquet("/opt/spark-data/processed/secop_features.parquet")
print(f"Registros: {df.count():,}")

# %%
# PASO 1: StandardScaler - Normaliza features (media=0, std=1)
scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features_scaled",
    withMean=False,  # Sparse vectors no soportan withMean=True
    withStd=True
)

print("Aplicando StandardScaler...")
scaler_model = scaler.fit(df)
df_scaled = scaler_model.transform(df)

print("✓ Features escaladas")

# %%
# PASO 2: PCA - Reducción de dimensionalidad
# Reducir a las k componentes principales
k = 20  # Mantener 20 dimensiones principales

pca = PCA(
    k=k,
    inputCol="features_scaled",
    outputCol="features_pca"
)

print(f"\nAplicando PCA (k={k})...")
pca_model = pca.fit(df_scaled)
df_pca = pca_model.transform(df_scaled)

# Ver varianza explicada
explained_variance = pca_model.explainedVariance
print(f"Varianza explicada por las {k} componentes:")
print(f"  Total: {sum(explained_variance):.4f}")
print(f"  Primera componente: {explained_variance[0]:.4f}")

# %%
# Pipeline completo: Scaler + PCA
pipeline_transform = Pipeline(stages=[scaler, pca])
pipeline_transform_model = pipeline_transform.fit(df)
df_final = pipeline_transform_model.transform(df)

# %%
# Seleccionar columnas finales para ML
df_ml_ready = df_final.select(
    "features_pca",
    "valor_del_contrato_num"  # Variable objetivo
)

print("\n=== DATASET LISTO PARA ML ===")
df_ml_ready.show(5)

# %%
# Guardar dataset final
output_path = "/opt/spark-data/processed/secop_ml_ready.parquet"
df_ml_ready.write.mode("overwrite").parquet(output_path)
print(f"\nDataset listo para ML guardado en: {output_path}")

# %%
# Guardar pipeline de transformación
pipeline_path = "/opt/spark-data/processed/transformation_pipeline"
pipeline_transform_model.save(pipeline_path)
print(f"Pipeline de transformación guardado en: {pipeline_path}")

# %%
spark.stop()
