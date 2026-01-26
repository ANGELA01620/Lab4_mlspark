# %% [markdown]
# # Notebook 03: Feature Engineering con Pipelines
#
# **Sección 13 - Spark ML**: Construcción de pipelines end-to-end
#
# **Objetivo**: Aplicar VectorAssembler y construir un pipeline de transformación.
#
# **Conceptos clave**:
# - **Transformer**: Aplica transformaciones (ej: StringIndexer)
# - **Estimator**: Aprende de los datos y genera un modelo
# - **Pipeline**: Encadena múltiples stages secuencialmente
#
# ## Actividades:
# 1. Crear StringIndexer para variables categóricas
# 2. Aplicar OneHotEncoder
# 3. Combinar features con VectorAssembler
# 4. Construir y ejecutar Pipeline

# %%
from pyspark.sql import SparkSession
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
)
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, when, isnull

# %%
# Configurar SparkSession
spark = SparkSession.builder \
    .appName("SECOP_FeatureEngineering") \
    .master("spark://spark-master:7077") \
    .config("spark.executor.memory", "2g") \
    .getOrCreate()

# %%
# Cargar datos
df = spark.read.parquet("/opt/spark-data/processed/secop_eda.parquet")
print(f"Registros cargados: {df.count():,}")

# %%
# Seleccionar features para el modelo
# Variables categóricas
categorical_cols = ["departamento", "tipo_de_contrato", "estado_contrato"]

# Variables numéricas
numeric_cols = ["plazo_de_ejec_del_contrato", "valor_del_contrato_num"]

# Verificar qué columnas existen
available_cat = [c for c in categorical_cols if c in df.columns]
available_num = [c for c in numeric_cols if c in df.columns]

print(f"Categóricas disponibles: {available_cat}")
print(f"Numéricas disponibles: {available_num}")

# %%
# Limpiar datos: eliminar nulos
df_clean = df.dropna(subset=available_cat + available_num)
print(f"Registros después de limpiar nulos: {df_clean.count():,}")

# %%
# PASO 1: StringIndexer para variables categóricas
# Convierte strings a índices numéricos
indexers = [
    StringIndexer(inputCol=col, outputCol=col + "_idx", handleInvalid="keep")
    for col in available_cat
]

print("StringIndexers creados:")
for idx in indexers:
    print(f"  - {idx.getInputCol()} -> {idx.getOutputCol()}")

# %%
# PASO 2: OneHotEncoder para generar variables dummy
encoders = [
    OneHotEncoder(inputCol=col + "_idx", outputCol=col + "_vec")
    for col in available_cat
]

print("\nOneHotEncoders creados:")
for enc in encoders:
    print(f"  - {enc.getInputCol()} -> {enc.getOutputCol()}")

# %%
# PASO 3: VectorAssembler para combinar todas las features
# Combinamos: features numéricas + features categóricas codificadas
feature_cols = available_num + [col + "_vec" for col in available_cat]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features_raw"
)

print(f"\nVectorAssembler combinará: {feature_cols}")

# %%
# PASO 4: Construir Pipeline
# Pipeline = secuencia de transformaciones
pipeline_stages = indexers + encoders + [assembler]

pipeline = Pipeline(stages=pipeline_stages)

print(f"\nPipeline con {len(pipeline_stages)} stages:")
for i, stage in enumerate(pipeline_stages):
    print(f"  Stage {i+1}: {type(stage).__name__}")

# %%
# PASO 5: Entrenar el pipeline (fit)
# Nota: StringIndexer y OneHotEncoder necesitan "aprender" del dataset
print("\nEntrenando pipeline...")
pipeline_model = pipeline.fit(df_clean)
print("Pipeline entrenado exitosamente")

# %%
# PASO 6: Aplicar transformaciones (transform)
df_transformed = pipeline_model.transform(df_clean)

print("\nTransformación completada")
print(f"Columnas después de transformar: {len(df_transformed.columns)}")

# %%
# Verificar el resultado
print("\nEsquema de features_raw:")
df_transformed.select("features_raw").printSchema()

# Ver dimensión del vector de features
sample_features = df_transformed.select("features_raw").first()[0]
print(f"Dimensión del vector de features: {len(sample_features)}")

# %%
# Mostrar ejemplo de transformación
df_transformed.select(
    available_cat[0] if available_cat else "id",
    available_cat[0] + "_idx" if available_cat else "id",
    available_cat[0] + "_vec" if available_cat else "id",
    "features_raw"
).show(5, truncate=True)

# %%
# Guardar pipeline entrenado
pipeline_path = "/opt/spark-data/processed/feature_pipeline"
pipeline_model.save(pipeline_path)
print(f"\nPipeline guardado en: {pipeline_path}")

# %%
# Guardar dataset transformado
output_path = "/opt/spark-data/processed/secop_features.parquet"
df_transformed.write.mode("overwrite").parquet(output_path)
print(f"Dataset transformado guardado en: {output_path}")

# %%
print("\n" + "="*60)
print("RESUMEN FEATURE ENGINEERING")
print("="*60)
print(f"✓ Variables categóricas procesadas: {len(available_cat)}")
print(f"✓ Variables numéricas: {len(available_num)}")
print(f"✓ Dimensión final del vector: {len(sample_features)}")
print(f"✓ Pipeline guardado y listo para usar")
print("="*60)

# %%
spark.stop()
