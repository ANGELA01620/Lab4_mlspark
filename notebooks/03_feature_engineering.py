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

# ==========================================================
# 1. IMPORTAR LIBRERÍAS
# ==========================================================

from pyspark.sql import SparkSession
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler
)
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType, IntegerType, LongType
import os


# ==========================================================
# 2. CREAR SPARK SESSION
# ==========================================================

spark = SparkSession.builder \
    .appName("SECOP_FeatureEngineering_FINAL") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("Spark Version:", spark.version)


# ==========================================================
# 3. CARGAR DATASET DESDE EDA
# ==========================================================

input_path = "/opt/spark-data/processed/secop_eda"
df = spark.read.parquet(input_path)

print(f"Registros cargados: {df.count():,}")
print(f"Columnas disponibles: {len(df.columns)}")

df.printSchema()


# ==========================================================
# 4. DEFINIR VARIABLE OBJETIVO (ALINEADO CON EDA)
# ==========================================================

if "valor_del_contrato" not in df.columns:
    raise ValueError("No se encontró 'valor_del_contrato'.")

label_col = "valor_del_contrato"

print(f"Variable objetivo: {label_col}")


# ==========================================================
# 5. SELECCIÓN DE FEATURES
# ==========================================================

# Categóricas (tipo string)
categorical_cols = [
    c for c in df.columns
    if df.schema[c].dataType.simpleString() == "string"
    and "fecha" not in c.lower()
]

# Limitar a 3 para evitar demasiadas columnas dummy
categorical_cols = categorical_cols[:3]

# Numéricas adicionales (excluyendo label)
numeric_cols = [
    c for c in df.columns
    if isinstance(df.schema[c].dataType, (DoubleType, IntegerType, LongType))
    and c != label_col
]

numeric_cols = numeric_cols[:3]

print("Categóricas seleccionadas:", categorical_cols)
print("Numéricas seleccionadas:", numeric_cols)


# ==========================================================
# 6. LIMPIEZA DE DATOS
# ==========================================================

df_clean = df.dropna(subset=categorical_cols + numeric_cols + [label_col])

print(f"Registros después de limpiar: {df_clean.count():,}")


# ==========================================================
# 7. STRING INDEXERS
# ==========================================================

indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=c + "_idx",
        handleInvalid="keep"
    )
    for c in categorical_cols
]


# ==========================================================
# 8. ONE HOT ENCODERS
# ==========================================================

encoders = [
    OneHotEncoder(
        inputCol=c + "_idx",
        outputCol=c + "_vec"
    )
    for c in categorical_cols
]


# ==========================================================
# 9. VECTOR ASSEMBLER
# ==========================================================

feature_cols = numeric_cols + [c + "_vec" for c in categorical_cols]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)


# ==========================================================
# 10. CONSTRUIR PIPELINE
# ==========================================================

pipeline_stages = indexers + encoders + [assembler]

pipeline = Pipeline(stages=pipeline_stages)

print(f"Pipeline con {len(pipeline_stages)} stages creado correctamente")


# ==========================================================
# 11. ENTRENAR Y TRANSFORMAR
# ==========================================================

print("Entrenando pipeline...")

pipeline_model = pipeline.fit(df_clean)

df_transformed = pipeline_model.transform(df_clean)

print("✓ Transformación completada")


# ==========================================================
# 12. VALIDAR VECTOR DE FEATURES
# ==========================================================

sample_vector = df_transformed.select("features").first()[0]

print(f"Dimensión final del vector: {len(sample_vector)}")


# ==========================================================
# 13. SELECCIONAR COLUMNAS FINALES (IMPORTANTE PARA CUADERNO 4)
# ==========================================================

df_final = df_transformed.select(
    label_col,
    "features"
)

df_final.show(5, truncate=False)


# ==========================================================
# 14. GUARDAR PIPELINE Y DATASET
# ==========================================================

os.makedirs("/opt/spark-data/processed", exist_ok=True)

pipeline_path = "/opt/spark-data/processed/feature_pipeline"
pipeline_model.write().overwrite().save(pipeline_path)

output_path = "/opt/spark-data/processed/secop_features.parquet"
df_final.write.mode("overwrite").parquet(output_path)

print("✓ Pipeline guardado correctamente")
print("✓ Dataset de features guardado correctamente")


# ==========================================================
# 15. RESUMEN
# ==========================================================

print("\n" + "="*60)
print("RESUMEN FEATURE ENGINEERING")
print("="*60)
print(f"✓ Variable objetivo: {label_col}")
print(f"✓ Variables categóricas procesadas: {len(categorical_cols)}")
print(f"✓ Variables numéricas: {len(numeric_cols)}")
print(f"✓ Dimensión final del vector: {len(sample_vector)}")
print("="*60)


spark.stop()
print("SparkSession finalizada correctamente")
