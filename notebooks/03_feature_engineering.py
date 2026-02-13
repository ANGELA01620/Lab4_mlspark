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

# ==========================================================
# Notebook 03: Feature Engineering con Pipelines
# ==========================================================

from pyspark.sql import SparkSession
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler
)
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, year
import numpy as np

# ----------------------------------------------------------
# Configurar SparkSession
# ----------------------------------------------------------

spark = SparkSession.builder \
    .appName("SECOP_FeatureEngineering") \
    .master("spark://spark-master:7077") \
    .config("spark.executor.memory", "2g") \
    .getOrCreate()

# ----------------------------------------------------------
# Cargar datos
# ----------------------------------------------------------

df = spark.read.parquet("/opt/spark-data/processed/secop_eda.parquet")
print(f"Registros cargados: {df.count():,}")

print("Columnas disponibles:")
for col_name in df.columns:
    print(f"  - {col_name}")

# ----------------------------------------------------------
# RETO 1: Selección de Features
# ----------------------------------------------------------

# Variable derivada numérica
df = df.withColumn("anio", year(col("fecha_de_firma")))

categorical_cols = [
    "departamento",
    "tipo_de_contrato",
    "estado_contrato"
]

numeric_cols = [
    "valor_del_contrato_num",
    "anio"
]

available_cat = [c for c in categorical_cols if c in df.columns]
available_num = [c for c in numeric_cols if c in df.columns]

print(f"Categóricas seleccionadas: {available_cat}")
print(f"Numéricas seleccionadas: {available_num}")

# ----------------------------------------------------------
# RETO 2: Limpieza
# ----------------------------------------------------------

df_clean = df.dropna(subset=available_cat + available_num)
print(f"Registros después de limpiar: {df_clean.count():,}")

# ----------------------------------------------------------
# PASO 1: StringIndexer
# ----------------------------------------------------------

indexers = [
    StringIndexer(
        inputCol=col_name,
        outputCol=col_name + "_idx",
        handleInvalid="keep"
    )
    for col_name in available_cat
]

# ----------------------------------------------------------
# PASO 2: OneHotEncoder
# ----------------------------------------------------------

encoders = [
    OneHotEncoder(
        inputCol=col_name + "_idx",
        outputCol=col_name + "_vec"
    )
    for col_name in available_cat
]

# ----------------------------------------------------------
# RETO 3: VectorAssembler
# ----------------------------------------------------------

feature_cols = (
    available_num +
    [col_name + "_vec" for col_name in available_cat]
)

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features_raw"
)

print(f"\nVectorAssembler combinará {len(feature_cols)} features:")
print(feature_cols)

# ----------------------------------------------------------
# RETO 4: Pipeline
# ----------------------------------------------------------

pipeline_stages = indexers + encoders + [assembler]

pipeline = Pipeline(stages=pipeline_stages)

print(f"\nPipeline con {len(pipeline_stages)} stages")

# ----------------------------------------------------------
# Entrenar y transformar
# ----------------------------------------------------------

print("\nEntrenando pipeline...")
pipeline_model = pipeline.fit(df_clean)
print("✓ Pipeline entrenado")

df_transformed = pipeline_model.transform(df_clean)
print("✓ Transformación completada")

# ----------------------------------------------------------
# Verificar resultado
# ----------------------------------------------------------

df_transformed.select("features_raw").printSchema()

sample_features = df_transformed.select("features_raw").first()[0]
print(f"Dimensión del vector de features: {len(sample_features)}")

# ----------------------------------------------------------
# BONUS 1
# ----------------------------------------------------------

print("\nCategorías únicas por variable categórica:")
for cat_col in available_cat:
    num_categorias = df_clean.select(cat_col).distinct().count()
    print(f"{cat_col}: {num_categorias}")

# ----------------------------------------------------------
# BONUS 2
# ----------------------------------------------------------

sample_df = df_transformed.select("features_raw").sample(0.01).limit(1000).toPandas()
features_matrix = np.array([row['features_raw'].toArray() for _, row in sample_df.iterrows()])

variances = np.var(features_matrix, axis=0)
top_5_idx = np.argsort(variances)[-5:]

print("\nTop 5 features con mayor varianza:")
for idx in top_5_idx:
    print(f"Feature {idx}: varianza = {variances[idx]:.2f}")

# ----------------------------------------------------------
# Guardar pipeline
# ----------------------------------------------------------

pipeline_path = "/opt/spark-data/processed/feature_pipeline"
pipeline_model.write().overwrite().save(pipeline_path)
print(f"\n✓ Pipeline guardado en: {pipeline_path}")

# ----------------------------------------------------------
# Guardar dataset transformado
# ----------------------------------------------------------

output_path = "/opt/spark-data/processed/secop_features.parquet"
df_transformed.write.mode("overwrite").parquet(output_path)
print(f"✓ Dataset transformado guardado en: {output_path}")

# ----------------------------------------------------------
# Resumen
# ----------------------------------------------------------

print("\n" + "="*60)
print("RESUMEN FEATURE ENGINEERING")
print("="*60)
print(f"✓ Variables categóricas procesadas: {len(available_cat)}")
print(f"✓ Variables numéricas: {len(available_num)}")
print(f"✓ Dimensión final del vector: {len(sample_features)}")
print("✓ Pipeline guardado y listo para usar")
print("="*60)

spark.stop()

# =============================================================
# RETO 1: Selección de Features
# =============================================================
# Después de revisar las columnas disponibles en el dataset,
# seleccionamos variables que consideramos relevantes para
# explicar y modelar el valor del contrato.
#
# Variables categóricas seleccionadas:
# - departamento
# - tipo_de_contrato
# - estado_contrato
#
# Justificación:
# Aunque el dataset fue filtrado únicamente para el
# "Distrito Capital de Bogotá", decidimos mantener la variable
# "departamento" para conservar consistencia estructural del
# pipeline y permitir escalabilidad futura si se amplía el
# alcance geográfico.
#
# "tipo_de_contrato" y "estado_contrato" son variables clave
# porque reflejan la naturaleza jurídica y la condición
# administrativa del contrato, lo cual puede influir en su valor.
#
# Variables numéricas seleccionadas:
# - valor_del_contrato_num
# - anio
#
# Justificación:
# "valor_del_contrato_num" es la variable cuantitativa principal.
# "anio" permite capturar posibles efectos temporales.
# =============================================================

# =============================================================
# RETO 2: Estrategia de limpieza de datos
# =============================================================
# Optamos por eliminar registros con valores nulos en las
# variables seleccionadas.
#
# Justificación:
# - El dataset ya viene previamente depurado desde el EDA.
# - La proporción de nulos es mínima.
# - Evitamos introducir sesgos mediante imputaciones
#   artificiales en variables contractuales.
#
# Consideramos que para esta fase del laboratorio es más
# apropiado trabajar con datos completos y consistentes.
# =============================================================

# =============================================================
# RETO 3: VectorAssembler
# =============================================================
# Combinamos variables numéricas originales y variables
# categóricas codificadas en un único vector.
#
# Esto es necesario porque los algoritmos de Machine Learning
# en Spark trabajan con una sola columna vectorial de entrada.
#
# El ensamblaje permite integrar información heterogénea
# (numérica y categórica transformada) en una representación
# matemática uniforme.
# =============================================================

# =============================================================
# RETO 4: Construcción del Pipeline
# =============================================================
# El orden correcto de los stages es:
#
# 1. StringIndexer  → convierte texto en índices numéricos.
# 2. OneHotEncoder  → transforma índices en vectores binarios.
# 3. VectorAssembler → combina todas las features en un vector.
#
# Este orden es fundamental porque cada etapa depende de
# la salida de la anterior.
#
# El uso de Pipeline garantiza reproducibilidad, orden lógico
# y facilidad de despliegue en producción.
# =============================================================

# =============================================================
# BONUS 1: Cálculo total de features
# =============================================================
# En nuestro caso:
#
# Variables numéricas: 2
# departamento: 1 categoría (debido al filtro a Bogotá)
# tipo_de_contrato: 18 categorías
# estado_contrato: 7 categorías
#
# Total features =
# 2 + 1 + 18 + 7 = 28
#
# Este valor coincide con la dimensión observada del
# vector "features_raw".
# =============================================================

# =============================================================
# BONUS 2: Análisis de varianza
# =============================================================
# Tomamos una muestra del dataset transformado y calculamos
# la varianza de cada dimensión del vector de features.
#
# Observamos que la mayor varianza corresponde a
# "valor_del_contrato_num", lo cual es coherente debido
# a la magnitud monetaria de los contratos.
#
# Esto sugiere que en un escenario de modelado real sería
# recomendable aplicar StandardScaler para evitar que
# esta variable domine el entrenamiento del modelo.
# =============================================================

# =============================================================
# RESPUESTAS DE REFLEXIÓN
# =============================================================

# 1. Pipeline:
# Utilizamos Pipeline porque permite estructurar el flujo de
# transformaciones de manera secuencial y reproducible.
# Facilita el entrenamiento, validación y posterior despliegue
# del modelo sin repetir manualmente cada transformación.

# 2. Orden de transformaciones:
# Si aplicáramos OneHotEncoder antes de StringIndexer,
# el proceso fallaría, ya que el encoder requiere índices
# numéricos como entrada y no valores de texto.

# 3. StandardScaler:
# Lo usaríamos cuando las variables numéricas presentan
# magnitudes muy diferentes (como ocurre con el valor del contrato),
# especialmente en modelos sensibles a la escala como
# regresión lineal o K-Means.

# 4. Guardar pipeline:
# Guardar el pipeline_model permite aplicar exactamente las
# mismas transformaciones a nuevos datos en el futuro,
# garantizando coherencia entre entrenamiento y producción.
# =============================================================

