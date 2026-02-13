# %% [markdown]
# # Notebook 04: Transformaciones Avanzadas
#
# **Sección 13**: StandardScaler, PCA y Normalización
#
# **Objetivo**: Aplicar transformaciones avanzadas para mejorar el desempeño del modelo.
#
# ## Actividades:
# 1. Normalizar features numéricas con StandardScaler
# 2. Aplicar PCA para reducción de dimensionalidad
# 3. Construir pipeline completo
# 4. Comparar resultados con y sin transformaciones

# ============================================================
# NOTEBOOK 04: TRANSFORMACIONES AVANZADAS
# StandardScaler + PCA + Pipeline Completo
# ============================================================

from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler, PCA
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql.functions import col
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# INICIAR SPARK
# ============================================================

spark = SparkSession.builder \
    .appName("SECOP_Transformaciones") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

# ============================================================
# CARGAR DATASET
# ============================================================

df = spark.read.parquet("/opt/spark-data/processed/secop_features.parquet")

print("\n" + "="*60)
print("DATASET CARGADO")
print("="*60)
print(f"Registros: {df.count():,}")
print(f"Columnas: {len(df.columns)}")

# ============================================================
# RETO 1: ANALIZAR ESCALAS
# ============================================================

print("\n" + "="*60)
print("RETO 1: ANÁLISIS DE ESCALAS")
print("="*60)

sample = df.select("features_raw").limit(5).collect()

for i, row in enumerate(sample):
    arr = row["features_raw"].toArray()
    print(f"\nRegistro {i+1} - primeros 10 valores:")
    print(arr[:10])

sample_large = df.select("features_raw").limit(1000).toPandas()
matrix = np.array([row["features_raw"].toArray() for row in sample_large.to_dict("records")])

print("\nEstadísticas globales (features_raw):")
print(f"Min: {matrix.min():.4f}")
print(f"Max: {matrix.max():.4f}")
print(f"Mean: {matrix.mean():.4f}")
print(f"Std: {matrix.std():.4f}")

# ============================================================
# STANDARD SCALER
# ============================================================

scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features_scaled",
    withMean=False,
    withStd=True
)

scaler_model = scaler.fit(df)
df_scaled = scaler_model.transform(df)

print("\n✓ Features escaladas correctamente")

# ============================================================
# RETO 2: COMPARACIÓN ANTES VS DESPUÉS
# ============================================================

print("\n" + "="*60)
print("RETO 2: COMPARACIÓN ANTES vs DESPUÉS")
print("="*60)

sample_df = df_scaled.select("features_raw", "features_scaled").limit(1000).toPandas()

raw_matrix = np.array([row["features_raw"].toArray() for row in sample_df.to_dict("records")])
scaled_matrix = np.array([row["features_scaled"].toArray() for row in sample_df.to_dict("records")])

print("\nANTES:")
print(f"Min: {raw_matrix.min():.4f}")
print(f"Max: {raw_matrix.max():.4f}")
print(f"Mean: {raw_matrix.mean():.4f}")
print(f"Std: {raw_matrix.std():.4f}")

print("\nDESPUÉS:")
print(f"Min: {scaled_matrix.min():.4f}")
print(f"Max: {scaled_matrix.max():.4f}")
print(f"Mean: {scaled_matrix.mean():.4f}")
print(f"Std: {scaled_matrix.std():.4f}")

# ============================================================
# PCA
# ============================================================

sample_vec = df_scaled.select("features_scaled").first()[0]
num_features = len(sample_vec)

# Crear PCA con máximo posible
pca_full = PCA(
    k=num_features,
    inputCol="features_scaled",
    outputCol="features_pca_full"
)

pca_full_model = pca_full.fit(df_scaled)

explained_variance = pca_full_model.explainedVariance

# ============================================
# CÁLCULO DE VARIANZA ACUMULADA
# ============================================

cumulative = np.cumsum(explained_variance)

print("\n============================================================")
print("VARIANZA EXPLICADA")
print("============================================================")

for i, (var, cum) in enumerate(zip(explained_variance, cumulative)):
    print(f"PC{i+1}: {var*100:.2f}% | Acumulada: {cum*100:.2f}%")

# Encontrar mínimo k para ≥80%
components_80 = np.argmax(cumulative >= 0.80) + 1

if cumulative[-1] < 0.80:
    print("\n Ni usando todos los componentes se alcanza 80%")
else:
    print(f"\n Componentes necesarios para ≥80% varianza: {components_80}")

# ============================================================
# BONUS: EXPERIMENTO CON DIFERENTES k
# ============================================================

print("\n" + "="*60)
print("EXPERIMENTO PCA CON DIFERENTES k")
print("="*60)

k_values = [5, 10, 15, 20]
explained_vars = []

for k in k_values:
    k_real = min(k, num_features)
    pca_temp = PCA(k=k_real, inputCol="features_scaled", outputCol="temp_pca")
    model_temp = pca_temp.fit(df_scaled)
    var_acum = sum(model_temp.explainedVariance)
    explained_vars.append(var_acum)
    print(f"k={k_real}: {var_acum*100:.2f}% varianza")

plt.figure(figsize=(8, 5))
plt.plot(k_values[:len(explained_vars)],
         [v*100 for v in explained_vars],
         marker='o')
plt.xlabel("Número de Componentes (k)")
plt.ylabel("Varianza Explicada (%)")
plt.title("PCA: Varianza vs Componentes")
plt.grid(True)
plt.savefig("/opt/spark-data/processed/pca_variance.png")
print("✓ Gráfico guardado en /opt/spark-data/processed/pca_variance.png")


# ============================================
# CREAR PCA FINAL CON k ÓPTIMO
# ============================================

k_optimal = components_80

pca = PCA(
    k=k_optimal,
    inputCol="features_scaled",
    outputCol="features_pca"
)

print(f"✓ PCA final configurado con k={k_optimal}")

# ============================================
# PIPELINE SOLO CON SCALER + PCA
# ============================================

ml_pipeline = Pipeline(stages=[scaler, pca])

ml_pipeline_model = ml_pipeline.fit(df)

df_transformed = ml_pipeline_model.transform(df)

print("✓ Pipeline de transformaciones aplicado correctamente")

# ============================================================
# DATASET LISTO PARA ML
# ============================================================

if "valor_del_contrato_num" in df_transformed.columns:
    df_ml_ready = df_transformed.select(
        "features_pca",
        col("valor_del_contrato_num").alias("label")
    )
else:
    print("ADVERTENCIA: No se encontró columna de valor.")
    df_ml_ready = df_transformed

output_path = "/opt/spark-data/processed/secop_ml_ready.parquet"
df_ml_ready.write.mode("overwrite").parquet(output_path)

print(f"\n✓ Dataset ML-ready guardado en: {output_path}")

# ============================================================
# RESUMEN FINAL
# ============================================================

print("\n" + "="*60)
print("RESUMEN DE TRANSFORMACIONES")
print("="*60)
print(f"Features originales: {num_features}")
print(f"Features después de PCA: {k_optimal}")
print(f"Varianza explicada total: {sum(explained_variance)*100:.2f}%")
print("="*60)

spark.stop()



##########################################################################
#                      ANÁLISIS DE PREPROCESAMIENTO Y PCA                #
##########################################################################
#
#  RETO 1: ¿POR QUÉ NORMALIZAR?
# --------------------------------------------------------------------------
# Observación: Existen diferencias enormes en las magnitudes de los datos.
# - Min: 0
# - Max: 6,349,971,000
# - Mean: 2,432,870
# - Std: 41,929,653
#
# Conclusión: Es fundamental normalizar porque los algoritmos de ML 
# (KNN, Regresión, PCA) son sensibles a la escala. Sin esto, las variables 
# con valores grandes dominarían el modelo y el PCA solo capturaría la 
# varianza de las variables con mayor magnitud, sesgando los resultados.
#
#  RETO 2: COMPARACIÓN ANTES VS DESPUÉS (StandardScaler)
# --------------------------------------------------------------------------
# Los resultados confirman que el escalamiento fue exitoso:
# - ANTES: Max ~6.35e9 | Std ~4.19e7
# - DESPUÉS: Max 42.63 | Std 1.35
# ✔ Las magnitudes bajaron drásticamente y la desviación quedó cercana a 1.
#
#  RETO 3: ¿CUÁNTOS COMPONENTES CONSERVAR?
# --------------------------------------------------------------------------
# Respuesta Correcta: C) Los que expliquen suficiente varianza (80%–95%).
#
# Justificación: PCA busca reducir la dimensionalidad conservando la mayor 
# cantidad de información posible. No importa el número fijo, importa la 
# varianza explicada acumulada.
#
# Resultados del experimento:
# - k=5   -> 27.32% de varianza
# - k=10  -> 46.97% de varianza
# - k=15  -> 66.21% de varianza
# - k=19  -> 81.59% de varianza <-- PUNTO ÓPTIMO (Supera el 80%)
#
#
#  RETO 4: ¿CUÁNTOS COMPONENTES NECESITAS PARA ≥80%?
# --------------------------------------------------------------------------
# Respuesta: Se necesitan 20 componentes, los cuales alcanzan un 85.41% 
# de varianza explicada acumulada.
#
# RETO 5: ¿POR QUÉ ES IMPORTANTE EL ORDEN DEL PIPELINE?
# --------------------------------------------------------------------------
# Orden correcto: 1. Feature Engineering -> 2. StandardScaler -> 3. PCA
#
# Justificación: El PCA se basa estrictamente en el cálculo de la varianza.
# Si no se escala antes, las variables con magnitudes grandes dominarán 
# los componentes principales. StandardScaler garantiza que todas las 
# variables contribuyan de manera equitativa al modelo.
#  Nota: Si se invierte el orden, el resultado de PCA sería incorrecto.
#
# PREGUNTAS DE REFLEXIÓN
# --------------------------------------------------------------------------
#  ¿Por qué StandardScaler usa withMean=False?
# Debido a que las features suelen ser vectores dispersos (sparse) tras el
# OneHotEncoder. Centrar los datos (withMean=True) los convertiría en densos,
# disparando el consumo de memoria. Se usa por eficiencia en Spark.
#
#  ¿Cuándo NO deberías usar PCA?
# - Cuando la interpretabilidad de cada variable original es crítica.
# - Cuando el número de variables es pequeño.
# - Cuando el modelo ya es eficiente y preciso sin reducción.
# *PCA transforma las variables, por lo que se pierde la relación directa.
#
#  Si tienes 100 features y usas k=10, ¿perdiste información?
# Sí, técnicamente hay pérdida. Sin embargo, si esos 10 componentes explican
# entre el 90-95% de la varianza, la pérdida es mínima y aceptable a cambio
# de una mayor eficiencia. Es una compresión inteligente.
#
#  ¿Qué ventaja tiene aplicar StandardScaler ANTES de PCA?
# Garantiza que el PCA no sea sesgado. Al maximizar la varianza, el PCA 
# ignoraría variables importantes pero con escalas pequeñas si no se
# normalizan todas al mismo rango de peso.
#
##########################################################################
##########################################################################


