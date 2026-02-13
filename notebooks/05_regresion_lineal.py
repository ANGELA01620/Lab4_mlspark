# ============================================================
# Notebook 05: Regresión Lineal
# Objetivo: Predecir valor del contrato (escala log)
# ============================================================

from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, log1p
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Crear sesión Spark
# ============================================================

spark = SparkSession.builder \
    .appName("SECOP_RegresionLineal") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

# ============================================================
# 2. Cargar datos
# ============================================================

df = spark.read.parquet("/opt/spark-data/processed/secop_ml_ready.parquet")

# El parquet ya contiene: label y features_pca
df = df.withColumnRenamed("features_pca", "features")

df = df.filter(col("label").isNotNull())

print(f"Registros totales: {df.count():,}")

# ============================================================
# 3. Transformación log (CRÍTICO para estabilidad)
# ============================================================

df = df.withColumn("label_log", log1p(col("label")))

# ============================================================
# 4. Train/Test Split
# ============================================================

train, test = df.randomSplit([0.7, 0.3], seed=42)

print(f"Train: {train.count():,}")
print(f"Test:  {test.count():,}")

# ============================================================
# 5. Configurar modelo (baseline lineal)
# ============================================================

lr = LinearRegression(
    featuresCol="features",
    labelCol="label_log",
    maxIter=100,
    regParam=0.0,
    elasticNetParam=0.0
)

print("Modelo configurado")

# ============================================================
# 6. Entrenamiento
# ============================================================

print("Entrenando modelo...")
lr_model = lr.fit(train)

print("Modelo entrenado")
print(f"Iteraciones: {lr_model.summary.totalIterations}")
print(f"RMSE Train (log): {lr_model.summary.rootMeanSquaredError:.4f}")
print(f"R² Train (log):   {lr_model.summary.r2:.4f}")

# ============================================================
# 7. Predicciones
# ============================================================

predictions = lr_model.transform(test)

predictions.select("label_log", "prediction").show(10)

# ============================================================
# 8. Evaluación formal (ESCALA LOG)
# ============================================================

evaluator_rmse = RegressionEvaluator(
    labelCol="label_log",
    predictionCol="prediction",
    metricName="rmse"
)

evaluator_mae = RegressionEvaluator(
    labelCol="label_log",
    predictionCol="prediction",
    metricName="mae"
)

evaluator_r2 = RegressionEvaluator(
    labelCol="label_log",
    predictionCol="prediction",
    metricName="r2"
)

rmse = evaluator_rmse.evaluate(predictions)
mae = evaluator_mae.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

print("\n" + "="*60)
print("MÉTRICAS TEST (ESCALA LOG)")
print("="*60)
print(f"RMSE (log): {rmse:.4f}")
print(f"MAE  (log): {mae:.4f}")
print(f"R²   (log): {r2:.4f}")
print("="*60)

# ============================================================
# 9. Comparación Train vs Test
# ============================================================

print("\n=== COMPARACIÓN TRAIN VS TEST ===")
print(f"R² Train (log): {lr_model.summary.r2:.4f}")
print(f"R² Test  (log): {r2:.4f}")
print(f"Diferencia: {abs(lr_model.summary.r2 - r2):.4f}")

# ============================================================
# 10. Análisis de coeficientes
# ============================================================

coefficients = lr_model.coefficients
intercept = lr_model.intercept

print(f"\nIntercept (log): {intercept:.4f}")
print(f"Número de features: {len(coefficients)}")

coef_array = np.array(coefficients)
abs_coefs = np.abs(coef_array)
top_idx = np.argsort(abs_coefs)[-5:]

print("\nTop 5 coeficientes más influyentes:")
for idx in reversed(top_idx):
    print(f"Feature {idx} | coef = {coef_array[idx]:.6f}")

# ============================================================
# 11. Análisis de residuos (escala log)
# ============================================================

predictions = predictions.withColumn(
    "residual_log",
    col("label_log") - col("prediction")
)

residuals_sample = predictions.select("residual_log") \
                               .sample(0.2, seed=42) \
                               .toPandas()

plt.figure(figsize=(10,5))
plt.hist(residuals_sample["residual_log"], bins=60)
plt.axvline(x=0)
plt.title("Distribución de Residuos (Escala Log)")
plt.xlabel("Residuo log")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig("/opt/spark-data/processed/residuals_log_distribution.png")

print("Gráfico de residuos guardado")

# ============================================================
# 12. Guardar modelo y predicciones
# ============================================================

model_path = "/opt/spark-data/processed/linear_regression_model_log"
lr_model.write().overwrite().save(model_path)

predictions_path = "/opt/spark-data/processed/predictions_lr_log.parquet"
predictions.write.mode("overwrite").parquet(predictions_path)

print("Modelo y predicciones guardados correctamente")

# ============================================================
# 13. Resumen final
# ============================================================

print("\n" + "="*60)
print("RESUMEN FINAL - REGRESIÓN LINEAL")
print("="*60)
print(f"Registros Train: {train.count():,}")
print(f"Registros Test:  {test.count():,}")
print(f"RMSE (log): {rmse:.4f}")
print(f"R² (log):   {r2:.4f}")
print("Modelo evaluado correctamente en escala log")
print("="*60)

spark.stop()

# Respuesta reto 1:
# Se utilizó una división 70/30 (70% entrenamiento, 30% prueba),
# ya que representa un balance clásico entre capacidad de aprendizaje
# del modelo y validación robusta.
#
# Con 132,641 registros, el conjunto de entrenamiento (92,851)
# es suficientemente grande para capturar patrones,
# mientras que el conjunto de prueba (39,790) permite
# una evaluación estadísticamente estable.
#
# Si el dataset tuviera 1 millón de registros,
# podría utilizarse 80/20 o incluso 90/10,
# ya que el conjunto de test seguiría siendo grande.
#
# Si el dataset tuviera solo 1,000 registros,
# sería más recomendable usar validación cruzada
# para evitar alta varianza en la evaluación.
#
# Es importante usar seed=42 para garantizar reproducibilidad,
# es decir, que la partición sea exactamente la misma
# en cada ejecución del experimento.

# Reto 2:
# Se configuró un modelo baseline sin regularización (regParam=0.0)
# con el objetivo de evaluar primero el comportamiento lineal puro.
#
# maxIter=100 es suficiente dado que el optimizador
# convergió inmediatamente (0 iteraciones),
# lo que indica que el problema se resolvió
# mediante solución analítica cerrada.
#
# Este modelo sirve como punto de comparación
# antes de aplicar regularización en el Notebook 07.

# Reto 3 R² significa:
# El coeficiente de determinación R² representa
# la proporción de la varianza del valor del contrato
# que es explicada por el modelo.
#
# No representa porcentaje de precisión.
#
# En este caso:
# R² Test (log) = 0.1791
# Esto significa que el modelo explica aproximadamente
# el 17.9% de la variabilidad del valor del contrato.
#
# Reto 4:
# Se observan errores elevados en contratos de gran magnitud.
# Esto indica presencia de alta dispersión y contratos outliers.
#
# La transformación logarítmica redujo significativamente
# la explosión numérica observada en la escala original.
#
# Los mayores errores tienden a concentrarse en contratos
# extremadamente grandes, lo cual sugiere que el modelo lineal
# no captura completamente la dinámica de contratos de alto valor.
#
# Reto 5:
# No se observa overfitting significativo,
# ya que la diferencia entre R² Train y Test es pequeña (0.0237).
#
# Tampoco existe sobreajuste severo.
#
# Sin embargo, el modelo presenta underfitting leve,
# debido a que el R² es bajo (~0.18),
# lo que indica que el modelo lineal explica
# una fracción limitada de la varianza.
#
# Esto sugiere que la relación entre variables
# podría no ser estrictamente lineal.

# Interpretación:
# Un coeficiente positivo indica que al aumentar esa feature,
# el valor logarítmico del contrato tiende a aumentar.
#
# Un coeficiente negativo indica relación inversa.
#
# Un coeficiente grande en valor absoluto implica
# mayor influencia relativa en la predicción.
#
# En este modelo, las features 18, 11 y 9
# muestran mayor impacto en la predicción.
#
# Debido a que se utilizó PCA,
# la interpretación directa es limitada,
# ya que cada componente principal representa
# una combinación lineal de variables originales.


# Interpretación de residuos:
# En un modelo bien ajustado,
# los residuos deberían distribuirse aproximadamente
# de forma normal y centrados en cero.
#
# En este caso, los residuos están razonablemente centrados,
# aunque presentan cierta dispersión,
# consistente con el R² moderadamente bajo.
#
# No se observa sesgo extremo,
# lo cual indica que el modelo no sobreestima
# ni subestima sistemáticamente.

# Nota:
# El análisis de feature importance removiendo
# una feature a la vez es computacionalmente costoso.
#
# Dado el tamaño del dataset (132k registros),
# este experimento podría ejecutarse,
# pero se recomienda realizarlo únicamente
# si se requiere análisis interpretativo profundo.
#
# Alternativamente, modelos como RandomForest
# proveen importance de manera nativa.


# 1. ¿Por qué usar RMSE en lugar de solo MAE?
# RMSE penaliza más los errores grandes,
# lo que lo hace útil cuando los outliers
# tienen impacto económico significativo.

# 2. Si todas las predicciones fueran = promedio de labels,
# el R² sería aproximadamente 0.

# 3. Preferiría RMSE cuando errores grandes son críticos.
# Preferiría MAE cuando se busca robustez ante outliers.

# 4. ¿Cómo mejorar el modelo?
# - Aplicar regularización (L1, L2 o ElasticNet)
# - Probar modelos no lineales (RandomForest, GBT)
# - Realizar ingeniería de features adicional
# - Detectar y tratar outliers extremos
# - Ajustar hiperparámetros mediante validación cruzada


