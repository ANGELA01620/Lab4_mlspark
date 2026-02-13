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

#  RETO 1: Proporción Train vs Test

# Seleccionamos la opción B) 70/30 – Balance clásico.

# Utilizamos una división 70% entrenamiento y 30% prueba porque representa
# un equilibrio adecuado entre capacidad de aprendizaje y validación robusta.
# Con 132,641 registros, el conjunto de entrenamiento (92,851) es suficientemente
# grande para capturar patrones relevantes, mientras que el conjunto de prueba
# (39,790) permite una evaluación estadísticamente estable y confiable.

# Si tuviéramos 1 millón de registros, podríamos usar 80/20 o incluso 90/10,
# ya que el conjunto de prueba seguiría siendo lo suficientemente grande
# para evaluar el modelo con baja varianza.

# Si tuviéramos solo 1,000 registros, sería más recomendable aplicar
# validación cruzada (k-fold), ya que una división simple podría generar
# alta variabilidad en los resultados.

# Usar seed=42 garantiza reproducibilidad, es decir,
# que la partición sea idéntica en cada ejecución.


#  RETO 2: Interpretación de R²

# La respuesta correcta es:
# B) El modelo explica 65% de la varianza en los datos.

# El coeficiente de determinación R² representa la proporción
# de la variabilidad de la variable objetivo que es explicada por el modelo.

# No representa porcentaje de precisión ni porcentaje de error.

# En nuestro caso:
# R² Test (log) = 0.1791

# Esto significa que el modelo explica aproximadamente el 17.9%
# de la variabilidad del valor del contrato (en escala logarítmica).

# ¿Es 0.65 un buen R²?
# Depende del contexto:

# En problemas financieros o sociales complejos,
# un R² de 0.65 puede considerarse bueno.

# En sistemas físicos altamente determinísticos,
# sería moderado.

# En datasets ruidosos o con alta variabilidad estructural,
# puede ser un resultado sólido.

# En nuestro caso, el R² (~0.18) indica capacidad explicativa limitada,
# lo que sugiere que la relación no es estrictamente lineal.


#  RETO 5: Comparación Train vs Test

# Escenarios teóricos:

# A) R² train = 0.9, R² test = 0.85
# No hay overfitting significativo. El modelo generaliza bien.

# B) R² train = 0.6, R² test = 0.58
# Modelo estable pero posiblemente simple.
# No hay sobreajuste, puede haber leve underfitting.

# C) R² train = 0.95, R² test = 0.45
# Existe overfitting severo.
# El modelo memoriza el entrenamiento y no generaliza.

# Nuestro caso:

# No observamos overfitting significativo,
# ya que la diferencia entre R² Train y Test es pequeña (~0.0237).

# Tampoco existe sobreajuste severo.

# Sí identificamos underfitting leve,
# debido a que el R² es bajo (~0.18),
# lo que indica que el modelo lineal explica
# una fracción limitada de la varianza.

# Esto sugiere que la relación entre variables
# podría no ser estrictamente lineal.


#  RETO BONUS 1: Residuos

# En un modelo bien ajustado, los residuos deberían:

# Distribuirse aproximadamente de forma normal
# Estar centrados en cero
# No presentar patrón sistemático

# En nuestro análisis:

# Los residuos están razonablemente centrados en cero.
# Existe dispersión consistente con el R² moderado.
# No se observa sesgo extremo.

# Esto indica que el modelo no sobreestima ni subestima sistemáticamente,
# aunque su capacidad explicativa es limitada.


#  RETO 6: Interpretación de Coeficientes

# Un coeficiente positivo indica que al aumentar esa feature,
# el valor logarítmico del contrato tiende a aumentar.

# Un coeficiente negativo indica relación inversa.

# Un coeficiente con mayor valor absoluto implica mayor
# influencia relativa en la predicción.

# En nuestro modelo, las features 18, 11 y 9
# muestran mayor impacto relativo.

# Sin embargo, debido al uso de PCA,
# la interpretación directa es limitada,
# ya que cada componente principal representa
# una combinación lineal de variables originales.


#  Preguntas 

# 1️ ¿Por qué usar RMSE en lugar de solo MAE?

# El RMSE penaliza más los errores grandes
# debido al término cuadrático.
# Es útil cuando los errores extremos
# tienen impacto económico significativo.

# 2️ Si todas las predicciones fueran iguales
# al promedio de los labels, ¿cuál sería el R²?

# El R² sería aproximadamente 0,
# ya que el modelo no estaría explicando
# variabilidad adicional respecto a la media.

# 3️ ¿Cuándo preferir RMSE vs MAE?

# Preferiríamos:

# RMSE cuando errores grandes son críticos
# y deben penalizarse más.

# MAE cuando buscamos mayor robustez
# frente a outliers.

# 4️ ¿Cómo mejorar el modelo?

# Podríamos:

# Aplicar regularización (L1, L2 o ElasticNet).
# Probar modelos no lineales como RandomForest
# o Gradient Boosted Trees.
# Realizar ingeniería de variables adicional.
# Detectar y tratar outliers extremos.
# Ajustar hiperparámetros mediante validación cruzada.
# Incorporar variables categóricas adicionales relevantes.


#  Conclusión General

# Nuestro modelo lineal:

# No presenta overfitting significativo.
# Presenta underfitting leve.
# Explica una fracción limitada de la varianza.
# Se beneficia de la transformación logarítmica.
# Puede mejorarse mediante modelos no lineales
# o regularización.
