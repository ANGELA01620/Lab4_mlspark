# %% [markdown]
# # Notebook 05: Regresión Lineal
#
# **Sección 14 - Regresión**: Predicción del valor de contratos
#
# **Objetivo**: Entrenar un modelo de regresión lineal para predecir el precio base.
#
# ## Actividades:
# 1. Dividir datos en train/test
# 2. Entrenar LinearRegression
# 3. Evaluar con RMSE, MAE, R²
# 4. Analizar coeficientes

# %%
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col

# %%
spark = SparkSession.builder \
    .appName("SECOP_RegresionLineal") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

# %%
# Cargar datos
df = spark.read.parquet("/opt/spark-data/processed/secop_ml_ready.parquet")

# Renombrar columnas para consistencia
df = df.withColumnRenamed("valor_del_contrato_num", "label") \
       .withColumnRenamed("features_pca", "features")

# Filtrar valores nulos
df = df.filter(col("label").isNotNull())
print(f"Registros: {df.count():,}")

# %%
# PASO 1: Dividir en train (70%) y test (30%)
train, test = df.randomSplit([0.7, 0.3], seed=42)

print(f"Train: {train.count():,} registros")
print(f"Test: {test.count():,} registros")

# %%
# PASO 2: Crear modelo de Regresión Lineal
lr = LinearRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=0.0,  # Sin regularización (modelo base)
    elasticNetParam=0.0
)

print("Entrenando modelo de regresión lineal...")

# %%
# PASO 3: Entrenar el modelo
lr_model = lr.fit(train)

print("✓ Modelo entrenado")
print(f"  Iteraciones: {lr_model.summary.totalIterations}")
print(f"  RMSE (train): {lr_model.summary.rootMeanSquaredError:,.2f}")
print(f"  R² (train): {lr_model.summary.r2:.4f}")

# %%
# PASO 4: Hacer predicciones en test
predictions = lr_model.transform(test)

print("\n=== PREDICCIONES EN TEST ===")
predictions.select("label", "prediction").show(10)

# %%
# PASO 5: Evaluar el modelo
evaluator_rmse = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="rmse"
)

evaluator_mae = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="mae"
)

evaluator_r2 = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="r2"
)

rmse = evaluator_rmse.evaluate(predictions)
mae = evaluator_mae.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

print("\n" + "="*60)
print("MÉTRICAS DEL MODELO")
print("="*60)
print(f"RMSE (Test): ${rmse:,.2f}")
print(f"MAE (Test):  ${mae:,.2f}")
print(f"R² (Test):   {r2:.4f}")
print("="*60)

# %%
# PASO 6: Analizar coeficientes
coefficients = lr_model.coefficients
intercept = lr_model.intercept

print(f"\nIntercept: {intercept:,.2f}")
print(f"Número de coeficientes: {len(coefficients)}")
print(f"Primeros 5 coeficientes: {coefficients[:5]}")

# %%
# Guardar modelo
model_path = "/opt/spark-data/processed/linear_regression_model"
lr_model.save(model_path)
print(f"\nModelo guardado en: {model_path}")

# %%
# Guardar predicciones
predictions_path = "/opt/spark-data/processed/predictions_lr.parquet"
predictions.write.mode("overwrite").parquet(predictions_path)
print(f"Predicciones guardadas en: {predictions_path}")

# %%
spark.stop()
