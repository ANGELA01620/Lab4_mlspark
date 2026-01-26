# %% [markdown]
# # Notebook 08: Validación Cruzada (K-Fold)
#
# **Sección 15 - Tuning**: Cross-validation para evitar overfitting
#
# **Objetivo**: Implementar K-Fold Cross-Validation
#
# ## Conceptos clave:
# - Divide datos en K folds
# - Entrena K veces, usando diferente fold como validación
# - Promedia métricas para obtener estimación robusta

# %%
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import col

# %%
spark = SparkSession.builder \
    .appName("SECOP_CrossValidation") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

# %%
# Cargar datos
df = spark.read.parquet("/opt/spark-data/processed/secop_ml_ready.parquet")
df = df.withColumnRenamed("valor_del_contrato_num", "label") \
       .withColumnRenamed("features_pca", "features") \
       .filter(col("label").isNotNull())

train, test = df.randomSplit([0.8, 0.2], seed=42)

print(f"Train: {train.count():,}")
print(f"Test: {test.count():,}")

# %%
# Crear modelo base
lr = LinearRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100
)

# %%
# Crear grid de hiperparámetros (simple para este ejemplo)
param_grid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

print(f"Total de combinaciones: {len(param_grid)}")

# %%
# Configurar evaluador
evaluator = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="rmse"
)

# %%
# Configurar CrossValidator
# numFolds = 5 significa K-Fold con K=5
crossval = CrossValidator(
    estimator=lr,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=5,  # K=5
    seed=42
)

print("\nEntrenando con Cross-Validation (K=5)...")
print("Esto puede tardar varios minutos...")

# %%
# Entrenar con cross-validation
cv_model = crossval.fit(train)

print("\n✓ Cross-validation completada")

# %%
# Obtener métricas de cada fold
avg_metrics = cv_model.avgMetrics
best_metric_idx = avg_metrics.index(min(avg_metrics))

print("\n=== MÉTRICAS PROMEDIO POR CONFIGURACIÓN ===")
for i, metric in enumerate(avg_metrics):
    params = param_grid[i]
    reg = params.get(lr.regParam)
    elastic = params.get(lr.elasticNetParam)
    print(f"Config {i+1}: λ={reg:.2f}, α={elastic:.1f} -> RMSE={metric:,.2f}")

# %%
# Mejor modelo
best_model = cv_model.bestModel

print("\n=== MEJOR MODELO (Cross-Validation) ===")
print(f"regParam (λ): {best_model.getRegParam()}")
print(f"elasticNetParam (α): {best_model.getElasticNetParam()}")
print(f"RMSE promedio (CV): ${avg_metrics[best_metric_idx]:,.2f}")

# %%
# Evaluar en test set
predictions = best_model.transform(test)
rmse_test = evaluator.evaluate(predictions)

print(f"\nRMSE en Test: ${rmse_test:,.2f}")

# %%
# Guardar mejor modelo
model_path = "/opt/spark-data/processed/cv_best_model"
best_model.save(model_path)
print(f"\nModelo guardado en: {model_path}")

# %%
spark.stop()
