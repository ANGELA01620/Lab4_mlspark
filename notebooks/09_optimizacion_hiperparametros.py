# %% [markdown]
# # Notebook 09: Optimización de Hiperparámetros
#
# **Sección 15**: Grid Search y Random Search
#
# **Objetivo**: Encontrar la mejor combinación de hiperparámetros
#
# ## Estrategias:
# - **Grid Search**: Búsqueda exhaustiva
# - **Random Search**: Muestreo aleatorio (más eficiente)

# %%
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit
from pyspark.sql.functions import col
import time

# %%
spark = SparkSession.builder \
    .appName("SECOP_HyperparameterTuning") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

# %%
# Cargar datos
df = spark.read.parquet("/opt/spark-data/processed/secop_ml_ready.parquet")
df = df.withColumnRenamed("valor_del_contrato_num", "label") \
       .withColumnRenamed("features_pca", "features") \
       .filter(col("label").isNotNull())

train, test = df.randomSplit([0.8, 0.2], seed=42)

# %%
# Modelo base
lr = LinearRegression(featuresCol="features", labelCol="label", maxIter=100)

# Evaluador
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")

# %%
# =====================================
# ESTRATEGIA 1: GRID SEARCH (Exhaustivo)
# =====================================

print("="*60)
print("ESTRATEGIA 1: GRID SEARCH")
print("="*60)

# Grid de hiperparámetros
grid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .addGrid(lr.maxIter, [50, 100, 200]) \
    .build()

print(f"Combinaciones totales: {len(grid)}")

# CrossValidator con Grid Search
cv_grid = CrossValidator(
    estimator=lr,
    estimatorParamMaps=grid,
    evaluator=evaluator,
    numFolds=3,  # K=3 para mayor rapidez
    seed=42
)

print("\nEntrenando Grid Search...")
start_time = time.time()
cv_grid_model = cv_grid.fit(train)
grid_time = time.time() - start_time

print(f"✓ Completado en {grid_time:.2f} segundos")

# Mejor modelo de Grid Search
best_grid_model = cv_grid_model.bestModel
predictions_grid = best_grid_model.transform(test)
rmse_grid = evaluator.evaluate(predictions_grid)

print("\n=== MEJOR MODELO (Grid Search) ===")
print(f"regParam: {best_grid_model.getRegParam()}")
print(f"elasticNetParam: {best_grid_model.getElasticNetParam()}")
print(f"maxIter: {best_grid_model.getMaxIter()}")
print(f"RMSE Test: ${rmse_grid:,.2f}")

# %%
# =====================================
# ESTRATEGIA 2: TRAIN-VALIDATION SPLIT
# (Alternativa más rápida que Cross-Validation)
# =====================================

print("\n" + "="*60)
print("ESTRATEGIA 2: TRAIN-VALIDATION SPLIT")
print("="*60)

# Mismo grid de hiperparámetros
tvs = TrainValidationSplit(
    estimator=lr,
    estimatorParamMaps=grid,
    evaluator=evaluator,
    trainRatio=0.8,  # 80% train, 20% validation
    seed=42
)

print("\nEntrenando con Train-Validation Split...")
start_time = time.time()
tvs_model = tvs.fit(train)
tvs_time = time.time() - start_time

print(f"✓ Completado en {tvs_time:.2f} segundos")

# Mejor modelo
best_tvs_model = tvs_model.bestModel
predictions_tvs = best_tvs_model.transform(test)
rmse_tvs = evaluator.evaluate(predictions_tvs)

print("\n=== MEJOR MODELO (Train-Validation Split) ===")
print(f"regParam: {best_tvs_model.getRegParam()}")
print(f"elasticNetParam: {best_tvs_model.getElasticNetParam()}")
print(f"maxIter: {best_tvs_model.getMaxIter()}")
print(f"RMSE Test: ${rmse_tvs:,.2f}")

# %%
# Comparación de estrategias
print("\n" + "="*60)
print("COMPARACIÓN DE ESTRATEGIAS")
print("="*60)
print(f"Grid Search + CV:")
print(f"  - Tiempo: {grid_time:.2f}s")
print(f"  - RMSE Test: ${rmse_grid:,.2f}")
print(f"\nTrain-Validation Split:")
print(f"  - Tiempo: {tvs_time:.2f}s")
print(f"  - RMSE Test: ${rmse_tvs:,.2f}")
print(f"\nMejor estrategia: {'Grid Search + CV' if rmse_grid < rmse_tvs else 'Train-Validation Split'}")
print("="*60)

# %%
# Guardar mejor modelo global
mejor_modelo = best_grid_model if rmse_grid < rmse_tvs else best_tvs_model
model_path = "/opt/spark-data/processed/tuned_model"
mejor_modelo.save(model_path)

print(f"\nMejor modelo guardado en: {model_path}")

# %%
# Guardar hiperparámetros óptimos
import json

hiperparametros_optimos = {
    "regParam": float(mejor_modelo.getRegParam()),
    "elasticNetParam": float(mejor_modelo.getElasticNetParam()),
    "maxIter": int(mejor_modelo.getMaxIter()),
    "rmse_test": float(rmse_grid if rmse_grid < rmse_tvs else rmse_tvs)
}

with open("/opt/spark-data/processed/hiperparametros_optimos.json", "w") as f:
    json.dump(hiperparametros_optimos, f, indent=2)

print("Hiperparámetros óptimos guardados")

# %%
spark.stop()
