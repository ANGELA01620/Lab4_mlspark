# ============================================================
# NOTEBOOK 09: OPTIMIZACIÓN DE HIPERPARÁMETROS
# Sección 15: Grid Search y Train-Validation Split
# Objetivo: Encontrar la mejor combinación de hiperparámetros
# ============================================================

from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit
from pyspark.sql.functions import col
import time
import json

# ============================================================
#  Inicializar Spark
# ============================================================

spark = SparkSession.builder \
    .appName("SECOP_HyperparameterTuning") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

# ============================================================
#  Cargar datos
# ============================================================
from pyspark.sql.functions import col, log1p

df = spark.read.parquet("/opt/spark-data/processed/secop_ml_ready.parquet")

df = df.withColumnRenamed("valor_del_contrato_num", "label") \
       .withColumnRenamed("features_pca", "features") \
       .filter(col("label").isNotNull())

df = df.withColumn("label", log1p(col("label")))

train, test = df.randomSplit([0.8, 0.2], seed=42)

print(f"Train: {train.count():,}")
print(f"Test: {test.count():,}")

# ============================================================
#  Modelo base y evaluador
# ============================================================

lr = LinearRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100
)

evaluator = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="rmse"
)

# ============================================================
#  RETO 1: Grid de Hiperparámetros
# ============================================================

grid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .addGrid(lr.maxIter, [50, 100, 200]) \
    .build()

print(f"Combinaciones totales: {len(grid)}")

# ============================================================
#  RETO 2: Grid Search + Cross Validation
# ============================================================

cv_grid = CrossValidator(
    estimator=lr,
    estimatorParamMaps=grid,
    evaluator=evaluator,
    numFolds=3,
    seed=42
)

print("\nEntrenando Grid Search + CV...")
start_time = time.time()
cv_grid_model = cv_grid.fit(train)
grid_time = time.time() - start_time
print(f"Completado en {grid_time:.2f} segundos")

best_grid_model = cv_grid_model.bestModel
predictions_grid = best_grid_model.transform(test)
rmse_grid = evaluator.evaluate(predictions_grid)

print("\n=== MEJOR MODELO (Grid Search + CV) ===")
print(f"regParam: {best_grid_model.getRegParam()}")
print(f"elasticNetParam: {best_grid_model.getElasticNetParam()}")
print(f"maxIter: {best_grid_model.getMaxIter()}")
print(f"RMSE Test: {rmse_grid:,.4f}")

# ============================================================
#  RETO 3: Train-Validation Split
# ============================================================

tvs = TrainValidationSplit(
    estimator=lr,
    estimatorParamMaps=grid,
    evaluator=evaluator,
    trainRatio=0.8,
    seed=42
)

print("\nEntrenando Train-Validation Split...")
start_time = time.time()
tvs_model = tvs.fit(train)
tvs_time = time.time() - start_time
print(f"Completado en {tvs_time:.2f} segundos")

best_tvs_model = tvs_model.bestModel
predictions_tvs = best_tvs_model.transform(test)
rmse_tvs = evaluator.evaluate(predictions_tvs)

print("\n=== MEJOR MODELO (Train-Validation Split) ===")
print(f"regParam: {best_tvs_model.getRegParam()}")
print(f"elasticNetParam: {best_tvs_model.getElasticNetParam()}")
print(f"maxIter: {best_tvs_model.getMaxIter()}")
print(f"RMSE Test: {rmse_tvs:,.4f}")

# ============================================================
#  RETO 4: Comparación
# ============================================================

print("\n" + "="*60)
print("COMPARACIÓN DE ESTRATEGIAS")
print("="*60)

print(f"\nGrid Search + CV:")
print(f" - Tiempo: {grid_time:.2f}s")
print(f" - RMSE Test: {rmse_grid:,.4f}")
print(f" - λ={best_grid_model.getRegParam()}, α={best_grid_model.getElasticNetParam()}")

print(f"\nTrain-Validation Split:")
print(f" - Tiempo: {tvs_time:.2f}s")
print(f" - RMSE Test: {rmse_tvs:,.4f}")
print(f" - λ={best_tvs_model.getRegParam()}, α={best_tvs_model.getElasticNetParam()}")

# ============================================================
#  RETO 5: Seleccionar y Guardar Modelo Final
# ============================================================

mejor_modelo = best_grid_model if rmse_grid < rmse_tvs else best_tvs_model
mejor_rmse = rmse_grid if rmse_grid < rmse_tvs else rmse_tvs
estrategia = "Grid Search + CV" if rmse_grid < rmse_tvs else "Train-Validation Split"

model_path = "/opt/spark-data/processed/tuned_model"
mejor_modelo.write().overwrite().save(model_path)

print(f"\nMejor modelo guardado en: {model_path}")

hiperparametros_optimos = {
    "regParam": float(mejor_modelo.getRegParam()),
    "elasticNetParam": float(mejor_modelo.getElasticNetParam()),
    "maxIter": int(mejor_modelo.getMaxIter()),
    "rmse_test": float(mejor_rmse),
    "estrategia": estrategia
}

with open("/opt/spark-data/processed/hiperparametros_optimos.json", "w") as f:
    json.dump(hiperparametros_optimos, f, indent=2)

print("Hiperparámetros óptimos guardados")

# ============================================================
#  RETO BONUS: Grid Más Fino
# ============================================================

mejor_lambda = mejor_modelo.getRegParam()
mejor_alpha = mejor_modelo.getElasticNetParam()

grid_fino = ParamGridBuilder() \
    .addGrid(lr.regParam, [
        mejor_lambda * 0.5,
        mejor_lambda * 0.8,
        mejor_lambda,
        mejor_lambda * 1.2,
        mejor_lambda * 1.5
    ]) \
    .addGrid(lr.elasticNetParam, [
        max(0.0, mejor_alpha - 0.2),
        mejor_alpha,
        min(1.0, mejor_alpha + 0.2)
    ]) \
    .addGrid(lr.maxIter, [best_grid_model.getMaxIter()]) \
    .build()

cv_fino = CrossValidator(
    estimator=lr,
    estimatorParamMaps=grid_fino,
    evaluator=evaluator,
    numFolds=3,
    seed=42
)

print("\nEntrenando Grid Fino...")
start_time = time.time()
cv_fino_model = cv_fino.fit(train)
fino_time = time.time() - start_time

best_fino_model = cv_fino_model.bestModel
predictions_fino = best_fino_model.transform(test)
rmse_fino = evaluator.evaluate(predictions_fino)

print("\n=== RESULTADO GRID FINO ===")
print(f"RMSE Test: {rmse_fino:,.4f}")
print(f"Tiempo: {fino_time:.2f}s")

##########################################################################
#                SOLUCION PREGUNTAS DE LOS RETOS Y REFLEXIONES                 #
##########################################################################
#
#  ¿POR QUÉ USAMOS ESCALA LOGARÍTMICA PARA regParam?
# --------------------------------------------------------------------------
# Usamos escala logarítmica (0.01, 0.1, 1.0) porque la regularización afecta 
# al modelo de forma no lineal. Cambios en órdenes de magnitud permiten 
# explorar intensidades (débil, media, fuerte) de manera más eficiente que 
# una escala lineal, donde los cambios pequeños apenas variarían el modelo.
#
#  COMBINACIONES Y MODELOS ENTRENADOS (K=3)
# --------------------------------------------------------------------------
# El grid genera:
# 3 (regParam) × 3 (elasticNetParam) × 3 (maxIter) = 27 combinaciones.
#
# Con K=3 en Cross-Validation:
# 27 combinaciones × 3 Folds = 81 modelos entrenados en total.
#
#  ¿POR QUÉ K=3 Y NO K=5?
# --------------------------------------------------------------------------
# Ofrece un balance óptimo entre robustez y tiempo de cómputo. Con más de 
# 100k registros, K=3 ya es estadísticamente estable. K=5 aumentaría el 
# tiempo de entrenamiento un 66% sin garantizar una mejora significativa.
#
#  ESTRATEGIA: RMSE Y VELOCIDAD
# --------------------------------------------------------------------------
# - MEJOR RMSE: Train-Validation Split (1.1947) vs Grid Search + CV (1.1977).
# - MÁS RÁPIDA: Train-Validation Split (~20s vs ~94s).
# - HIPERPARÁMETROS: No coincidieron. CV eligió λ=0.1, α=0.5; mientras que 
#   TVS eligió λ=0.01, α=0.0.
#
#  ¿MEJORÓ EL RMSE CON EL GRID FINO?
# --------------------------------------------------------------------------
# Sí. El grid fino redujo el RMSE a 1.1939. Esto confirma que refinar la 
# búsqueda en la "zona ganadora" permite extraer mejoras marginales.
#
#  GRID SEARCH VS RANDOM SEARCH
# --------------------------------------------------------------------------
# - GRID SEARCH: Espacios pequeños/manejables. Evaluación exhaustiva.
# - RANDOM SEARCH: Espacios grandes o continuos. Más eficiente al explorar 
#   regiones diversas sin probar cada combinación posible.
#
#  ¿POR QUÉ TVS ES MÁS RÁPIDO QUE CV?
# --------------------------------------------------------------------------
# Porque entrena cada combinación una sola vez (divide los datos en 2 partes).
# Cross-Validation entrena cada combinación K veces (una por cada Fold).
# En este caso: TVS = 27 modelos | CV = 81 modelos.
#
#  RIESGOS DE UN GRID DEMASIADO GRANDE
# --------------------------------------------------------------------------
# El costo crece exponencialmente. Combinaciones × Folds = mayor tiempo y 
# memoria. Puede saturar el cluster de Spark y volver el proceso impráctico.
#
#  IMPLEMENTACIÓN DE RANDOM SEARCH EN SPARK
# --------------------------------------------------------------------------
# Al no ser nativo, se simula:
# 1. Generando valores aleatorios con Python (random.sample).
# 2. Construyendo el ParamGridBuilder solo con esas muestras.
# 3. Usando el CrossValidator sobre ese subconjunto.
# *También se pueden integrar herramientas como Hyperopt.
#
##########################################################################

print("\n" + "="*60)
print("RESUMEN OPTIMIZACIÓN COMPLETADO")
print("="*60)

spark.stop()

