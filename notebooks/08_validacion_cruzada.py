# %% [markdown]
# ============================================================
# Notebook 08: Validación Cruzada (K-Fold)
# Sección 15 - Tuning
# ============================================================

from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import col
import time

# ============================================================
# INICIAR SPARK
# ============================================================

spark = SparkSession.builder \
    .appName("SECOP_CrossValidation") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

# ============================================================
# CARGAR DATOS
# ============================================================
from pyspark.sql.functions import log1p
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
# RETO 1: ENTENDER K-FOLD CROSS-VALIDATION
# ============================================================

# 1. ¿En cuántos subconjuntos se dividen los datos de train?
#    Respuesta: En K subconjuntos (si K=5 → 5 folds)

# 2. ¿Cuántos modelos se entrenan en total?
#    Respuesta: K modelos por cada combinación de hiperparámetros

# 3. ¿Qué porcentaje se usa para validación en cada iteración?
#    Respuesta: 1/K (si K=5 → 20%)

# 4. ¿Qué métrica se reporta al final?
#    Respuesta: El promedio de la métrica (ej. RMSE) sobre los K folds

# ¿Por qué es mejor que train/test simple?
#    Porque reduce la varianza de la estimación y usa mejor los datos,
#    dando una métrica más robusta y menos dependiente de una sola partición.

# ============================================================
# RETO 2: MODELO BASE Y EVALUADOR
# ============================================================

lr = LinearRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100
)

evaluator = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="rmse"   # También podría usarse "mae" o "r2"
)

# ============================================================
# RETO 3: PARAM GRID
# ============================================================

param_grid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

print(f"Combinaciones en el grid: {len(param_grid)}")

# Si agregas 3 valores de regParam y 3 de elasticNetParam:
# combinaciones = 3 x 3 = 9
# Con K=5:
# Total modelos = 9 x 5 = 45

# ============================================================
# RETO 4: CONFIGURAR CROSSVALIDATOR
# ============================================================

K = 5  # Elegimos 5 porque es el balance clásico entre robustez y costo

crossval = CrossValidator(
    estimator=lr,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=K,
    seed=42
)

print(f"Cross-Validation con K={K} folds")
print(f"Total modelos a entrenar: {len(param_grid) * K}")

# ============================================================
# RETO 5: EJECUTAR CROSS-VALIDATION
# ============================================================

print("Entrenando con Cross-Validation...")
start_time = time.time()

cv_model = crossval.fit(train)

elapsed_time = time.time() - start_time
print(f"Cross-validation completada en {elapsed_time:.2f} segundos")

avg_metrics = cv_model.avgMetrics
best_metric_idx = avg_metrics.index(min(avg_metrics))

print("\n=== MÉTRICAS PROMEDIO POR CONFIGURACIÓN ===")

for i, metric in enumerate(avg_metrics):
    params = param_grid[i]
    reg = params.get(lr.regParam)
    elastic = params.get(lr.elasticNetParam)
    marker = " <-- MEJOR" if i == best_metric_idx else ""
    print(f"Config {i+1}: λ={reg:.2f}, α={elastic:.1f} -> RMSE={metric:,.2f}{marker}")

best_model = cv_model.bestModel

print("\n=== MEJOR MODELO ===")
print(f"regParam: {best_model.getRegParam()}")
print(f"elasticNetParam: {best_model.getElasticNetParam()}")

predictions = best_model.transform(test)
rmse_test = evaluator.evaluate(predictions)

print(f"RMSE en Test: ${rmse_test:,.2f}")

# ============================================================
# RETO 6: COMPARAR CV vs SIMPLE SPLIT
# ============================================================

lr_simple = LinearRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=best_model.getRegParam(),
    elasticNetParam=best_model.getElasticNetParam()
)

model_simple = lr_simple.fit(train)
rmse_simple = evaluator.evaluate(model_simple.transform(test))

print("\n=== COMPARACIÓN ===")
print(f"RMSE con CV: ${rmse_test:,.2f}")
print(f"RMSE sin CV: ${rmse_simple:,.2f}")
print(f"Diferencia:  ${abs(rmse_test - rmse_simple):,.2f}")

# ¿Los resultados son similares? 
#    Si son similares → el modelo es estable.
#    Si son muy diferentes → el simple split era poco confiable.
#    en este caso son identicos.
# ¿Cuál método es más confiable?
#    Cross-validation, porque evalúa múltiples particiones y reduce varianza.

# ============================================================
# RETO BONUS: DIFERENTES VALORES DE K
# ============================================================

print("\n=== EXPERIMENTO CON DIFERENTES K ===")

for k in [3, 5, 10]:
    cv_temp = CrossValidator(
        estimator=lr,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=k,
        seed=42
    )
    start = time.time()
    cv_temp_model = cv_temp.fit(train)
    elapsed = time.time() - start
    best_rmse = min(cv_temp_model.avgMetrics)
    print(f"K={k:2d} | Mejor RMSE: ${best_rmse:,.2f} | Tiempo: {elapsed:.1f}s")

# ¿Más folds siempre es mejor?
#    No. Más folds reduce varianza pero aumenta mucho el costo computacional.

# ============================================================
# PREGUNTAS DE REFLEXIÓN
# ============================================================

# 1. ¿Cuándo usarías K=3 vs K=10?
#    K=3 cuando el dataset es muy grande y quieres rapidez.
#    K=10 cuando el dataset es pequeño y necesitas mayor robustez.

# 2. ¿Cross-validation reemplaza la necesidad de un test set?
#    No. El test set sigue siendo necesario para evaluación final imparcial.

# 3. Si tu dataset tiene solo 100 registros, ¿qué K usarías?
#    K=5 o K=10 para aprovechar mejor los datos.

# 4. ¿Es posible hacer CV con time series?
#    Sí, pero no se usa K-Fold clásico.
#    Se usa validación temporal (rolling window), respetando el orden cronológico.

# ============================================================
# GUARDAR MEJOR MODELO
# ============================================================

model_path = "/opt/spark-data/processed/cv_best_model"
best_model.write().overwrite().save(model_path)

print(f"\nModelo guardado en: {model_path}")

print("\n" + "="*60)
print("RESUMEN VALIDACIÓN CRUZADA")
print("="*60)
print("  [X] Entendido el concepto de K-Fold")
print("  [X] Configurado ParamGrid con hiperparámetros")
print("  [X] Ejecutado CrossValidator")
print("  [X] Identificado el mejor modelo")
print("  [X] Comparado con entrenamiento simple")
print("="*60)

spark.stop()
