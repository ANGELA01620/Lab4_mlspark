# %% [markdown]
# # Notebook 06: Regresión Logística
#
# **Sección 14**: Clasificación binaria - Predicción de riesgo
#
# **Objetivo**: Clasificar contratos con riesgo de incumplimiento
#
# ## Actividades:
# 1. Crear variable objetivo binaria
# 2. Entrenar LogisticRegression
# 3. Evaluar con AUC-ROC, Precision, Recall
# 4. Analizar matriz de confusión

# %%
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col, when

# %%
spark = SparkSession.builder \
    .appName("SECOP_RegresionLogistica") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

# %%
# Cargar datos
df = spark.read.parquet("/opt/spark-data/processed/secop_features.parquet")

# %%
# Crear variable objetivo binaria: riesgo_incumplimiento
# Ejemplo: Contratos con valor muy alto o plazo muy corto tienen mayor riesgo
# NOTA: Esta es una simplificación. En producción, usar datos reales de incumplimiento

df = df.withColumn(
    "riesgo_incumplimiento",
    when(
        (col("valor_del_contrato_num") > 1000000000) |  # Contratos > 1B
        (col("plazo_de_ejec_del_contrato") < 30),       # Plazo < 30 días
        1
    ).otherwise(0)
)

# Renombrar features
df = df.withColumnRenamed("features_raw", "features") \
       .withColumnRenamed("riesgo_incumplimiento", "label")

# Filtrar nulos
df = df.filter(col("label").isNotNull() & col("features").isNotNull())

print(f"Total registros: {df.count():,}")
print(f"Clase 0 (Sin riesgo): {df.filter(col('label') == 0).count():,}")
print(f"Clase 1 (Con riesgo): {df.filter(col('label') == 1).count():,}")

# %%
# Dividir en train/test
train, test = df.randomSplit([0.7, 0.3], seed=42)

# %%
# Crear modelo de Regresión Logística
lr_classifier = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100
)

print("Entrenando clasificador...")
lr_model = lr_classifier.fit(train)
print("✓ Modelo entrenado")

# %%
# Predicciones
predictions = lr_model.transform(test)
predictions.select("label", "prediction", "probability").show(10, truncate=False)

# %%
# Evaluación: AUC-ROC
evaluator_auc = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

auc = evaluator_auc.evaluate(predictions)

# Evaluación: Precision, Recall, F1
evaluator_multi = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

accuracy = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "accuracy"})
precision = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "weightedPrecision"})
recall = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "weightedRecall"})
f1 = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "f1"})

print("\n" + "="*60)
print("MÉTRICAS DE CLASIFICACIÓN")
print("="*60)
print(f"AUC-ROC:   {auc:.4f}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("="*60)

# %%
# Guardar modelo
model_path = "/opt/spark-data/processed/logistic_regression_model"
lr_model.save(model_path)
print(f"\nModelo guardado en: {model_path}")

# %%
spark.stop()
