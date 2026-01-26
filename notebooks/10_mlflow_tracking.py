# %% [markdown]
# # Notebook 10: MLflow Tracking
#
# **Sección 16 - MLOps**: Registro de experimentos con MLflow
#
# **Objetivo**: Rastrear experimentos, métricas y modelos con MLflow
#
# ## Actividades:
# 1. Configurar MLflow tracking server
# 2. Registrar experimentos con hiperparámetros
# 3. Guardar métricas y artefactos
# 4. Comparar runs en MLflow UI

# %%
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
import mlflow
import mlflow.spark
import os

# %%
spark = SparkSession.builder \
    .appName("SECOP_MLflow") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

# %%
# Configurar MLflow
# El tracking server está en http://mlflow:5000
mlflow.set_tracking_uri("http://mlflow:5000")

# Crear o usar experimento existente
experiment_name = "/SECOP_Contratos_Prediccion"
mlflow.set_experiment(experiment_name)

print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Experimento: {experiment_name}")

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
# Evaluador
evaluator = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="rmse"
)

# %%
# =====================================
# EXPERIMENTO 1: Modelo sin regularización
# =====================================

print("\n=== EXPERIMENTO 1: Sin Regularización ===")

with mlflow.start_run(run_name="baseline_no_regularization"):
    # Hiperparámetros
    reg_param = 0.0
    elastic_param = 0.0
    max_iter = 100

    # Log de hiperparámetros
    mlflow.log_param("regParam", reg_param)
    mlflow.log_param("elasticNetParam", elastic_param)
    mlflow.log_param("maxIter", max_iter)
    mlflow.log_param("model_type", "LinearRegression")

    # Entrenar modelo
    lr = LinearRegression(
        featuresCol="features",
        labelCol="label",
        regParam=reg_param,
        elasticNetParam=elastic_param,
        maxIter=max_iter
    )

    model = lr.fit(train)

    # Predicciones
    predictions = model.transform(test)

    # Evaluar
    rmse = evaluator.evaluate(predictions)
    mae = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae").evaluate(predictions)
    r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2").evaluate(predictions)

    # Log de métricas
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # Guardar modelo
    mlflow.spark.log_model(model, "model")

    print(f"✓ RMSE: ${rmse:,.2f}")
    print(f"✓ MAE: ${mae:,.2f}")
    print(f"✓ R²: {r2:.4f}")

# %%
# =====================================
# EXPERIMENTO 2: Ridge (L2)
# =====================================

print("\n=== EXPERIMENTO 2: Ridge Regression (L2) ===")

with mlflow.start_run(run_name="ridge_l2_regression"):
    reg_param = 0.1
    elastic_param = 0.0  # L2 pure

    mlflow.log_param("regParam", reg_param)
    mlflow.log_param("elasticNetParam", elastic_param)
    mlflow.log_param("maxIter", 100)
    mlflow.log_param("model_type", "Ridge")

    lr = LinearRegression(
        featuresCol="features",
        labelCol="label",
        regParam=reg_param,
        elasticNetParam=elastic_param,
        maxIter=100
    )

    model = lr.fit(train)
    predictions = model.transform(test)

    rmse = evaluator.evaluate(predictions)
    mae = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae").evaluate(predictions)
    r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2").evaluate(predictions)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.spark.log_model(model, "model")

    print(f"✓ RMSE: ${rmse:,.2f}")

# %%
# =====================================
# EXPERIMENTO 3: Lasso (L1)
# =====================================

print("\n=== EXPERIMENTO 3: Lasso Regression (L1) ===")

with mlflow.start_run(run_name="lasso_l1_regression"):
    reg_param = 0.1
    elastic_param = 1.0  # L1 pure

    mlflow.log_param("regParam", reg_param)
    mlflow.log_param("elasticNetParam", elastic_param)
    mlflow.log_param("maxIter", 100)
    mlflow.log_param("model_type", "Lasso")

    lr = LinearRegression(
        featuresCol="features",
        labelCol="label",
        regParam=reg_param,
        elasticNetParam=elastic_param,
        maxIter=100
    )

    model = lr.fit(train)
    predictions = model.transform(test)

    rmse = evaluator.evaluate(predictions)
    mae = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae").evaluate(predictions)
    r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2").evaluate(predictions)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.spark.log_model(model, "model")

    print(f"✓ RMSE: ${rmse:,.2f}")

# %%
# =====================================
# EXPERIMENTO 4: ElasticNet
# =====================================

print("\n=== EXPERIMENTO 4: ElasticNet (L1 + L2) ===")

with mlflow.start_run(run_name="elasticnet_l1_l2"):
    reg_param = 0.1
    elastic_param = 0.5  # Mezcla 50/50

    mlflow.log_param("regParam", reg_param)
    mlflow.log_param("elasticNetParam", elastic_param)
    mlflow.log_param("maxIter", 100)
    mlflow.log_param("model_type", "ElasticNet")

    lr = LinearRegression(
        featuresCol="features",
        labelCol="label",
        regParam=reg_param,
        elasticNetParam=elastic_param,
        maxIter=100
    )

    model = lr.fit(train)
    predictions = model.transform(test)

    rmse = evaluator.evaluate(predictions)
    mae = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae").evaluate(predictions)
    r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2").evaluate(predictions)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.spark.log_model(model, "model")

    print(f"✓ RMSE: ${rmse:,.2f}")

# %%
print("\n" + "="*60)
print("EXPERIMENTOS COMPLETADOS")
print("="*60)
print("✓ 4 experimentos registrados en MLflow")
print(f"✓ Accede a MLflow UI: http://localhost:5000")
print(f"✓ Experimento: {experiment_name}")
print("="*60)

# %%
spark.stop()
