# %% [markdown]
# # Notebook 11: Model Registry con MLflow
#
# **Sección 16 - MLOps**: Versionamiento y gestión del ciclo de vida
#
# **Objetivo**: Registrar modelos, crear versiones y promover a producción
#
# ## Actividades:
# 1. Registrar modelo en MLflow Model Registry
# 2. Crear versiones (v1, v2, etc.)
# 3. Transicionar entre stages: None → Staging → Production
# 4. Archivar modelos obsoletos

# %%
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient

# %%
spark = SparkSession.builder \
    .appName("SECOP_ModelRegistry") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

# %%
# Configurar MLflow
mlflow.set_tracking_uri("http://mlflow:5000")
client = MlflowClient()

print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

# %%
# Nombre del modelo en el registry
model_name = "secop_prediccion_contratos"

print(f"Nombre del modelo: {model_name}")

# %%
# Cargar datos
df = spark.read.parquet("/opt/spark-data/processed/secop_ml_ready.parquet")
df = df.withColumnRenamed("valor_del_contrato_num", "label") \
       .withColumnRenamed("features_pca", "features") \
       .filter(col("label").isNotNull())

train, test = df.randomSplit([0.8, 0.2], seed=42)

# %%
# =====================================
# PASO 1: Entrenar y registrar modelo (Versión 1)
# =====================================

print("\n=== PASO 1: Entrenar Modelo Versión 1 ===")

mlflow.set_experiment("/SECOP_Model_Registry")

with mlflow.start_run(run_name="model_v1_baseline") as run:
    # Entrenar modelo básico
    lr = LinearRegression(
        featuresCol="features",
        labelCol="label",
        regParam=0.0,
        maxIter=100
    )

    model_v1 = lr.fit(train)

    # Evaluar
    predictions = model_v1.transform(test)
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse_v1 = evaluator.evaluate(predictions)

    # Log
    mlflow.log_param("version", "1.0")
    mlflow.log_param("model_type", "baseline")
    mlflow.log_metric("rmse", rmse_v1)

    # Registrar modelo en MLflow
    mlflow.spark.log_model(
        spark_model=model_v1,
        artifact_path="model",
        registered_model_name=model_name
    )

    run_id_v1 = run.info.run_id

    print(f"✓ Modelo v1 registrado")
    print(f"  Run ID: {run_id_v1}")
    print(f"  RMSE: ${rmse_v1:,.2f}")

# %%
# =====================================
# PASO 2: Entrenar y registrar modelo mejorado (Versión 2)
# =====================================

print("\n=== PASO 2: Entrenar Modelo Versión 2 (Mejorado) ===")

with mlflow.start_run(run_name="model_v2_regularized") as run:
    # Modelo con regularización
    lr = LinearRegression(
        featuresCol="features",
        labelCol="label",
        regParam=0.1,
        elasticNetParam=0.5,
        maxIter=100
    )

    model_v2 = lr.fit(train)

    # Evaluar
    predictions = model_v2.transform(test)
    rmse_v2 = evaluator.evaluate(predictions)

    # Log
    mlflow.log_param("version", "2.0")
    mlflow.log_param("model_type", "regularized")
    mlflow.log_param("regParam", 0.1)
    mlflow.log_param("elasticNetParam", 0.5)
    mlflow.log_metric("rmse", rmse_v2)

    # Registrar modelo
    mlflow.spark.log_model(
        spark_model=model_v2,
        artifact_path="model",
        registered_model_name=model_name
    )

    run_id_v2 = run.info.run_id

    print(f"✓ Modelo v2 registrado")
    print(f"  Run ID: {run_id_v2}")
    print(f"  RMSE: ${rmse_v2:,.2f}")

# %%
# =====================================
# PASO 3: Gestión de versiones en Model Registry
# =====================================

print("\n=== PASO 3: Gestión de Versiones ===")

# Listar todas las versiones del modelo
model_versions = client.search_model_versions(f"name='{model_name}'")

print(f"\nVersiones registradas del modelo '{model_name}':")
for mv in model_versions:
    print(f"  - Versión {mv.version}: Stage={mv.current_stage}, Run ID={mv.run_id}")

# %%
# =====================================
# PASO 4: Transicionar versiones entre stages
# =====================================

print("\n=== PASO 4: Transición de Stages ===")

# Promover versión 1 a Staging
client.transition_model_version_stage(
    name=model_name,
    version=1,
    stage="Staging"
)
print("✓ Versión 1 → Staging")

# Si v2 es mejor, promover a Production
if rmse_v2 < rmse_v1:
    client.transition_model_version_stage(
        name=model_name,
        version=2,
        stage="Production"
    )
    print("✓ Versión 2 → Production (mejor desempeño)")

    # Archivar versión 1
    client.transition_model_version_stage(
        name=model_name,
        version=1,
        stage="Archived"
    )
    print("✓ Versión 1 → Archived")
else:
    # Si v1 sigue siendo mejor
    client.transition_model_version_stage(
        name=model_name,
        version=1,
        stage="Production"
    )
    print("✓ Versión 1 → Production")

# %%
# =====================================
# PASO 5: Añadir metadata a versiones
# =====================================

print("\n=== PASO 5: Añadir Metadata ===")

# Añadir descripción a la versión en producción
best_version = 2 if rmse_v2 < rmse_v1 else 1

client.update_model_version(
    name=model_name,
    version=best_version,
    description=f"Mejor modelo para predicción de contratos SECOP. RMSE: ${min(rmse_v1, rmse_v2):,.2f}"
)

print(f"✓ Metadata añadida a versión {best_version}")

# %%
# =====================================
# PASO 6: Cargar modelo desde Registry
# =====================================

print("\n=== PASO 6: Cargar Modelo desde Registry ===")

# Cargar modelo en producción
model_uri = f"models:/{model_name}/Production"
loaded_model = mlflow.spark.load_model(model_uri)

print(f"✓ Modelo cargado desde: {model_uri}")
print(f"  Tipo: {type(loaded_model)}")

# Verificar que funciona
test_predictions = loaded_model.transform(test)
test_rmse = evaluator.evaluate(test_predictions)

print(f"  RMSE en test: ${test_rmse:,.2f}")

# %%
# Resumen final
print("\n" + "="*60)
print("RESUMEN MODEL REGISTRY")
print("="*60)
print(f"✓ Modelo registrado: {model_name}")
print(f"✓ Versiones creadas: 2")
print(f"✓ Versión en producción: {best_version}")
print(f"✓ RMSE producción: ${min(rmse_v1, rmse_v2):,.2f}")
print(f"✓ Accede a Model Registry: http://localhost:5000/#/models")
print("="*60)

# %%
spark.stop()
