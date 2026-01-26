# %% [markdown]
# # Notebook 07: Regularización L1, L2 y ElasticNet
#
# **Sección 14**: Prevención de overfitting con regularización
#
# **Objetivo**: Comparar Ridge (L2), Lasso (L1) y ElasticNet
#
# ## Conceptos clave:
# - **Ridge (L2)**: regParam > 0, elasticNetParam = 0
# - **Lasso (L1)**: regParam > 0, elasticNetParam = 1
# - **ElasticNet**: regParam > 0, elasticNetParam ∈ (0, 1)

# %%
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
import pandas as pd

# %%
spark = SparkSession.builder \
    .appName("SECOP_Regularizacion") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

# %%
# Cargar datos
df = spark.read.parquet("/opt/spark-data/processed/secop_ml_ready.parquet")
df = df.withColumnRenamed("valor_del_contrato_num", "label") \
       .withColumnRenamed("features_pca", "features") \
       .filter(col("label").isNotNull())

train, test = df.randomSplit([0.7, 0.3], seed=42)

# %%
# Configurar evaluador
evaluator = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="rmse"
)

# %%
# Experimentar con diferentes valores de regularización
resultados = []

# Valores de lambda (regParam) a probar
reg_params = [0.0, 0.01, 0.1, 1.0, 10.0]

# Valores de elasticNet a probar
elastic_params = [0.0, 0.5, 1.0]  # 0=Ridge, 0.5=ElasticNet, 1=Lasso

print("Entrenando modelos con diferentes regularizaciones...\n")

for reg in reg_params:
    for elastic in elastic_params:
        # Configurar modelo
        lr = LinearRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=100,
            regParam=reg,
            elasticNetParam=elastic
        )

        # Entrenar
        model = lr.fit(train)

        # Evaluar
        predictions = model.transform(test)
        rmse = evaluator.evaluate(predictions)

        # Determinar tipo de regularización
        if reg == 0.0:
            reg_type = "Sin regularización"
        elif elastic == 0.0:
            reg_type = "Ridge (L2)"
        elif elastic == 1.0:
            reg_type = "Lasso (L1)"
        else:
            reg_type = "ElasticNet"

        resultados.append({
            "regParam": reg,
            "elasticNetParam": elastic,
            "tipo": reg_type,
            "rmse_test": rmse,
            "rmse_train": model.summary.rootMeanSquaredError,
            "r2_train": model.summary.r2
        })

        print(f"{reg_type:25s} | λ={reg:5.2f} | α={elastic:.1f} | RMSE Test: ${rmse:,.2f}")

# %%
# Convertir a DataFrame de pandas para análisis
df_resultados = pd.DataFrame(resultados)

print("\n" + "="*80)
print("COMPARACIÓN DE MODELOS REGULARIZADOS")
print("="*80)
print(df_resultados.to_string(index=False))
print("="*80)

# %%
# Encontrar mejor modelo
mejor_modelo = df_resultados.loc[df_resultados['rmse_test'].idxmin()]

print("\n=== MEJOR MODELO ===")
print(f"Tipo: {mejor_modelo['tipo']}")
print(f"regParam (λ): {mejor_modelo['regParam']}")
print(f"elasticNetParam (α): {mejor_modelo['elasticNetParam']}")
print(f"RMSE Test: ${mejor_modelo['rmse_test']:,.2f}")
print(f"RMSE Train: ${mejor_modelo['rmse_train']:,.2f}")
print(f"R² Train: {mejor_modelo['r2_train']:.4f}")

# %%
# Entrenar modelo final con mejores hiperparámetros
lr_final = LinearRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=mejor_modelo['regParam'],
    elasticNetParam=mejor_modelo['elasticNetParam']
)

modelo_final = lr_final.fit(train)

# %%
# Guardar mejor modelo
model_path = "/opt/spark-data/processed/regularized_model"
modelo_final.save(model_path)
print(f"\nMejor modelo guardado en: {model_path}")

# %%
# Guardar resultados del experimento
import json
with open("/opt/spark-data/processed/regularizacion_resultados.json", "w") as f:
    json.dump(df_resultados.to_dict('records'), f, indent=2)

print("Resultados guardados en: /opt/spark-data/processed/regularizacion_resultados.json")

# %%
spark.stop()
