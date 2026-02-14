# %% [markdown]
# # Notebook 07: Regularización L1, L2 y ElasticNet
#
# Sección 14: Prevención de overfitting con regularización
#
# Objetivo:
# Comparar Ridge (L2), Lasso (L1) y ElasticNet
#
# Conceptos clave:
# - Ridge (L2): regParam > 0, elasticNetParam = 0
# - Lasso (L1): regParam > 0, elasticNetParam = 1
# - ElasticNet: regParam > 0, elasticNetParam ∈ (0,1)

# %%
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, log1p
import pandas as pd
import numpy as np
import json

# %%
spark = SparkSession.builder \
    .appName("SECOP_Regularizacion") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

# %%
# Cargar datos
df = spark.read.parquet("/opt/spark-data/processed/secop_ml_ready.parquet")

from pyspark.sql.functions import log1p

df = df.withColumnRenamed("features_pca", "features") \
       .filter(col("label").isNotNull())

df = df.withColumn("label", log1p(col("label")))


train, test = df.randomSplit([0.7, 0.3], seed=42)

print(f"Train: {train.count():,}")
print(f"Test: {test.count():,}")

# ============================================================
# RETO 1: Entender la Regularización
# ============================================================

# Pregunta conceptual:
# ¿Por qué necesitamos regularización?
#
# Escenario:
# R² train = 0.95
# R² test = 0.45
#
# Opciones:
# A) Underfitting
# B) Overfitting
# C) Perfecto
# D) Más features
#
# Respuesta:
# B) El modelo está overfitting.
#
# Explicación:
# El modelo aprende demasiado bien el entrenamiento,
# pero generaliza mal en test.
# La regularización penaliza coeficientes grandes,
# simplifica el modelo y mejora generalización.

# ============================================================
# RETO 2: Configurar el Evaluador
# ============================================================

# Pregunta:
# ¿Qué métrica usarías para comparar modelos?
# - RMSE
# - MAE
# - R²
#
# Respuesta:
# Usamos RMSE porque penaliza más los errores grandes
# y es adecuada para regresión monetaria.

evaluator = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="rmse"
)

# ============================================================
# RETO 3: Experimento de Regularización
# ============================================================

# Parámetros sugeridos:
# regParam: [0.0, 0.01, 0.1, 1.0, 10.0]
# elasticNetParam: [0.0, 0.5, 1.0]

# 🔹 Ampliamos rango para ver efecto real de regularización
reg_params = [0.0, 0.01, 0.1, 1.0, 10.0]
elastic_params = [0.0, 0.5, 1.0]

print(f"Combinaciones totales: {len(reg_params) * len(elastic_params)}")

resultados = []

for reg in reg_params:
    for elastic in elastic_params:

        lr = LinearRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=200,
            regParam=reg,
            elasticNetParam=elastic
        )

        model = lr.fit(train)

        rmse_train = evaluator.evaluate(model.transform(train))
        rmse_test = evaluator.evaluate(model.transform(test))

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
            "rmse_train": rmse_train,
            "rmse_test": rmse_test,
            "gap": rmse_test - rmse_train
        })

        print(f"{reg_type:20s} | λ={reg:7.2f} | α={elastic:.1f} | "
              f"Train: {rmse_train:,.4f} | Test: {rmse_test:,.4f}")

# ============================================================
# RETO 4: Analizar Resultados
# ============================================================

df_resultados = pd.DataFrame(resultados)
df_resultados = df_resultados.sort_values("rmse_test")

print("\nResultados ordenados por RMSE Test:")
print(df_resultados.to_string(index=False))

mejor_modelo = df_resultados.iloc[0]

print("\nMejor modelo encontrado:")
print(mejor_modelo)

# Pregunta:
# ¿El mejor modelo es siempre el menor RMSE test?
#
# Respuesta:
# No necesariamente.
# También debemos considerar:
# - Gap train-test
# - Estabilidad
# - Interpretabilidad
# - Complejidad del modelo

# ============================================================
# RETO 5: Comparar Overfitting
# ============================================================

print("\nAnálisis de Overfitting:")
for _, row in df_resultados.iterrows():
    print(f"{row['tipo']:20s} | λ={row['regParam']:7.2f} | "
          f"Gap: {row['gap']:,.4f}")

# Preguntas:
# Si regParam=0.0 tiene train bajo y test alto → Overfitting ✔
# Si regParam=10.0 tiene ambos altos → Underfitting ✔
#
# ¿Qué regularización reduce más el overfitting?
#
# Respuesta:
# Generalmente Ridge o ElasticNet moderado reducen el gap
# sin incrementar demasiado el error total.
#
# ¿Hay trade-off?
# Sí. Más regularización reduce overfitting,
# pero demasiada produce underfitting.

# ============================================================
# RETO 6: Modelo Final
# ============================================================

best_reg = float(mejor_modelo["regParam"])
best_elastic = float(mejor_modelo["elasticNetParam"])

lr_final = LinearRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=200,
    regParam=best_reg,
    elasticNetParam=best_elastic
)

modelo_final = lr_final.fit(train)

rmse_final = evaluator.evaluate(modelo_final.transform(test))

print(f"\nRMSE final del mejor modelo: {rmse_final:,.4f}")

model_path = "/opt/spark-data/processed/regularized_model"

modelo_final.write().overwrite().save(model_path)

print(f"Modelo guardado en: {model_path}")

# ============================================================
# RETO BONUS: Efecto de Lambda
# ============================================================

print("\nEfecto de Lasso en coeficientes:")

for reg in [0.01, 0.1, 1.0, 10.0]:

    lr_lasso = LinearRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=200,
        regParam=reg,
        elasticNetParam=1.0
    )

    model_lasso = lr_lasso.fit(train)

    coefs = np.array(model_lasso.coefficients)
    zeros = np.sum(np.abs(coefs) < 1e-6)

    rmse = evaluator.evaluate(model_lasso.transform(test))

    print(f"λ={reg:7.2f} | Coeficientes en 0: {zeros}/{len(coefs)} | RMSE: {rmse:,.4f}")

# Pregunta:
# ¿Por qué Lasso elimina features y Ridge no?
#
# Respuesta:
# Lasso usa penalización L1 que permite coeficientes exactamente 0.
# Ridge usa L2 que solo reduce magnitudes pero nunca a 0.

# ============================================================
# Preguntas de Reflexión
# ============================================================

# 1. ¿Cuándo usar Ridge vs Lasso vs ElasticNet?
# - Ridge: Muchas variables correlacionadas.
# - Lasso: Cuando quieres selección automática.
# - ElasticNet: Cuando hay alta dimensionalidad y correlación.
#
# 2. ¿Qué pasa si regParam es demasiado grande?
# - El modelo se vuelve demasiado simple (Underfitting).
#
# 3. ¿Es posible que sin regularización sea el mejor?
# - Sí, si el dataset es grande y tiene poco ruido.
#
# 4. ¿Cómo elegir regParam en producción?
# - CrossValidation con múltiples folds.
# - GridSearch.
# - Evaluar estabilidad temporal.

# ============================================================
# Guardar resultados
# ============================================================

with open("/opt/spark-data/processed/regularizacion_resultados.json", "w") as f:
    json.dump(resultados, f, indent=2)

print("Resultados guardados.")

# %%
print("\n" + "="*60)
print("RESUMEN REGULARIZACIÓN")
print("="*60)
print("✔ Entendido diferencia entre L1, L2 y ElasticNet")
print("✔ Experimentado con múltiples combinaciones")
print("✔ Identificado el mejor modelo")
print("✔ Analizado overfitting vs underfitting")
print("✔ Guardado modelo final")
print("="*60)

spark.stop()
