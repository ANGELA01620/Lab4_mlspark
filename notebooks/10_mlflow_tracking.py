# %% [markdown]
# # Notebook 10: MLflow Tracking
#
# **Sección 16 - MLOps**: Registro de experimentos con MLflow
#
# **Objetivo**: Rastrear experimentos, métricas y modelos con MLflow
#
# ## Conceptos clave:
# - **Experiment**: Agrupación lógica de runs (un proyecto)
# - **Run**: Una ejecución individual (un modelo entrenado)
# - **Parameters**: Hiperparámetros registrados (regParam, maxIter, etc.)
# - **Metrics**: Métricas de rendimiento (RMSE, R², etc.)
# - **Artifacts**: Archivos guardados (modelos, gráficos, etc.)
#
# ## Actividades:
# 1. Configurar MLflow tracking server
# 2. Registrar experimentos con hiperparámetros
# 3. Guardar métricas y artefactos
# 4. Comparar runs en MLflow UI

# ============================================================
# NOTEBOOK 10: MLFLOW TRACKING
# ============================================================

from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, log1p
import mlflow
import mlflow.spark
import time

# ============================================================
# INICIAR SPARK
# ============================================================

spark = SparkSession.builder \
    .appName("SECOP_MLflow") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

print("Spark inicializado correctamente")

# ============================================================
# RETO 1: CONFIGURAR MLFLOW
# ============================================================

print("\n" + "="*60)
print("RETO 1: CONFIGURAR MLFLOW")
print("="*60)

# Configurar tracking server
mlflow.set_tracking_uri("http://mlflow:5000")

# Crear experimento
experiment_name = "/SECOP_Contratos_Prediccion"
mlflow.set_experiment(experiment_name)

print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Experimento: {experiment_name}")

print("\n¿Por qué es importante un tracking server centralizado?")

print("""
Para nosotras, un tracking server centralizado no es solo
una herramienta adicional, sino parte fundamental de una
arquitectura seria de MLOps.

Primero, nos garantiza persistencia.
Cuando entrenamos modelos, no queremos que las métricas,
parámetros o artefactos dependan de nuestra máquina local.
Si se reinicia el entorno o se cae un contenedor,
la información sigue disponible porque está almacenada
en un servidor centralizado.

Segundo, nos permite trabajar realmente en equipo.
Ambas podemos acceder a los mismos experimentos,
ver qué probó cada una, comparar resultados
y evitar repetir trabajo.
Además, facilita revisiones técnicas y auditorías,
porque todo queda en un mismo lugar.

Tercero, nos da trazabilidad completa.
Cada experimento queda registrado con su contexto:
qué datos usamos, qué hiperparámetros configuramos,
qué métricas obtuvimos y qué artefactos generamos.
Eso nos permite justificar decisiones y mantener
un historial claro del ciclo de vida del modelo.

Cuarto, mejora la reproducibilidad.
Si en el futuro necesitamos volver a una versión anterior
o explicar por qué elegimos cierto modelo,
tenemos toda la información organizada y accesible.

Y finalmente, nos facilita la comparación.
Desde la interfaz podemos analizar modelos lado a lado,
sin depender de archivos sueltos ni scripts manuales.

En contraste, guardar métricas en CSV locales
es frágil y poco escalable.
Se pueden perder archivos,
no hay colaboración estructurada
y comparar resultados se vuelve manual y propenso a errores.

Por eso, para nosotras,
un tracking server centralizado es la base
para trabajar de manera profesional,
ordenada y reproducible.
""")

# ============================================================
# CARGAR DATOS
# ============================================================

df = spark.read.parquet("/opt/spark-data/processed/secop_ml_ready.parquet")

df = df.withColumnRenamed("features_pca", "features") \
       .filter(col("label").isNotNull())

df = df.withColumn("label", log1p(col("label")))

train, test = df.randomSplit([0.8, 0.2], seed=42)

print(f"\nTrain: {train.count():,}")
print(f"Test: {test.count():,}")

# ============================================================
# CONFIGURAR EVALUADOR
# ============================================================

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

# ============================================================
# RETO 2: REGISTRAR EXPERIMENTO BASELINE
# ============================================================

print("\n" + "="*60)
print("RETO 2: REGISTRAR EXPERIMENTO BASELINE")
print("="*60)

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
    mlflow.log_param("dataset_size_train", train.count())
    mlflow.log_param("dataset_size_test", test.count())
    
    # Entrenar modelo
    lr = LinearRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=max_iter,
        regParam=reg_param,
        elasticNetParam=elastic_param
    )
    
    start_time = time.time()
    model = lr.fit(train)
    training_time = time.time() - start_time
    
    # Evaluar
    predictions = model.transform(test)
    
    rmse = evaluator_rmse.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)
    
    # Log de métricas
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("training_time_seconds", training_time)
    
    # Guardar modelo
    mlflow.spark.log_model(model, "model")
    
    print(f"\nMétricas Baseline:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Tiempo de entrenamiento: {training_time:.2f}s")

# ============================================================
# RETO 3: REGISTRAR MÚLTIPLES EXPERIMENTOS
# ============================================================

print("\n" + "="*60)
print("RETO 3: REGISTRAR MÚLTIPLES EXPERIMENTOS")
print("="*60)

print("\n¿Por qué registrar múltiples métricas y no solo RMSE?")

print("""
Para nosotras, evaluar un modelo con una sola métrica es
quedarnos con una visión parcial del rendimiento.
Cada métrica captura un comportamiento distinto del error,
y necesitamos esa perspectiva multidimensional para tomar
decisiones técnicas sólidas.

RMSE (Root Mean Squared Error):
- Penaliza fuertemente los errores grandes (por el término al cuadrado)
- Es altamente sensible a outliers
- Refleja riesgo cuando los errores extremos son críticos
- Útil cuando una mala predicción grande tiene alto impacto económico

MAE (Mean Absolute Error):
- Promedio simple de errores absolutos
- Más robusta frente a valores atípicos
- Interpretación directa: “en promedio nos equivocamos por X”
- Muy útil para comunicar resultados a stakeholders no técnicos

R² (Coeficiente de Determinación):
- Métrica adimensional (independiente de la escala)
- Indica qué proporción de la varianza explica el modelo
- Permite comparar modelos incluso con transformaciones distintas
- Toma valores entre 0 y 1 (más cercano a 1 implica mejor ajuste)

¿Por qué no basta con RMSE?
Porque podríamos tener:
- Un RMSE bajo pero un R² mediocre (modelo inestable)
- Un R² alto pero errores grandes en casos críticos
- Un MAE razonable pero alta sensibilidad a outliers

CONCLUSIÓN:
Registrar múltiples métricas nos permite:

- Analizar el modelo desde distintas perspectivas estadísticas
- Entender trade-offs entre estabilidad y sensibilidad a errores grandes
- Detectar comportamientos anómalos
- Alinear la evaluación técnica con el impacto real en el negocio

En MLOps, medir bien es tan importante como modelar bien.
Y medir bien implica observar el rendimiento desde varios ángulos,
no desde uno solo.
""")


experiments = [
    {"name": "ridge_l2", "reg": 0.1, "elastic": 0.0, "type": "Ridge"},
    {"name": "lasso_l1", "reg": 0.1, "elastic": 1.0, "type": "Lasso"},
    {"name": "elasticnet", "reg": 0.1, "elastic": 0.5, "type": "ElasticNet"},
]

print("\nEntrenando modelos con diferentes tipos de regularización...")

resultados_comparacion = []

for exp in experiments:
    with mlflow.start_run(run_name=exp["name"]):
        
        # Log parámetros
        mlflow.log_param("regParam", exp["reg"])
        mlflow.log_param("elasticNetParam", exp["elastic"])
        mlflow.log_param("maxIter", 100)
        mlflow.log_param("model_type", exp["type"])
        mlflow.log_param("dataset_size_train", train.count())
        mlflow.log_param("dataset_size_test", test.count())
        
        # Entrenar modelo
        lr = LinearRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=100,
            regParam=exp["reg"],
            elasticNetParam=exp["elastic"]
        )
        
        start_time = time.time()
        model = lr.fit(train)
        training_time = time.time() - start_time
        
        # Evaluar
        predictions = model.transform(test)
        
        rmse = evaluator_rmse.evaluate(predictions)
        mae = evaluator_mae.evaluate(predictions)
        r2 = evaluator_r2.evaluate(predictions)
        
        # Log métricas
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("training_time_seconds", training_time)
        
        # Guardar modelo
        mlflow.spark.log_model(model, "model")
        
        print(f"\n{exp['type']}:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Tiempo: {training_time:.2f}s")
        
        resultados_comparacion.append({
            "tipo": exp["type"],
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        })

# Análisis comparativo
print("\n" + "="*60)
print("ANÁLISIS COMPARATIVO DE MODELOS")
print("="*60)

mejor_rmse = min(resultados_comparacion, key=lambda x: x["rmse"])
mejor_r2 = max(resultados_comparacion, key=lambda x: x["r2"])

print(f"\nMejor RMSE: {mejor_rmse['tipo']} ({mejor_rmse['rmse']:.4f})")
print(f"Mejor R²: {mejor_r2['tipo']} ({mejor_r2['r2']:.4f})")

if mejor_rmse['tipo'] == mejor_r2['tipo']:
    print(f"\nCONCLUSIÓN: {mejor_rmse['tipo']} es el mejor modelo consistentemente")
else:
    print(f"\nOBSERVACIÓN: Hay trade-off entre métricas")
    print(f"  - {mejor_rmse['tipo']} minimiza error (RMSE)")
    print(f"  - {mejor_r2['tipo']} maximiza varianza explicada (R²)")

# ============================================================
# RETO 4: EXPLORAR MLFLOW UI
# ============================================================

print("\n" + "="*60)
print("RETO 4: EXPLORAR MLFLOW UI")
print("="*60)

print("""
INSTRUCCIONES PASO A PASO PARA ANALIZAR LOS EXPERIMENTOS EN MLFLOW

1. Abrimos el navegador y vamos a:
   http://localhost:5000

Ahí accedemos a la interfaz del Tracking Server.

2. En el panel izquierdo vamos a ver la lista de experimentos.
   Buscamos y seleccionamos:

   /SECOP_Contratos_Prediccion

Ese experimento contiene todos los modelos
que hemos entrenado para este caso.

3. Dentro del experimento veremos varios runs.
   Cada run representa un entrenamiento distinto,
   con su propia configuración y resultados.

   En la tabla podemos observar:
   - Métricas (rmse, mae, r2, etc.)
   - Parámetros (hiperparámetros utilizados)
   - Fecha y duración del entrenamiento

4. Si queremos comparar modelos,
   marcamos las casillas de los runs que nos interesen
   y hacemos clic en "Compare".

   Esto nos permite ver:
   - Gráficos comparativos
   - Métricas lado a lado
   - Diferencias en parámetros

   Es una forma visual y rápida de tomar decisiones.

5. Para identificar el mejor modelo según RMSE,
   hacemos clic en la columna "rmse"
   y ordenamos de menor a mayor.

   El valor más bajo indicará el modelo
   con menor error cuadrático promedio.

6. Si queremos profundizar en un modelo específico,
   hacemos clic sobre el nombre del run.

   Allí podemos revisar:
   - Parámetros completos
   - Todas las métricas registradas
   - Artefactos generados
   - Código asociado
   - Y descargar el modelo desde la sección "Artifacts"

Este análisis visual nos permite entender
qué modelo funciona mejor,
por qué funciona mejor
y con qué configuración fue entrenado.
""")

print("\nPREGUNTAS DE ANÁLISIS:")


print("\n1. ¿Qué modelo tiene el mejor RMSE?")

print("""
Para responder esta pregunta, nosotras primero ordenaríamos
los runs por la columna RMSE en orden ascendente,
porque menor RMSE implica menor error cuadrático promedio.

Luego identificaríamos el run con el valor más bajo
y revisaríamos también su R² para confirmar
que el buen desempeño no sea aislado.

No nos quedaríamos solo con el número más bajo.
Validaríamos que:
- El R² también sea competitivo.
- No haya señales de inestabilidad.
- El modelo no tenga una configuración extrema.

En muchos casos, modelos como Ridge o ElasticNet
logran un mejor balance entre sesgo y varianza.

Lasso podría degradar si la regularización es demasiado alta,
porque puede eliminar demasiadas features.

Y un baseline sin regularización podría mostrar
buen desempeño en entrenamiento pero peor generalización.
""")

print("\n2. ¿Hay correlación entre regularización y rendimiento?")

print("""
Aquí analizaríamos cómo cambia el rendimiento
a medida que aumenta el parámetro de regularización.

Compararíamos:
- El modelo baseline (regParam = 0.0)
- Modelos con regularización moderada
- Modelos con regularización alta

Nos fijaríamos especialmente en:
- Diferencias entre métricas de train y test
- Estabilidad del R²
- Comportamiento del RMSE

El patrón esperado suele ser:

- Sin regularización puede haber overfitting.
- Regularización moderada mejora la generalización.
- Regularización excesiva genera underfitting.

Lo importante no es solo mejorar una métrica,
sino encontrar el punto donde el modelo generaliza mejor.
""")

print("\n3. ¿Cómo compartiríamos estos resultados con el equipo?")

print("""
Nosotras combinaríamos varias estrategias.

Primero, usaríamos la función de comparación en MLflow:
- Seleccionamos los runs.
- Hacemos clic en Compare.
- Exportamos la tabla como CSV si necesitamos compartirla formalmente.

También podríamos compartir directamente la URL del experimento,
si el equipo tiene acceso al tracking server.

Para presentaciones ejecutivas,
haríamos un pequeño reporte con:
- Métricas clave.
- Gráficos comparativos.
- Conclusión y recomendación.

En reuniones técnicas, incluso podemos usar la UI en vivo
para mostrar la comparación interactiva
y discutir trade-offs en tiempo real.

La mejor práctica es combinar:
reporte escrito + acceso al tracking server + documentación
de la decisión final en el Model Registry.
""")

print("\n" + "="*60)
print("EJERCICIO PRÁCTICO")
print("="*60)

print("""
Ahora vamos a analizarlo directamente en la UI:

1. Identificamos el modelo con menor RMSE.
2. Revisamos sus hiperparámetros y los comparamos con los demás.
3. Descargamos el artefacto model_report.txt del run correspondiente.
4. Verificamos que el modelo esté correctamente almacenado en Artifacts.

Finalmente reflexionamos:

- ¿Qué modelo elegiríamos para producción y por qué?
- ¿Existe algún trade-off entre RMSE, MAE y R²?
- ¿El comportamiento observado coincide con lo que esperamos
  sobre regularización y generalización?

La decisión debe estar basada en evidencia,
no solo en intuición.
""")

# ============================================================
# RETO 5: AGREGAR ARTEFACTOS PERSONALIZADOS
# ============================================================

print("\n" + "="*60)
print("RETO 5: AGREGAR ARTEFACTOS PERSONALIZADOS")
print("="*60)

with mlflow.start_run(run_name="model_with_artifacts"):
    
    # Entrenar modelo con mejores hiperparámetros
    best_reg = 0.1
    best_elastic = 0.0
    
    mlflow.log_param("regParam", best_reg)
    mlflow.log_param("elasticNetParam", best_elastic)
    mlflow.log_param("maxIter", 100)
    mlflow.log_param("model_type", "Ridge_with_artifacts")
    
    lr = LinearRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=100,
        regParam=best_reg,
        elasticNetParam=best_elastic
    )
    
    model = lr.fit(train)
    predictions = model.transform(test)
    
    rmse = evaluator_rmse.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)
    
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    
    # Crear reporte de texto
    report = f"""
REPORTE DE MODELO
==================

DATASET:
- Train: {train.count():,} registros
- Test: {test.count():,} registros

HIPERPARÁMETROS:
- regParam: {best_reg}
- elasticNetParam: {best_elastic}
- maxIter: 100
- Tipo: Ridge (L2)

MÉTRICAS DE RENDIMIENTO:
- RMSE: {rmse:.4f}
- MAE: {mae:.4f}
- R²: {r2:.4f}

INTERPRETACIÓN:
- El modelo explica {r2*100:.2f}% de la varianza
- Error promedio absoluto: {mae:.4f} (escala log)
- Regularización L2 previene overfitting

CONCLUSIÓN:
Modelo estable con generalización adecuada.
Listo para producción tras validación adicional.
"""
    
    mlflow.log_text(report, "model_report.txt")
    
    print("Reporte guardado como artefacto")
    print(report)
    
    # Guardar modelo
    mlflow.spark.log_model(model, "model")

# ============================================================
# PREGUNTAS DE REFLEXIÓN
# ============================================================

print("\n" + "="*60)
print("PREGUNTAS DE REFLEXIÓN - ANÁLISIS DETALLADO")
print("="*60)
print("""
1. ¿Qué ventajas tiene MLflow sobre guardar métricas en archivos CSV?

Si lo pensamos desde nuestra experiencia trabajando en equipo,
guardar métricas en CSV funciona al inicio, pero escala muy mal.

Con CSV:
- Los archivos quedan regados en carpetas.
- Es fácil sobrescribir resultados sin darnos cuenta.
- No hay conexión directa entre métricas, modelo y código.
- Comparar 20 o 30 experimentos se vuelve manual y lento.
- No existe trazabilidad real.

Con MLflow, en cambio:
- Todo queda centralizado en un tracking server.
- Cada run tiene un ID único.
- Se guardan parámetros, métricas, artefactos y el modelo.
- Podemos comparar experimentos visualmente en la UI.
- Tenemos histórico completo y reproducible.

En la práctica, si entrenamos 50 modelos en tres meses,
con CSV nos tocaría buscarlos manualmente.
Con MLflow simplemente filtramos por métrica y listo.

La diferencia es que MLflow no solo almacena números,
gestiona todo el ciclo de vida del modelo.
""")


print("""
2. ¿Cómo implementaríamos MLflow en un proyecto de equipo?

Nosotras lo haríamos por fases para que sea ordenado y sostenible.

FASE 1 – Infraestructura:
- Levantar un tracking server central.
- Configurar base de datos para metadata.
- Configurar almacenamiento de artefactos (S3, Blob, etc.).
- Definir acceso y permisos.

FASE 2 – Convenciones:
- Estandarizar nombres de experimentos.
- Definir qué parámetros y métricas son obligatorias.
- Crear una plantilla base de entrenamiento con MLflow integrado.
- Documentar buenas prácticas.

FASE 3 – Capacitación:
- Hacer un workshop práctico.
- Mostrar cómo comparar modelos en la UI.
- Crear una guía interna.
- Nombrar una persona referente del tema.

FASE 4 – Automatización:
- Integrar MLflow en pipelines CI/CD.
- Validar métricas automáticamente antes de promover modelos.
- Agregar alertas si hay degradación.

La clave no es solo instalar MLflow,
sino convertirlo en parte del proceso estándar del equipo.
""")


print("""
3. ¿Qué artefactos adicionales guardaríamos además del modelo?

El modelo por sí solo no es suficiente.

Nosotras guardaríamos:

- Pipeline completo de preprocesamiento.
- Estadísticas del dataset de entrenamiento.
- Lista exacta de features usadas.
- requirements.txt con versiones fijas.
- Código de entrenamiento.
- Reportes visuales (residuos, feature importance, etc.).
- Una model card con contexto y limitaciones.
- Muestras pequeñas del dataset para trazabilidad.

La idea es que cualquier persona del equipo,
incluso meses después,
pueda entender qué se hizo y reproducirlo.

Un modelo sin contexto es muy difícil de mantener.
""")


print("""
4. ¿Cómo automatizaríamos el registro de experimentos?

No dejaríamos el logging manual.

Algunas estrategias que usaríamos:

A) Función wrapper:
Crear una función estándar que:
- Inicie el run.
- Loggee parámetros.
- Loggee métricas.
- Guarde el modelo automáticamente.

B) Autolog:
Usar mlflow.autolog() cuando el framework lo permita,
para reducir fricción y evitar olvidos.

C) Scripts parametrizados:
Entrenar modelos desde línea de comandos
y registrar automáticamente cada ejecución.

D) Integración en CI/CD:
Que cada push relevante ejecute entrenamiento,
registre el modelo y compare contra baseline.

E) Orquestación con Airflow:
Programar entrenamientos recurrentes
y registrar todo automáticamente.

El objetivo es reducir errores humanos,
asegurar consistencia
y hacer que el registro sea parte natural del flujo de trabajo,
no un paso opcional.
""")


# ============================================================
# RESUMEN FINAL
# ============================================================

print("\n" + "="*60)
print("RESUMEN MLFLOW TRACKING")
print("="*60)
print("Verifica que hayas completado:")
print("  Configurado MLflow tracking server")
print("  Registrado experimento baseline")
print("  Registrado 3 experimentos adicionales (Ridge, Lasso, ElasticNet)")
print("  Guardado múltiples métricas (RMSE, MAE, R²)")
print("  Agregado artefactos personalizados (reporte)")
print("  Guardado modelos entrenados")
print(f"   Accede a MLflow UI: http://localhost:5000")
print("="*60)
print("\nPróximo paso: Model Registry (notebook 11)")

spark.stop()
print("Proceso finalizado correctamente")