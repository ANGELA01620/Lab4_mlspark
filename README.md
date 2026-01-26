# Proyecto Final: Sistema Predictivo de Contratos Públicos con Spark ML

**Módulo 4: Machine Learning Escalable con Spark**
**Diplomado en Gestión de Datos - Universidad Santo Tomás**

---

## Descripción del Proyecto

Este proyecto integra los conceptos del **Módulo 4 (MLSpark)** mediante la construcción de un sistema completo de Machine Learning distribuido que predice y analiza contratos públicos usando datos reales de **Colombia Compra Eficiente (SECOP II)**.

El proyecto evalúa las cuatro secciones del módulo:
- **Sección 13**: Spark ML - Pipelines y Feature Engineering distribuido
- **Sección 14**: Modelos de Regresión y Regularización
- **Sección 15**: Optimización de Hiperparámetros y Validación Cruzada
- **Sección 16**: MLOps - Tracking, Registro y Despliegue con MLflow

---

## Objetivos de Aprendizaje

Al completar este proyecto, habrás demostrado competencias en:

1. **Machine Learning Distribuido**: Escalar algoritmos de ML a datasets masivos usando Spark MLlib
2. **Pipelines de ML**: Construir flujos reproducibles con Transformers y Estimators
3. **Feature Engineering**: Manipular variables categóricas y numéricas en entornos distribuidos
4. **Modelos Predictivos**: Implementar regresión lineal y logística con regularización
5. **Optimización**: Aplicar búsqueda en rejilla y validación cruzada para tuning
6. **MLOps**: Gestionar el ciclo de vida completo del modelo con MLflow

---

## Dataset: SECOP II - Contratos Electrónicos

**Fuente**: [Datos Abiertos Colombia - SECOP II](https://www.datos.gov.co/Gastos-Gubernamentales/SECOP-II-Contratos-Electr-nicos/jbjy-vk9h/data)

**Descripción**: Contratos electrónicos del Sistema Electrónico para la Contratación Pública de Colombia, actualizado diariamente por la Agencia Nacional de Contratación Pública.

**Campos clave**:
- `Referencia del Contrato`: Identificador único
- `Precio Base`: Valor del contrato (variable objetivo para regresión)
- `Departamento`: Ubicación geográfica
- `Tipo de Contrato`: Categoría del contrato
- `Fecha de Firma`: Temporalidad
- `Plazo de Ejecución`: Duración en días
- `Proveedor Adjudicado`: Entidad contratista
- `Estado del Contrato`: Activo, Liquidado, etc.

---

## Casos de Uso de ML

### 1. Regresión Lineal: Predicción del Valor del Contrato
**Objetivo**: Predecir el `Precio Base` de un contrato en función de características como departamento, tipo de contrato, duración y entidad.

**Técnicas aplicadas**:
- VectorAssembler para preparación de features
- Escalado con StandardScaler
- Regularización Ridge (L2) y Lasso (L1)
- ElasticNet para combinación L1+L2

### 2. Clasificación Binaria: Predicción de Riesgo de Incumplimiento
**Objetivo**: Clasificar contratos con alta probabilidad de incumplimiento basado en patrones históricos.

**Variables**:
- Duración del contrato
- Monto
- Historial del proveedor
- Departamento

**Modelo**: Regresión Logística con regularización

### 3. Optimización de Hiperparámetros
**Objetivo**: Encontrar la mejor combinación de hiperparámetros usando:
- Grid Search (búsqueda exhaustiva)
- Random Search (muestreo aleatorio)
- Cross-Validation (validación k-fold)

### 4. MLOps: Ciclo de Vida Completo
**Objetivo**: Implementar prácticas profesionales de MLOps:
- Tracking de experimentos con MLflow
- Versionamiento de modelos
- Registro en Model Registry
- Despliegue para inferencia batch

---

## Estructura del Proyecto

```
proyecto_final_mlspark/
├── README.md                           # Este archivo
├── docker-compose.yml                  # Orquestación de servicios
├── Dockerfile                          # Imagen personalizada de Spark
├── requirements.txt                    # Dependencias Python
├── .gitignore                          # Archivos ignorados en Git
│
├── data/
│   ├── raw/                            # Datos descargados de SECOP II
│   └── processed/                      # Datos procesados por Spark
│
├── notebooks/
│   ├── 01_ingesta_datos.py             # Descarga y carga inicial
│   ├── 02_exploracion_eda.py           # Análisis exploratorio
│   │
│   ├── 03_feature_engineering.py       # Pipelines y VectorAssembler
│   ├── 04_transformaciones.py          # Scalers, PCA, One-Hot Encoding
│   │
│   ├── 05_regresion_lineal.py          # Predicción de precios base
│   ├── 06_regresion_logistica.py       # Clasificación de riesgos
│   ├── 07_regularizacion.py            # Ridge, Lasso, ElasticNet
│   │
│   ├── 08_validacion_cruzada.py        # K-Fold Cross-Validation
│   ├── 09_optimizacion_hiperparametros.py  # Grid/Random Search
│   │
│   ├── 10_mlflow_tracking.py           # Experimentos y métricas
│   ├── 11_model_registry.py            # Versionamiento de modelos
│   └── 12_inferencia_produccion.py     # Predicciones batch
│
├── src/
│   ├── __init__.py                     # Módulo Python
│   ├── data_loader.py                  # Utilidades de descarga
│   ├── feature_utils.py                # Funciones de features
│   └── model_trainer.py                # Entrenamiento modular
│
└── mlruns/                             # Tracking de MLflow
```

---

## Prerrequisitos

- **Docker Desktop** instalado y corriendo
- **Git** (para clonar el repositorio)
- **Al menos 8 GB de RAM** disponible para Docker
- **10 GB de espacio en disco** para datos y logs

---

## Guía de Implementación

### Fase 1: Infraestructura (30 min)

#### 1.1 Clonar el Repositorio

```bash
git clone <url-del-repositorio>
cd lab4_mlspark
```

#### 1.2 Levantar el Cluster Spark + MLflow

```bash
docker-compose up --build -d
```

#### 1.3 Verificar Servicios

Accede a las siguientes URLs en tu navegador:

- **Jupyter Lab**: [http://localhost:8888](http://localhost:8888) (contraseña: `spark`)
- **Spark Master UI**: [http://localhost:8080](http://localhost:8080)
- **MLflow UI**: [http://localhost:5000](http://localhost:5000)

---

### Fase 2: Ingesta y Exploración (1 hora)

#### Notebook 01: Ingesta de Datos

**Objetivos**:
- Descargar datos desde la API de Datos Abiertos Colombia
- Cargar CSV en Spark DataFrame
- Guardar en formato Parquet optimizado

**Archivo**: [notebooks/01_ingesta_datos.py](notebooks/01_ingesta_datos.py)

**Actividades**:
1. Configurar SparkSession con Delta Lake
2. Descargar dataset SECOP II usando API Socrata
3. Explorar esquema inicial con `printSchema()`
4. Persistir en `data/raw/` como Parquet particionado

#### Notebook 02: EDA (Análisis Exploratorio)

**Objetivos**:
- Entender la distribución de las variables
- Identificar valores nulos y outliers
- Generar estadísticas descriptivas

**Archivo**: [notebooks/02_exploracion_eda.py](notebooks/02_exploracion_eda.py)

**Actividades**:
1. Calcular estadísticas con `describe()`
2. Analizar distribución de `Precio Base` por departamento
3. Identificar contratos con datos faltantes
4. Visualizar top 10 proveedores

---

### Fase 3: Feature Engineering Distribuido (2 horas)

> **Evalúa Sección 13**: Pipelines, VectorAssembler, Transformers

#### Notebook 03: Pipelines y Feature Engineering

**Objetivos**:
- Construir un Pipeline de transformación end-to-end
- Aplicar VectorAssembler para crear vectores de features
- Manejar variables categóricas con StringIndexer y OneHotEncoder

**Archivo**: [notebooks/03_feature_engineering.py](notebooks/03_feature_engineering.py)

**Conceptos clave**:
- **Transformer**: Aplica transformaciones (ej: `StringIndexer`)
- **Estimator**: Aprende de los datos y genera un modelo (ej: `StandardScaler`)
- **Pipeline**: Encadena múltiples stages secuencialmente

**Actividades**:
1. Crear `StringIndexer` para columnas categóricas (Departamento, Tipo Contrato)
2. Aplicar `OneHotEncoder` para generar variables dummy
3. Combinar features numéricas y categóricas con `VectorAssembler`
4. Persistir el pipeline entrenado

#### Notebook 04: Transformaciones Avanzadas

**Objetivos**:
- Escalar features numéricas con StandardScaler
- Aplicar reducción de dimensionalidad con PCA
- Manejar valores nulos con Imputer

**Archivo**: [notebooks/04_transformaciones.py](notebooks/04_transformaciones.py)

**Actividades**:
1. Normalizar `Precio Base` y `Plazo Ejecución`
2. Aplicar PCA para reducir dimensiones
3. Crear un Pipeline completo: Indexing → Encoding → Scaling → PCA

---

### Fase 4: Modelos de Regresión (3 horas)

> **Evalúa Sección 14**: Regresión Lineal, Logística y Regularización

#### Notebook 05: Regresión Lineal

**Objetivos**:
- Entrenar un modelo de regresión lineal para predecir `Precio Base`
- Evaluar con métricas RMSE, MAE, R²
- Interpretar coeficientes del modelo

**Archivo**: [notebooks/05_regresion_lineal.py](notebooks/05_regresion_lineal.py)

**Actividades**:
1. Dividir datos en train/test (70/30)
2. Entrenar `LinearRegression` de Spark ML
3. Evaluar con `RegressionEvaluator`
4. Analizar coeficientes y feature importance

#### Notebook 06: Regresión Logística

**Objetivos**:
- Construir un clasificador de riesgo de incumplimiento
- Evaluar con métricas de clasificación (AUC-ROC, Precision, Recall)
- Analizar la curva ROC

**Archivo**: [notebooks/06_regresion_logistica.py](notebooks/06_regresion_logistica.py)

**Actividades**:
1. Crear variable objetivo binaria: `riesgo_incumplimiento`
2. Entrenar `LogisticRegression`
3. Calcular AUC, F1-Score y Confusion Matrix
4. Ajustar threshold de clasificación

#### Notebook 07: Regularización

**Objetivos**:
- Prevenir overfitting con regularización L1 (Lasso), L2 (Ridge) y ElasticNet
- Comparar desempeño de modelos regularizados
- Seleccionar el mejor λ (lambda)

**Archivo**: [notebooks/07_regularizacion.py](notebooks/07_regularizacion.py)

**Conceptos clave**:
- **Ridge (L2)**: Penaliza suma de cuadrados de coeficientes
- **Lasso (L1)**: Penaliza valor absoluto, realiza selección de features
- **ElasticNet**: Combina L1 y L2 con parámetro `elasticNetParam`

**Actividades**:
1. Entrenar modelos con diferentes valores de `regParam`
2. Comparar RMSE en test set
3. Visualizar impacto de λ en coeficientes
4. Seleccionar mejor modelo

---

### Fase 5: Optimización de Hiperparámetros (2 horas)

> **Evalúa Sección 15**: Cross-Validation, Grid Search, Random Search

#### Notebook 08: Validación Cruzada

**Objetivos**:
- Implementar K-Fold Cross-Validation
- Evitar overfitting en la selección de hiperparámetros
- Entender bias-variance tradeoff

**Archivo**: [notebooks/08_validacion_cruzada.py](notebooks/08_validacion_cruzada.py)

**Actividades**:
1. Configurar `CrossValidator` con k=5
2. Evaluar estabilidad de métricas entre folds
3. Comparar train vs validation error

#### Notebook 09: Optimización de Hiperparámetros

**Objetivos**:
- Aplicar Grid Search para búsqueda exhaustiva
- Aplicar Random Search para espacios grandes
- Comparar eficiencia y resultados

**Archivo**: [notebooks/09_optimizacion_hiperparametros.py](notebooks/09_optimizacion_hiperparametros.py)

**Hiperparámetros a optimizar**:
- `regParam`: [0.01, 0.1, 1.0]
- `elasticNetParam`: [0.0, 0.5, 1.0]
- `maxIter`: [50, 100, 200]

**Actividades**:
1. Crear `ParamGridBuilder` con grid de búsqueda
2. Ejecutar Grid Search con Cross-Validation
3. Implementar Random Search (alternativa)
4. Seleccionar mejor modelo y guardar hiperparámetros

---

### Fase 6: MLOps y Producción (2 horas)

> **Evalúa Sección 16**: Tracking, Registry, Deployment

#### Notebook 10: MLflow Tracking

**Objetivos**:
- Registrar experimentos con métricas, parámetros y artefactos
- Comparar múltiples runs en MLflow UI
- Visualizar curvas de aprendizaje

**Archivo**: [notebooks/10_mlflow_tracking.py](notebooks/10_mlflow_tracking.py)

**Actividades**:
1. Configurar `mlflow.spark.autolog()`
2. Registrar hiperparámetros con `mlflow.log_param()`
3. Registrar métricas con `mlflow.log_metric()`
4. Guardar modelo con `mlflow.spark.log_model()`
5. Comparar experimentos en [http://localhost:5000](http://localhost:5000)

#### Notebook 11: Model Registry

**Objetivos**:
- Versionar modelos en MLflow Model Registry
- Transicionar modelos entre etapas (Staging → Production)
- Implementar versionamiento semántico

**Archivo**: [notebooks/11_model_registry.py](notebooks/11_model_registry.py)

**Actividades**:
1. Registrar modelo en registry: `mlflow.register_model()`
2. Crear versiones (v1, v2, etc.)
3. Promover modelo a `Production`
4. Archivar modelos obsoletos

#### Notebook 12: Inferencia en Producción

**Objetivos**:
- Cargar modelo desde registry
- Realizar predicciones batch sobre datos nuevos
- Simular pipeline de producción

**Archivo**: [notebooks/12_inferencia_produccion.py](notebooks/12_inferencia_produccion.py)

**Actividades**:
1. Cargar modelo desde `models:/nombre_modelo/Production`
2. Aplicar transformaciones del pipeline
3. Generar predicciones sobre nuevos contratos
4. Guardar resultados en `data/processed/predictions/`

---

## Evaluación del Proyecto

El proyecto final se evalúa en función de:

| Criterio | Peso | Descripción |
|----------|------|-------------|
| **Feature Engineering** | 25% | Construcción de pipelines, manejo de categóricas, escalado |
| **Modelos de ML** | 25% | Implementación correcta de regresión lineal/logística con regularización |
| **Optimización** | 20% | Aplicación de Grid Search y Cross-Validation |
| **MLOps** | 20% | Tracking con MLflow, versionamiento, deployment |
| **Documentación** | 10% | Claridad en notebooks, interpretación de resultados |

---

## Retos Adicionales (Opcional)

Para profundizar tu aprendizaje:

1. **Reto 1**: Implementar un modelo de clasificación multiclase para predecir el `Tipo de Contrato`
2. **Reto 2**: Crear un dashboard interactivo con visualizaciones de las predicciones
3. **Reto 3**: Implementar detección de Data Drift entre train y test
4. **Reto 4**: Automatizar el pipeline con Airflow o CI/CD

---

## Recursos Adicionales

### Documentación Oficial
- [Spark MLlib Guide](https://spark.apache.org/docs/latest/ml-guide.html)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Colombia Compra Eficiente - Datos Abiertos](https://www.colombiacompra.gov.co/transparencia/gestion-documental/datos-abiertos)

### Material del Curso
- [Sección 13: Spark ML](https://ustatisticaldatapulse.github.io/diplomado-gestion-datos/seccion_13.html)
- [Sección 14: Regresión](https://ustatisticaldatapulse.github.io/diplomado-gestion-datos/seccion_14.html)
- [Sección 15: Tuning](https://ustatisticaldatapulse.github.io/diplomado-gestion-datos/seccion_15.html)
- [Sección 16: MLOps](https://ustatisticaldatapulse.github.io/diplomado-gestion-datos/seccion_16.html)

---

## Limpieza del Entorno

Para detener y eliminar todos los contenedores:

```bash
docker-compose down
```

Para eliminar también los volúmenes (datos y logs):

```bash
docker-compose down -v
```

Para reiniciar completamente el proyecto:

```bash
rm -rf data/processed/* mlruns/*
docker-compose up --build -d
```

---

## Soporte y Contacto

Si encuentras problemas durante el desarrollo del proyecto:

1. Revisa los logs de Docker: `docker-compose logs -f`
2. Verifica que los servicios estén corriendo: `docker-compose ps`
3. Consulta la documentación oficial de Spark ML y MLflow
4. Contacta al instructor del curso

---

**Última actualización**: Enero 2026
**Versión**: 1.0
**Diplomado en Gestión de Datos - Universidad Santo Tomás**
