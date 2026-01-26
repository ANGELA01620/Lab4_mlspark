"""
Utilidades para entrenamiento y evaluación de modelos
"""

from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator
from typing import Dict, Tuple


def train_linear_regression(
    train_df,
    features_col: str = "features",
    label_col: str = "label",
    reg_param: float = 0.0,
    elastic_net_param: float = 0.0,
    max_iter: int = 100
):
    """
    Entrena un modelo de regresión lineal

    Args:
        train_df: DataFrame de entrenamiento
        features_col: Nombre de la columna de features
        label_col: Nombre de la columna objetivo
        reg_param: Parámetro de regularización (lambda)
        elastic_net_param: Parámetro de ElasticNet (alpha)
        max_iter: Número máximo de iteraciones

    Returns:
        Modelo entrenado
    """
    lr = LinearRegression(
        featuresCol=features_col,
        labelCol=label_col,
        regParam=reg_param,
        elasticNetParam=elastic_net_param,
        maxIter=max_iter
    )

    model = lr.fit(train_df)
    return model


def evaluate_regression_model(
    model,
    test_df,
    label_col: str = "label",
    prediction_col: str = "prediction"
) -> Dict[str, float]:
    """
    Evalúa un modelo de regresión

    Args:
        model: Modelo entrenado
        test_df: DataFrame de test
        label_col: Columna objetivo
        prediction_col: Columna de predicciones

    Returns:
        Diccionario con métricas
    """
    predictions = model.transform(test_df)

    # RMSE
    evaluator_rmse = RegressionEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col,
        metricName="rmse"
    )
    rmse = evaluator_rmse.evaluate(predictions)

    # MAE
    evaluator_mae = RegressionEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col,
        metricName="mae"
    )
    mae = evaluator_mae.evaluate(predictions)

    # R²
    evaluator_r2 = RegressionEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col,
        metricName="r2"
    )
    r2 = evaluator_r2.evaluate(predictions)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }


def train_logistic_regression(
    train_df,
    features_col: str = "features",
    label_col: str = "label",
    reg_param: float = 0.0,
    max_iter: int = 100
):
    """
    Entrena un modelo de regresión logística

    Args:
        train_df: DataFrame de entrenamiento
        features_col: Nombre de la columna de features
        label_col: Nombre de la columna objetivo
        reg_param: Parámetro de regularización
        max_iter: Número máximo de iteraciones

    Returns:
        Modelo entrenado
    """
    lr = LogisticRegression(
        featuresCol=features_col,
        labelCol=label_col,
        regParam=reg_param,
        maxIter=max_iter
    )

    model = lr.fit(train_df)
    return model


def split_data(
    df,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple:
    """
    Divide un DataFrame en train y test

    Args:
        df: DataFrame a dividir
        train_ratio: Proporción para entrenamiento
        seed: Semilla para reproducibilidad

    Returns:
        Tupla (train_df, test_df)
    """
    train, test = df.randomSplit([train_ratio, 1 - train_ratio], seed=seed)
    return train, test
