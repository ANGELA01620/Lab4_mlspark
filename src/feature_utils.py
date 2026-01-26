"""
Utilidades para feature engineering
"""

from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler,
    StandardScaler, PCA
)
from pyspark.ml import Pipeline
from typing import List


def create_feature_pipeline(
    categorical_cols: List[str],
    numeric_cols: List[str],
    output_col: str = "features"
):
    """
    Crea un pipeline completo de feature engineering

    Args:
        categorical_cols: Lista de columnas categóricas
        numeric_cols: Lista de columnas numéricas
        output_col: Nombre de la columna de salida

    Returns:
        Pipeline configurado
    """
    # StringIndexers
    indexers = [
        StringIndexer(
            inputCol=col,
            outputCol=col + "_idx",
            handleInvalid="keep"
        )
        for col in categorical_cols
    ]

    # OneHotEncoders
    encoders = [
        OneHotEncoder(
            inputCol=col + "_idx",
            outputCol=col + "_vec"
        )
        for col in categorical_cols
    ]

    # VectorAssembler
    feature_cols = numeric_cols + [col + "_vec" for col in categorical_cols]
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol=output_col
    )

    # Pipeline
    pipeline = Pipeline(stages=indexers + encoders + [assembler])

    return pipeline


def create_scaling_pipeline(
    input_col: str = "features",
    output_col: str = "features_scaled",
    apply_pca: bool = False,
    pca_k: int = 20
):
    """
    Crea un pipeline de escalado y PCA

    Args:
        input_col: Columna de entrada
        output_col: Columna de salida
        apply_pca: Si aplicar PCA
        pca_k: Número de componentes principales

    Returns:
        Pipeline configurado
    """
    stages = []

    # StandardScaler
    scaler = StandardScaler(
        inputCol=input_col,
        outputCol=output_col,
        withMean=False,
        withStd=True
    )
    stages.append(scaler)

    # PCA (opcional)
    if apply_pca:
        pca = PCA(
            k=pca_k,
            inputCol=output_col,
            outputCol="features_pca"
        )
        stages.append(pca)

    pipeline = Pipeline(stages=stages)
    return pipeline
