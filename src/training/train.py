import json
import logging
import time
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from tensorflow.python.lib.io import file_io
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification

from src.common.constants import (
    ACCURACY,
    ACTIVATION,
    BATCH_SIZE,
    CALLBACKS,
    CLEANED_FEATURE_NAME,
    DATA,
    EARLY_STOPPING,
    ENCODED_LABEL,
    ENCODING,
    EPOCHS,
    FACTOR,
    FEATURE_NAME,
    INIT_LR,
    INPUT_IDS,
    LABEL_ENCODER_PKL,
    LABEL_NAME,
    MAX_LENGTH,
    METRICS_FULL_REPORT,
    MIN_DELTA,
    MIN_LR,
    MODEL,
    OPTIMIZATION,
    PATIENCE,
    RANDOM_STATE,
    REDUCE_ON_PLATEAU,
    TEST_METRICS_JSON,
    TEST_SIZE,
    TOKENIZER,
    TRAIN,
    TRANSFORMERS,
    VAL_LOSS,
    VAL_METRICS_JSON,
    VAL_SIZE,
    VERBOSE,
)
from src.preprocessing.data_preprocessing import read_csv_files, text_preprocessing

logger = logging.getLogger(__name__)


def apply_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function perform text preprocessing and removes the duplicate
    instances from the dataframe and then shuffling the dataframe
    to ensure randomness.

    :param df: Dataframe of IMDB data
    :return: preprocessed dataframe
    """
    df[CLEANED_FEATURE_NAME] = df[FEATURE_NAME].apply(text_preprocessing)
    df = df.drop_duplicates(subset=CLEANED_FEATURE_NAME, keep="first")
    df = shuffle(df)
    df = df.reset_index(drop=True)
    logger.info(
        f"After removing the duplicate product name, "
        f"the total number of samples is:: {len(df)}; "
    )
    return df


def get_metrics(
    true_values: Iterable[str],
    predicted_values: Iterable[str],
    round_n: int = 3,
) -> Dict[str, float]:
    """
    classification report with accuracy.

    :param true_values: Iterable with the actual values.
    :param predicted_values: Iterable with the predicted values.
    :param round_n: The number after the decimal point.
    :return: The dictionary with the model metrics.
    """
    metrics = dict()

    acc_value = accuracy_score(true_values, predicted_values)
    metrics[ACCURACY] = round(acc_value, round_n)

    f1_macro = f1_score(true_values, predicted_values, average="macro")
    metrics["f1-score macro"] = round(f1_macro, round_n)
    precision_macro = precision_score(true_values, predicted_values, average="macro")
    metrics["Precision macro"] = round(precision_macro, round_n)
    recall_macro = recall_score(true_values, predicted_values, average="macro")
    metrics["Recall macro"] = round(recall_macro, round_n)

    f1_micro = f1_score(true_values, predicted_values, average="micro")
    metrics["f1-score micro"] = round(f1_micro, round_n)
    precision_micro = precision_score(true_values, predicted_values, average="micro")
    metrics["Precision micro"] = round(precision_micro, round_n)
    recall_micro = recall_score(true_values, predicted_values, average="micro")
    metrics["Recall micro"] = round(recall_micro, round_n)

    f1_weighted = f1_score(true_values, predicted_values, average="weighted")
    metrics["f1-score weighted"] = round(f1_weighted, round_n)
    precision_weighted = precision_score(
        true_values, predicted_values, average="weighted"
    )
    metrics["Precision weighted"] = round(precision_weighted, round_n)
    recall_weighted = recall_score(true_values, predicted_values, average="weighted")
    metrics["Recall weighted"] = round(recall_weighted, round_n)

    metrics[METRICS_FULL_REPORT] = classification_report(
        true_values, predicted_values, output_dict=True
    )
    return metrics


def calculate_scores(
    x_val_ids: Tuple[tf.Tensor],
    y_val_true_ids: Iterable[int],
    model: Any,
    target_encoder: LabelEncoder,
) -> Dict[str, float]:
    """
    This function performs various operations, including model
    predictions on validation data and test data, as well as
    calculating metrics.

    :param x_val_ids: Tokenized tensor values obtained from text.
    :param y_val_true_ids: True classes extracted from the data.
    :param model: Finetuned Distilbert model.
    :param target_encoder: Label encoder.
    :return: Metrics in dictionary format.
    """
    prediction = model.predict(x_val_ids)
    y_val_pred_ids = prediction.flatten()
    y_val_pred = np.where(y_val_pred_ids > 0.5, 1, 0)
    y_val_pred = target_encoder.inverse_transform(y_val_pred)
    y_val_true = target_encoder.inverse_transform(y_val_true_ids)
    metrics = get_metrics(y_val_true, y_val_pred)
    return metrics


def save_artifacts(
    dir_name: str,
    model: Any,
    target_encoder: LabelEncoder,
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
) -> None:
    """
    This function is responsible for saving the model-related
    artifacts to the specified directory.

    :param dir_name: The name of the directory where the model artifacts will be saved.
    :param model: The fine-tuned Distilbert model.
    :param target_encoder: The label encoder.
    :param val_metrics: Validation metrics in dictionary format.
    :param test_metrics: Test metrics in dictionary format.
    :return: None.
    """
    # model
    model.save(f"{dir_name}/")
    # label encoder
    joblib.dump(target_encoder, f"{dir_name}/{LABEL_ENCODER_PKL}")

    # metrics
    with open(f"{dir_name}/metrics/{VAL_METRICS_JSON}", "w") as val:
        json.dump(val_metrics, val)

    with open(f"{dir_name}/metrics/{TEST_METRICS_JSON}", "w") as test:
        json.dump(test_metrics, test)


def encode_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    This function conducts label encoding on the target classes
    column of the given dataframe.

    :param df: The dataframe with a column containing target values.
    :return: The updated dataframe with a new column containing the
            label-encoded values.
    """
    logger.info("Perform label encoding.")
    label_encoder = LabelEncoder()
    df[ENCODED_LABEL] = label_encoder.fit_transform(list(df[LABEL_NAME].values))
    logger.info(
        f"The total number of samples: {len(df[ENCODED_LABEL])}; "
        f"classes: {len(label_encoder.classes_)}"
    )
    return df, label_encoder


def train_val_test(
    df: pd.DataFrame, model_params: [str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    This function split the dataframe into three parts Train, val and test.
    :param df: Entire dataset in dataframe
    :param model_params: model parameters
    :return: Tuple(dataframe)
    """
    logger.info("Perform train/validation split.")

    x_rest, x_test = train_test_split(
        df,
        random_state=model_params[RANDOM_STATE],
        test_size=model_params[DATA][TEST_SIZE],
        stratify=list(df[ENCODED_LABEL]),
    )

    x_train, x_val = train_test_split(
        x_rest,
        random_state=model_params[RANDOM_STATE],
        test_size=model_params[DATA][VAL_SIZE],
        stratify=list(x_rest[ENCODED_LABEL]),
    )
    return x_train, x_val, x_test


def batch_encode(
    tokenizer: Any,
    text: List,
    model_params: Dict,
) -> Tuple[tf.Tensor]:
    """
    This function tokenize the dataframe text column and return the tf.Tensor value

    :param tokenizer: Distilbert tokenizer
    :param text: string in the List
    :param model_params: model parameters
    :return: tokenized text in tensor format
    """

    logger.info("Perform batch encoding.")

    encoded_ids = tokenizer(
        text,
        max_length=model_params[ENCODING][MAX_LENGTH],
        padding="max_length",
        truncation=True,
        return_attention_mask=False,
        return_token_type_ids=False,
    )[INPUT_IDS]
    encoded_ids = tf.convert_to_tensor(encoded_ids)
    return encoded_ids


def get_callbacks(callback_params: Dict[str, Any]) -> List[Any]:
    """
    A custom function to provide the needed callbacks
    based on the parameters specified.

    :param callback_params: The dictionary with the callback params.
    :return: The list of Tensorflow callbacks.
    """
    callbacks = []
    if EARLY_STOPPING in callback_params:
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=VAL_LOSS,
            verbose=callback_params[VERBOSE],
            min_delta=callback_params[EARLY_STOPPING][MIN_DELTA],
            patience=callback_params[EARLY_STOPPING][PATIENCE],
            mode="auto",
        )
        callbacks.append(early_stopping)
    if REDUCE_ON_PLATEAU in callback_params:
        reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=VAL_LOSS,
            verbose=callback_params[VERBOSE],
            factor=callback_params[REDUCE_ON_PLATEAU][FACTOR],
            patience=callback_params[REDUCE_ON_PLATEAU][PATIENCE],
            min_lr=callback_params[REDUCE_ON_PLATEAU][MIN_LR],
        )
        callbacks.append(reduce_on_plateau)
    logger.info(f"Callbacks: {callbacks}")
    return callbacks


def get_tokenizer(model_params: Dict) -> Any:
    """
    This function load the distilbert tokenizer
    :param model_params: model parameters
    :return: Distilbert Tokenizer
    """
    logger.info("Download the tokenizer from Huggingface.")
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        model_params[TRANSFORMERS][TOKENIZER]
    )
    return tokenizer


def build_distilbert_model(model_params: Dict[str, Any], n_outputs: int) -> Any:
    """
    building a model of the DistilBERT architecture for the binary
    classification task.

    :param model_params: The specified parameters of the model.
    :param n_outputs: number of classes
    :return: A compiled tf.keras.Model with added classification layers
             on top of the base pre-trained model architecture.
    """

    input_ids_layer = tf.keras.layers.Input(
        shape=(model_params[ENCODING][MAX_LENGTH],), name=INPUT_IDS, dtype="int32"
    )

    transformer = TFDistilBertForSequenceClassification.from_pretrained(
        model_params[TRANSFORMERS][MODEL], num_labels=n_outputs
    )

    X = transformer([input_ids_layer])[0]

    X = tf.keras.layers.Dense(1, activation=model_params[TRANSFORMERS][ACTIVATION])(X)

    model = tf.keras.Model([input_ids_layer], outputs=X)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=model_params[TRAIN][OPTIMIZATION][INIT_LR]
    )
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )

    return model


def train_distilbert_model(
    dataset_path: str,
    model_params_path: str,
    artifacts_dir: str,
) -> None:
    """
    Train a Keras-based DistilBERT model.

    :param dataset_path: The path to the CSV file containing the dataset.
    :param model_params_path: The path to the YAML file containing the model params.
    :param artifacts_dir: The name of the directory to save the model and the logs.
    :return: None.
    """
    logger.info("load model params")
    with file_io.FileIO(model_params_path, "r") as f:
        model_params = yaml.safe_load(f)
    df = read_csv_files(dataset_path)
    df = apply_preprocessing(df)
    df, target_encoder = encode_labels(df)
    n_classes = len(target_encoder.classes_)
    df_train, df_val, df_test = train_val_test(df, model_params)
    tokenizer = get_tokenizer(model_params)

    x_train_ids = batch_encode(
        tokenizer, df_train[CLEANED_FEATURE_NAME].astype(str).to_list(), model_params
    )
    x_val_ids = batch_encode(
        tokenizer, df_val[CLEANED_FEATURE_NAME].astype(str).to_list(), model_params
    )

    logger.info("Load the model.")
    model = build_distilbert_model(model_params, n_classes)

    logger.info("Training...")
    callbacks = get_callbacks(model_params[TRAIN][CALLBACKS])
    train_start = time.time()
    model.fit(
        x=x_train_ids,
        y=df_train[ENCODED_LABEL].to_numpy(),
        epochs=model_params[TRAIN][EPOCHS],
        batch_size=model_params[TRAIN][BATCH_SIZE],
        validation_data=(x_val_ids, df_val[ENCODED_LABEL].to_numpy()),
        callbacks=callbacks,
    )
    train_time_minutes = round((time.time() - train_start) / 60, 2)

    logger.info(
        f"The training has finished, took {train_time_minutes} minutes."
        f"Calculating the metrics."
    )
    logger.info("Metrics calculation")
    val_metrics = calculate_scores(
        x_val_ids, df_val[ENCODED_LABEL].to_numpy(), model, target_encoder
    )

    x_test_ids = batch_encode(
        tokenizer, df_test[CLEANED_FEATURE_NAME].astype(str).to_list(), model_params
    )
    test_metrics = calculate_scores(
        x_test_ids, df_test[ENCODED_LABEL].to_numpy(), model, target_encoder
    )

    logger.info("Saving artifacts locally.")
    save_artifacts(artifacts_dir, model, target_encoder, val_metrics, test_metrics)
    logger.info("The artifacts are successfully saved.")


if __name__ == "__main__":
    train_distilbert_model(
        "//data/IMDB_Dataset.csv",
        "//config/distilbert.yml",
        "//model_artifacts",
    )
