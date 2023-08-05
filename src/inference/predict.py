import logging
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import yaml
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.lib.io import file_io

from src.common.constants import ENCODING, INPUT_IDS, MAX_LENGTH
from src.training.train import batch_encode, get_tokenizer

logger = logging.getLogger(__name__)


def load_model_params(model_params_path: str) -> Dict:
    """
    This function is responsible for loading model parameters from a YAML file.

    :param model_params_path: The path to the model parameter file.
    :return: The model parameters in dictionary format.
    """
    logger.info(f'Read model params file from: "{model_params_path}".')
    with file_io.FileIO(model_params_path, "r") as f:
        model_params = yaml.safe_load(f)
    if not model_params:
        raise Exception(
            f"Failed to load the model parameters file in the"
            f'following path: "{model_params_path}", the file was not found.'
        )
    logger.info(f"The model params:\n{model_params}")
    logger.info("Model parameters loaded successfully.")
    return model_params


def build_model_from_pb(pb_model: Any, model_params: Dict) -> Any:
    """
    This function applies the custom classification layer and returns
    keras based model with custom classification layer.

    :param pb_model: The base Distilbert model.
    :param model_params: The model parameters.
    :return: keras based model with custom classification layer
    """
    input_ids_layer = tf.keras.layers.Input(
        shape=(model_params[ENCODING][MAX_LENGTH],), name=INPUT_IDS, dtype="int32"
    )
    keras_layer = hub.KerasLayer(pb_model, trainable=True)(input_ids_layer)
    model = tf.keras.Model([input_ids_layer], keras_layer)
    return model


def load_model(model_path: str, model_params: Dict) -> Any:
    """
    This function is responsible for loading the Distilbert model.

    :param model_path: The directory where the model is stored.
    :param model_params: The model parameters.
    :return: The loaded model.
    """
    logger.info(f'Loading the Model... The path: "{model_path}"')
    model = tf.saved_model.load(model_path)
    model = build_model_from_pb(model, model_params)
    if not model:
        raise Exception("Failed to load the model")
    logger.info("Model was loaded successfully.")
    return model


def load_label_encoder(label_encoder_path: str) -> LabelEncoder:
    """
    This function is used to load the label encoder for inference.

    :param label_encoder_path: The path where the label encoder is saved.
    :return: The loaded label encoder.
    """
    logger.info(f'Loading the Label Encoder... The path: "{label_encoder_path}"')
    with file_io.FileIO(label_encoder_path, mode="rb") as encoder_file:
        label_encoder = joblib.load(encoder_file)
    if not label_encoder:
        raise Exception("Failed to load the Label Encoder")
    logger.info("Label Encoder was loaded successfully.")
    return label_encoder


def predict(
    input_ids: Tuple[tf.Tensor],
    model: Any,
    target_encoder: LabelEncoder,
) -> Dict[str, float]:
    """
    This function is designed to make predictions on the given
    input during the inference step.

    :param input_ids: Tokenized input in tensor format.
    :param model: Distilbert model.
    :param target_encoder: Label encoder.
    :return: Model predictions.
    """
    prediction = model.predict(input_ids)
    pred_id = prediction.flatten()
    pred = np.where(pred_id > 0.5, 1, 0)
    pred = target_encoder.inverse_transform(pred)
    return pred


def main(
    input_text: str,
    model_path: str,
    model_params_path: str,
    label_encoder_path: str,
) -> None:
    """
    This is the main function used for running inference.

    :param input_text: The input in string format.
    :param model_path: The path to load the model.
    :param model_params_path: The path to load the model parameters.
    :param label_encoder_path: The path to load the label encoder.
    :return: Model prediction.
    """
    model_params = load_model_params(model_params_path)
    model = load_model(model_path, model_params)
    tokenizer = get_tokenizer(model_params)
    label_encoder = load_label_encoder(label_encoder_path)

    logger.info("Enter Input Text")
    text_encoding = batch_encode(tokenizer, [input_text], model_params)
    model_output = predict(
        text_encoding,
        model,
        label_encoder,
    )
    return model_output[0]


if __name__ == "__main__":
    input_text = "There was a movie called json and I dont like that movie."
    data = main(
        input_text,
        "//model_artifacts",
        "//config/distilbert.yml",
        "//model_artifacts/label_encoder.pkl",
    )
