import logging
import warnings

from src.common.constants import INFERENCE_LOGS
from src.inference.predict import main

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def setup_logging():
    logging.basicConfig(
        filename=f"model_artifacts/logs/{INFERENCE_LOGS}",
        level=logging.INFO,
        format="%(asctime)s :: %(levelname)s :: %(message)s",
        datefmt="%Y-%m-%d %H:%M",
    )


def inference_parser():
    while True:
        review_text = input("Enter the review text (type 'exit' to quit): ").strip()
        if not review_text:
            print("No review text provided. Please enter a valid review text.")
        elif review_text.lower() == "exit":
            return None
        else:
            return review_text


if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting inference...")

    try:
        while True:
            review_text = inference_parser()

            if review_text is None:
                break
            data = main(
                input_text=review_text,
                model_path="model_artifacts",
                model_params_path="config/distilbert.yml",
                label_encoder_path="model_artifacts/label_encoder.pkl",
            )
            print(data)
            logger.info(f"{review_text}: {data}")
        logger.info("Inference completed successfully.")
    except Exception as e:
        logger.exception("An unexpected exception occurred during inference.")
        logger.error(f"Exception message: {str(e)}")
