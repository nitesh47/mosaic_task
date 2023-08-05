import logging

from src.common.constants import TRAINING_LOGS
from src.training.train import train_distilbert_model


def setup_logging(log_file_path):
    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format="%(asctime)s :: %(levelname)s :: %(message)s",
        datefmt="%Y-%m-%d %H:%M",
    )


def main():
    dataset_path = "data/IMDB_Dataset.csv"
    model_params_path = "config/distilbert.yml"
    artifacts_dir = "model_artifacts"
    log_file_path = f"{artifacts_dir}/logs/{TRAINING_LOGS}"

    setup_logging(log_file_path)

    logger = logging.getLogger(__name__)
    logger.info("Starting training...")

    try:
        train_distilbert_model(
            dataset_path=dataset_path,
            model_params_path=model_params_path,
            artifacts_dir=artifacts_dir,
        )
        logger.info("Training completed successfully.")
    except Exception as e:
        logger.exception("An unexpected exception occurred during training.")
        logger.error(f"Exception message: {str(e)}")


if __name__ == "__main__":
    main()
