import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)


def read_csv_files(path: str):
    """
    This function reads csv file.

    :param path: csv file path i.e. /data/imdb_data.csv
    :return: pd.DataFrame
    """
    logger.info("Read csv file")
    data = pd.read_csv(f"{path}")
    return data


def text_preprocessing(text: str) -> str:
    """
    This function cleans text, as it applies several rules

    :param text: review string
    :return: clean text
    """
    regex_subs = [
        (r"=", " "),
        (r"<br />", " "),
        (r"\.+", "."),
        (r"\!+", " "),
        (r"\?+", " "),
        (r"\s+", " "),
    ]
    for pattern, replacement in regex_subs:
        text = re.sub(pattern, replacement, text)
    return text


if __name__ == "__main__":
    data = read_csv_files("//data/IMDB_Dataset.csv")
    data["cleaned_review"] = data["review"].apply(text_preprocessing)
    print(data.head())
