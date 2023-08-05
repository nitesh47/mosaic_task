# Mosaic_test

Git Repository for the binary text classification for sentiment analysis.

### Overview

The objective of this project is to perform sentiment analysis on movie reviews using IMDB data. Below is a brief overview of the dataset, the chosen model.

### Data

The dataset consists of IMDB data in CSV format, containing two columns: "review" and "sentiment". The "review" column contains strings, and the "sentiment" column defines the sentiment of the corresponding string as either "Positive" or "Negative". For example, if the review is "The movie was good," the sentiment would be labeled as "Positive". The dataset size is approximately 66 MB.

### Model

For this sentiment analysis task, the selected model is Distilbert, specifically TFDistilBertForSequenceClassification. To classify sentiments, a custom classification layer has been added to the model.


## Structure

The project directory structure is organized as follows:

- **config**: Contains the model configuration file.
- **data**: Holds the CSV file for training and evaluating the model.
- **model_artifacts**: This directory is used to save the model, logs, and metrics.
- **notebook**: Here, you can find notebooks containing experiments, including data analysis, model training, and evaluation on validation and test data.
- **src**: Contains packages, modules, and scripts utilized in the project.

## Brief setup instructions for development

First of all, we need an environment with `python>=3.9`. You can achieve this with `conda`:
### conda

```bash
conda create --name test python==3.9.13
```
```bash
conda activate test
```
### Or
### pyenv
Make sure you have `pyenv` installed:
```bash
pyenv install 3.9.13
```
```bash
pyenv virtualenv 3.9.13 test
pyenv activate test

```

### Run the following commands to installed the required library to run the modules
```bash
pip install poetry
```
```bash
poetry update
```

### To train the model and evaluate the model
```bash
python train.py
```
Once you run the training module, it will save the model artifacts in below way:
- **model_artifacts**

in the model_artifacts, you can find the model, logs, training metrics, validation metrics.


### To run the inference for the user console.
```bash
python inference.py
```


## Pre-commit


### Manual run of `pre-commit`

To run auto-formatters and linters manually:

```
$ pre-commit run -a
```
