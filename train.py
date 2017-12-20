# Spark configuration and packages specification. The dependencies defined in
# this file will be automatically provisioned for each run that uses Spark.

import sys
import os
import argparse

from azureml.logging import get_azureml_logger

# initialize the logger
logger = get_azureml_logger()

# add experiment arguments
parser = argparse.ArgumentParser()
# parser.add_argument('--arg', action='store_true', help='My Arg')
args = parser.parse_args()
print(args)

# This is how you log scalar metrics
# logger.log("MyMetric", value)

# Create the outputs folder - save any outputs you want managed by AzureML here
os.makedirs('./outputs', exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import utils

# Use the Azure Machine Learning data preparation package
from azureml.dataprep import package

# Use the Azure Machine Learning data collector to log various metrics
from azureml.logging import get_azureml_logger
logger = get_azureml_logger()

# This call will load the referenced package and return a DataFrame.
# If run in a PySpark environment, this call returns a
# Spark DataFrame. If not, it will return a Pandas DataFrame.
df = package.run('prep_airline_data.dprep', dataflow_idx=0)

# Remove this line and add code that uses the DataFrame
df.head(10)
